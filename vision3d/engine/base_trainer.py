import json
import os.path as osp
import time
from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig, OmegaConf
from torch import GradScaler
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision3d.utils.distributed import (
    all_reduce_tensors,
    get_local_rank,
    get_world_size,
    is_distributed,
    master_only,
    setup_distributed,
)
from vision3d.utils.io import ensure_dir
from vision3d.utils.logger import logger
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import get_default_parser
from vision3d.utils.summary_board import SummaryBoard
from vision3d.utils.tensor import move_to_cuda, tensor_to_array
from vision3d.utils.timer import Timer
from vision3d.utils.watchboard import get_watchboard

from .checkpoint_manager import CheckpointManager
from .context_manager import clear_context_manager
from .utils import setup_engine


class BaseTrainer(ABC):
    """Base Trainer.

    1. Usages:
        1.1 Use 'register_model' to add the model to the trainer.
        1.2 Use 'register_optimizer' to add the optimizer to the trainer.
        1.3 Use 'register_scheduler' to add the scheduler to the trainer.
        1.4 Use 'register_loader' to add the training and validation data loaders to the trainer.
        1.6 Use 'save_best_model_on_metric' to save best models during training.
        1.7 The 'train_step' and 'val_step' methods must be implemented. Other methods can be optionally implemented.

    2. Pseudo Batch:
        Use `cfg.trainer.grad_acc_steps` pseudo batching. Every `cfg.trainer.grad_acc_steps` iterations
        are used as a pseudo batch. And the gradients in a pseudo batch are accumulated during backward propagation.
        Note that the gradients are NOT averaged in the pseudo batch.

    3. About Multi-GPU training.
        Multi-GPU training is supported by `DistributedDataParallel` and the trainer automatically enables it when
        launched with `torchrun`. Only one GPU (cuda:0) will be used otherwise even if there are multiple GPUs. Note
        that the learning rate is scaled by the world size by default.

    4. Validation pipeline:
        1. before_val_epoch
        2. for each iteration:
            2.1 before_val_step
            2.2 val_step
            2.3 after_val_step
            2.4 log iteration
        3. summary logging
        4. update best models
        5. after_val_epoch

    5. Checkpointing:
        5.1 After each training epoch, the metadata (epoch, total_steps) and the weights are saved to 'epoch-N.pth'.
        5.2 After each validation epoch, the metadata, the weights and the training states (optimizer, scheduler and
            checkpoint manager) are saved to 'checkpoint.pth'. If the training is interrupted, use '--resume' to resume
            training from the last finished validation epoch.
    """

    model: Module
    optimizer: Optimizer
    scheduler: LRScheduler
    train_loader: DataLoader
    val_loader: DataLoader
    ema_model: AveragedModel

    def __init__(self, cfg: DictConfig):
        # parser
        parser = get_default_parser()
        self.args = parser.parse_args()
        self.grad_check = self.args.grad_check
        self.log_steps = self.args.log_steps
        self.debug_mode = self.args.debug_mode
        self.only_validation = self.args.only_validation

        # cuda check
        assert torch.cuda.is_available(), "No CUDA devices available."

        # distributed data parallel
        setup_distributed()
        self.is_distributed = is_distributed()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()

        # parse config
        self.cfg = cfg
        self.output_dir = cfg.exp.output_dir
        self.checkpoint_dir = cfg.exp.checkpoint_dir
        self.log_dir = cfg.exp.log_dir
        self.manual_seed = cfg.exp.seed + self.local_rank
        self.max_epoch = cfg.trainer.max_epoch
        self.grad_acc_steps = cfg.trainer.grad_acc_steps

        # logger
        self.log_file = osp.join(self.log_dir, "train-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
        logger.initialize(log_file=self.log_file)

        # watchboard
        self.watchboard = get_watchboard(cfg, use_swanlab=self.args.use_swanlab, id=self.args.swanlab_id)
        if self.watchboard is not None:
            logger.info(f"Swanlab is enabled. Log saved to {self.watchboard.logdir}.")

        # print info
        logger.info("Args:\n" + json.dumps(vars(self.args), indent=4))
        logger.info("Configs:\n" + json.dumps(OmegaConf.to_object(cfg), indent=4))
        if self.is_distributed:
            logger.info(f"Using DistributedDataParallel mode (world_size: {self.world_size})")
        else:
            logger.info("Using Single-GPU mode.")

        # initialization
        setup_engine(
            seed=self.manual_seed,
            cudnn_deterministic=self.args.cudnn_deterministic,
            detect_anomaly=self.args.detect_anomaly,
        )

        # checkpoint manager
        self.ckptmgr = CheckpointManager(self.checkpoint_dir, save_n_latest_models=self.args.save_latest_n_models)

        # state
        self.epoch: int = 0
        self.iteration: int = 0
        self.total_steps: int = 0

        # EMA
        self.use_ema = cfg.ema.use_ema
        self.ema_cfg = cfg.ema

        # AMP
        self.use_amp = cfg.trainer.use_amp
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # find checkpoint
        self.checkpoint: str | None = None
        self.ema_checkpoint: str | None = None
        if self.args.resume:
            self.checkpoint = osp.join(self.checkpoint_dir, "checkpoint.pth")
            if self.use_ema:
                self.ema_checkpoint = osp.join(self.checkpoint_dir, "ema_model.pth")

    @property
    def metadata(self) -> dict:
        """Metadata (epoch, total_steps)."""
        metadata = {"epoch": self.epoch, "total_steps": self.total_steps}
        return metadata

    @master_only
    def save(self, filename: str, save_training_states: bool = False, save_ema: bool = False) -> None:
        """Save checkpoint."""
        if save_ema:
            # save EMA model, ignore optimizer and scheduler
            self.ckptmgr.save_checkpoint(
                filename,
                self.metadata,
                self.ema_model.module,
                optimizer=None,
                scheduler=None,
                save_ckpt_manager=False,
                clean_checkpoints=False,
            )
            return

        if not save_training_states:
            # save model only, used after each training epoch
            self.ckptmgr.save_checkpoint(
                filename,
                self.metadata,
                self.model,
                optimizer=None,
                scheduler=None,
                save_ckpt_manager=False,
                clean_checkpoints=True,
            )
            return

        # save checkpoint with optimizer, scheduler and ckpt_manager, used after each validation epoch
        self.ckptmgr.save_checkpoint(
            filename,
            self.metadata,
            self.model,
            optimizer=self.optimizer,
            scheduler=getattr(self, "scheduler", None),
            ema_model=getattr(self, "ema_model", None),
            scaler=self.scaler,
            save_ckpt_manager=True,
            clean_checkpoints=False,
        )

    def load(self, filename: str) -> None:
        """Load checkpoint."""
        # load model, optimizer and scheduler
        if self.use_ema:
            metadata = self.ckptmgr.load_checkpoint(
                filename,
                self.model,
                optimizer=self.optimizer,
                scheduler=getattr(self, "scheduler", None),
                ema_model=self.ema_model,
            )
        else:
            metadata = self.ckptmgr.load_checkpoint(
                filename,
                self.model,
                optimizer=self.optimizer,
                scheduler=getattr(self, "scheduler", None),
            )
        # Load metadata (epoch, total_steps)
        self.epoch = metadata["epoch"]
        self.total_steps = metadata["total_steps"]
        logger.info(f"Checkpoint metadata: epoch: {self.epoch}, total_steps: {self.total_steps}.")

    def update_best_models(self, metric_dict: dict) -> None:
        """Update best models."""
        self.ckptmgr.update_best_model(metric_dict, self.metadata, self.model)

    def save_best_model_on_metric(self, metric: str, largest: bool = True) -> None:
        """Set metric to save best model."""
        self.ckptmgr.add_metric(metric, largest=largest)

    @master_only
    def setup_ema_model(self) -> None:
        """Setup EMA model. EMA model is maintained only on the master node."""
        if not self.use_ema:
            return

        model = self.model

        if model is None:
            logger.warning("Model is None. EMA model is not created.")
            return

        # handle DDP model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        self.ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(self.ema_cfg.decay), use_buffers=self.ema_cfg.use_buffers
        )

    @master_only
    def update_ema_model(self) -> None:
        """Update EMA model."""
        if not self.use_ema:
            return

        if self.epoch < self.ema_cfg.start_epoch:
            return

        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        self.ema_model.update_parameters(model)
        self.save("ema_model.pth", save_training_states=False, save_ema=True)

    def register_model(self, model: Module) -> Module:
        """Register model. DDP is automatically used."""
        model = model.cuda()
        # Create DDP
        if self.is_distributed:
            model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.model = model
        logger.info("Model description:\n" + str(model))
        # Create EMA
        self.setup_ema_model()
        return model

    def register_optimizer(self, optimizer: Optimizer) -> None:
        """Register optimizer. DDP is automatically used."""
        if self.is_distributed:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * self.world_size
        self.optimizer = optimizer

    def register_scheduler(self, scheduler: LRScheduler | None) -> None:
        """Register LR scheduler."""
        if scheduler is None:
            return
        self.scheduler = scheduler

    def register_loader(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Register data loader."""
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_lr(self) -> float:
        """Get the learning rate for the first param group."""
        return self.optimizer.param_groups[0]["lr"]

    def set_train_mode(self) -> None:
        """Set training mode."""
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self) -> None:
        """Set evaluation mode."""
        self.model.eval()
        torch.set_grad_enabled(False)

    def write_event(self, phase: str, event_dict: dict, index: int) -> None:
        """Write event to watchboard."""
        if self.watchboard is None:
            return
        for key, value in event_dict.items():
            self.watchboard.log(f"{phase}/{key}", value, index)

    def unpack_tensors(self, result_dict: dict) -> dict:
        """All reduce and release tensors."""
        if self.is_distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
        result_dict = tensor_to_array(result_dict)
        return result_dict

    def optimizer_step(self) -> None:
        """Optimizer step."""
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def scheduler_step(self) -> None:
        """LR Scheduler step."""
        if hasattr(self, "scheduler"):
            self.scheduler.step()

    def before_train(self) -> None:
        """Before training function."""
        pass

    def before_train_epoch(self, epoch: int) -> None:
        """Before training epoch function."""
        pass

    def before_train_step(self, epoch: int, iteration: int, data_dict: dict) -> None:
        """Before training step function."""
        pass

    @abstractmethod
    def train_step(self, epoch: int, iteration: int, data_dict: dict) -> tuple[dict, dict]:
        """Training step. Must be implemented."""
        pass

    def after_backward(
        self,
        epoch: int,
        iteration: int,
        data_dict: dict,
        output_dict: dict,
        result_dict: dict,
    ) -> None:
        """After backward propagation function."""
        pass

    def after_train_step(
        self,
        epoch: int,
        iteration: int,
        data_dict: dict,
        output_dict: dict,
        result_dict: dict,
    ) -> None:
        """After training step function."""
        pass

    def after_train_epoch(self, epoch: int, summary_dict: dict) -> None:
        """After training epoch function."""
        pass

    def before_val_epoch(self, epoch: int) -> None:
        """Before validation epoch function."""
        pass

    def before_val_step(self, epoch: int, iteration: int, data_dict: dict) -> None:
        """Before validation step function."""
        pass

    @abstractmethod
    def val_step(self, epoch: int, iteration: int, data_dict: dict) -> tuple[dict, dict]:
        """Validation step. Must be implemented."""
        pass

    def after_val_step(
        self,
        epoch: int,
        iteration: int,
        data_dict: dict,
        output_dict: dict,
        result_dict: dict,
    ) -> None:
        """After validation step function."""
        pass

    def after_val_epoch(self, epoch: int, summary_dict: dict) -> None:
        """After validation epoch function."""
        pass

    def check_gradients(
        self,
        epoch: int,
        iteration: int,
        data_dict: dict,
        output_dict: dict,
        result_dict: dict,
    ) -> None:
        """Check if any gradients are NaN or Inf."""
        if not self.grad_check:
            return

        all_gradients_ok = True
        for param in self.model.parameters():
            if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                all_gradients_ok = False
                break

        if all_gradients_ok:
            return

        logger.error("Epoch: {}, iter: {}, check gradients failed.".format(epoch, iteration))
        debug_dir = osp.join(self.output_dir, "debug")
        ensure_dir(debug_dir)
        data_file = osp.join(debug_dir, "data.pth")
        torch.save(data_dict, data_file)
        logger.error(f"Data dict saved to '{data_file}'.")
        model_file = osp.join(debug_dir, "model.pth")
        torch.save(self.model, model_file)
        logger.error(f"Model object saved to '{model_file}'.")
        output_file = osp.join(debug_dir, "output.pth")
        torch.save(output_dict, output_file)
        logger.error(f"Output object saved to '{output_file}'.")
        result_file = osp.join(debug_dir, "result.pth")
        torch.save(result_dict, result_file)
        logger.error(f"Result object saved to '{result_file}'.")

        breakpoint()

    @abstractmethod
    def train_epoch(self) -> None:
        """Training epoch."""
        pass

    def val_epoch(self) -> None:
        """Validation epoch."""
        # before val epoch
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        # setup watcher
        summary_board = SummaryBoard(auto_register=True)
        timer = Timer()
        # val loop
        max_iteration = len(self.val_loader)
        pbar = tqdm(enumerate(self.val_loader), total=max_iteration, disable=self.local_rank != 0)
        timer.tic("data")
        for iteration, data_dict in pbar:
            clear_context_manager()
            # load data
            self.iteration = iteration + 1
            data_dict = move_to_cuda(data_dict)
            self.before_val_step(self.epoch, self.iteration, data_dict)
            timer.toc("data")
            # val step
            torch.cuda.synchronize()
            timer.tic("model")
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                output_dict, result_dict = self.val_step(self.epoch, self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.toc("model")
            # after val step
            timer.tic("data")
            self.after_val_step(self.epoch, self.iteration, data_dict, output_dict, result_dict)
            result_dict = self.unpack_tensors(result_dict)
            summary_board.update_from_dict(result_dict)
            # logging
            message = get_log_string(
                summary_board.summary(),
                epoch=self.epoch,
                max_epoch=self.max_epoch,
                iteration=self.iteration,
                max_iteration=max_iteration,
                time_dict=timer.summary(keys=["data", "model"]),
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
        # summary logging
        summary_dict = summary_board.summary()
        message = get_log_string(summary_dict, epoch=self.epoch, time_dict=timer.summary(keys=["data", "model"]))
        logger.success("[Val] " + message)
        self.write_event("val", summary_dict, self.epoch)
        # after val epoch
        self.after_val_epoch(self.epoch, summary_dict)
        self.set_train_mode()
        # update model
        self.update_best_models(summary_dict)

    def debug(self) -> None:
        pass

    def run(self) -> None:
        """Run trainer."""
        # before train check
        assert hasattr(self, "train_loader"), "No training loader found."
        assert hasattr(self, "val_loader"), "No validation loader found."
        assert len(self.train_loader) > 0, "Training loader is empty."
        assert len(self.val_loader) > 0, "Validation loader is empty."
        assert hasattr(self, "model"), "No model found."
        assert hasattr(self, "optimizer"), "No optimizer found."
        if not hasattr(self, "scheduler"):
            logger.warning("LRScheduler is not set.")

        # load checkpoint
        if self.checkpoint is not None:
            self.load(self.checkpoint)

        # before train
        self.set_train_mode()
        self.before_train()

        # debug mode
        if self.debug_mode:
            self.debug()

        # run training
        while self.epoch < self.max_epoch:
            self.epoch += 1

            if not self.only_validation:
                self.train_epoch()
                self.save(f"epoch-{self.epoch}.pth", save_training_states=False)

                # EMA update
                self.update_ema_model()

            self.val_epoch()

            self.save("checkpoint.pth", save_training_states=True)

            # early return
            if self.only_validation:
                return

        # EMA update bn
        if not self.use_ema:
            return

        if self.ema_cfg.update_bn:
            torch.optim.swa_utils.update_bn(self.train_loader, self.ema_model)
            self.save("ema-update-bn.pth", save_training_states=False)
