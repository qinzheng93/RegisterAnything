from collections.abc import Iterator

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from vision3d.utils.logger import logger
from vision3d.utils.misc import get_log_string
from vision3d.utils.summary_board import SummaryBoard
from vision3d.utils.tensor import move_to_cuda
from vision3d.utils.timer import Timer

from .base_trainer import BaseTrainer
from .context_manager import clear_context_manager


class CycleLoader:
    """CycleLoader for IterBasedTrainer.

    Note: CycleLoader cannot be perfectly resumed as the internal epoch and the training epoch are not the same with
          the training epoch, so we just set its epoch to the resumed training epoch.
    """

    def __init__(self, data_loader: DataLoader, epoch: int):
        self.data_loader = data_loader
        self.last_epoch = epoch
        self.iterator = self.initialize_iterator()

    def initialize_iterator(self) -> Iterator[dict]:
        if hasattr(self.data_loader.sampler, "set_epoch"):
            self.data_loader.sampler.set_epoch(self.last_epoch + 1)  # type: ignore
        return iter(self.data_loader)

    def __next__(self) -> dict:
        try:
            data_dict = next(self.iterator)
        except StopIteration:
            self.last_epoch += 1
            self.iterator = self.initialize_iterator()
            data_dict = next(self.iterator)
        return data_dict


class IterBasedTrainer(BaseTrainer):
    """Iteration-based Trainer.

    The training lasts for `cfg.trainer.max_epoch` epochs, and each epoch contains
    `cfg.trainer.num_iters_per_epoch` iterations. The training dataloader is automatically
    restarted during training.

    The learning rate is decayed after each iteration or pseudo batch.

    Training pipeline:
        1. before_train_epoch
        2. for each iteration:
            2.1 before_train_step
            2.2 train_step
            2.3 after_backward
            2.4 optimizer_step
            2.5 after_train_step
            2.6 log iteration
            2.7 scheduler_step
        3. after_train_epoch
    """

    cycle_loader: CycleLoader

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.num_iters_per_epoch = cfg.trainer.num_iters_per_epoch

    def before_train(self) -> None:
        """Before training function (create CycleLoader)."""
        self.cycle_loader = CycleLoader(self.train_loader, self.epoch)

    def train_epoch(self) -> None:
        """Training epoch."""
        # before train epoch
        self.before_train_epoch(self.epoch)
        # setup watcher
        summary_board = SummaryBoard(auto_register=True)
        timer = Timer()
        # train loop
        self.optimizer.zero_grad()
        timer.tic("data")
        for batch_index in range(self.num_iters_per_epoch):
            clear_context_manager()
            # load data
            self.iteration = batch_index + 1
            self.total_steps += 1
            data_dict = next(self.cycle_loader)
            data_dict = move_to_cuda(data_dict)
            self.before_train_step(self.epoch, self.iteration, data_dict)
            timer.toc("data")
            # forward
            timer.tic("model")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                output_dict, result_dict = self.train_step(self.epoch, self.iteration, data_dict)
            if "loss" in result_dict:
                # backward
                self.scaler.scale(result_dict["loss"]).backward()
                self.after_backward(self.epoch, self.iteration, data_dict, output_dict, result_dict)
                self.check_gradients(self.epoch, self.iteration, data_dict, output_dict, result_dict)
                # optimization
                if self.iteration % self.grad_acc_steps == 0:
                    self.optimizer_step()
            timer.toc("model")
            # after training
            timer.tic("data")
            self.after_train_step(self.epoch, self.iteration, data_dict, output_dict, result_dict)
            result_dict = self.unpack_tensors(result_dict)
            summary_board.update_from_dict(result_dict)
            # logging
            if self.iteration % self.log_steps == 0:
                summary_dict = summary_board.summary(last_n=self.log_steps)
                message = get_log_string(
                    summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.iteration,
                    max_iteration=self.num_iters_per_epoch,
                    lr=self.get_lr(),
                    time_dict=timer.summary(keys=["data", "model"]),
                )
                logger.info(message)
                self.write_event("train", summary_dict, self.total_steps)
            # scheduler
            if self.iteration % self.grad_acc_steps == 0:
                self.scheduler_step()
            # empty cache
            torch.cuda.empty_cache()
        # after train epoch
        summary_dict = summary_board.summary()
        self.after_train_epoch(self.epoch, summary_dict)
