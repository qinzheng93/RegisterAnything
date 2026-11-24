import os
import os.path as osp

import torch
from torch import GradScaler
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

from vision3d.utils.distributed import master_only
from vision3d.utils.logger import logger


def load_state_dict(model, state_dict: dict, strict: bool = False) -> None:
    if isinstance(model, DistributedDataParallel):
        model = model.module

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if len(missing_keys) > 0:
        logger.warning(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    if strict and (len(missing_keys) != 0 or len(unexpected_keys) != 0):
        raise RuntimeError("The keys in model and state_dict do not match.")


class CheckpointManager:
    """Checkpoint Manager.

    Note:
        If the program stopped during validation, the last epoch will be ignored during model selection. Resuming
        training will fix this.
    """

    def __init__(self, checkpoint_dir: str, save_n_latest_models: int = -1):
        self.checkpoint_dir = checkpoint_dir
        self.save_n_latest_models = save_n_latest_models
        self.saved_checkpoints = []
        self.best_scores = {}

    def state_dict(self) -> dict:
        return {"saved_checkpoints": self.saved_checkpoints, "best_scores": self.best_scores}

    def load_state_dict(self, state_dict: dict) -> None:
        self.saved_checkpoints = state_dict["saved_checkpoints"]
        self.best_scores = state_dict["best_scores"]

    @master_only
    def save_checkpoint(
        self,
        filename: str,
        metadata: dict,
        model: Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: GradScaler | None = None,
        ema_model: AveragedModel | None = None,
        save_ckpt_manager: bool = False,
        clean_checkpoints: bool = True,
    ) -> None:
        # handle DDP model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        state_dict = {"metadata": metadata, "model": model.state_dict()}

        # ema_model
        if ema_model is not None:
            state_dict["ema_model"] = ema_model.state_dict()

        # optimizer, scheduler
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state_dict["scheduler"] = scheduler.state_dict()

        # gradient scaler
        if scaler is not None:
            state_dict["scaler"] = scaler.state_dict()

        # checkpoint manager
        if save_ckpt_manager:
            state_dict["ckpt_manager"] = self.state_dict()

        # NOTE: save only model weights and metadata
        filename = osp.join(self.checkpoint_dir, filename)
        torch.save(state_dict, filename)
        logger.info("Checkpoint saved to '{}'.".format(filename))

        # only track epoch checkpoints
        basename = osp.basename(filename)
        if basename.startswith("epoch") and clean_checkpoints:
            self.saved_checkpoints.append(filename)
            self.clean_checkpoints()

    def load_checkpoint(
        self,
        filename: str,
        model: Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: GradScaler | None = None,
        ema_model: AveragedModel | None = None,
    ) -> dict:
        logger.info("Loading checkpoint from '{}'.".format(filename))

        state_dict = torch.load(filename, map_location=torch.device("cpu"), weights_only=False)

        # load metadata
        if "metadata" in state_dict:
            metadata = state_dict["metadata"]
        else:
            # Legacy: for old snapshots only
            metadata = {}
            if "epoch" in state_dict:
                metadata["epoch"] = state_dict["epoch"]
            if "iteration" in state_dict:
                metadata["total_steps"] = state_dict["iteration"]

        # load model
        if isinstance(model, DistributedDataParallel):
            model = model.module
        load_state_dict(model, state_dict["model"])
        logger.info("Model loaded.")

        # load optimizer
        if optimizer is not None:
            if "optimizer" not in state_dict:
                logger.warning(f"Optimizer checkpoint not found in '{filename}'.")
            else:
                optimizer.load_state_dict(state_dict["optimizer"])
                logger.info("Optimizer loaded.")

        # load scheduler
        if scheduler is not None:
            if "scheduler" not in state_dict:
                logger.warning(f"Scheduler checkpoint not found in '{filename}'.")
            else:
                scheduler.load_state_dict(state_dict["scheduler"])
                logger.info("Scheduler loaded.")

        # load scaler
        if scaler is not None:
            if "scaler" not in state_dict:
                logger.warning(f"Scaler checkpoint not found in '{filename}'.")
            else:
                scaler.load_state_dict(state_dict["scaler"])
                logger.info("Scaler loaded.")

        # load EMA model
        if ema_model is not None:
            if "ema_model" not in state_dict:
                logger.warning(f"EMA model checkpoint not found in '{filename}'.")
            else:
                ema_model.load_state_dict(state_dict["ema_model"])
                logger.info("EMA model loaded.")

        # load checkpoint manager
        if "ckpt_manager" in state_dict:
            self.load_state_dict(state_dict["ckpt_manager"])
            logger.info("CheckpointManager loaded.")

        return metadata

    @master_only
    def clean_checkpoints(self) -> None:
        # save all models
        if self.save_n_latest_models <= 0:
            return

        # clean old checkpoints
        while len(self.saved_checkpoints) > self.save_n_latest_models:
            oldest_checkpoint = self.saved_checkpoints.pop(0)
            if osp.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)

    def add_metric(self, metric: str, largest: bool = True) -> None:
        assert metric not in self.best_scores, f"Duplicate metric: {metric}"
        self.best_scores[metric] = {"score": None, "largest": largest}

    def compare_metric(self, metric: str, score: float) -> bool:
        best_score = self.best_scores[metric]["score"]
        if best_score is None:
            return True
        if self.best_scores[metric]["largest"]:
            return score > best_score
        else:
            return score < best_score

    @master_only
    def update_best_model(self, metric_dict: dict, metadata: dict, model: Module) -> None:
        for metric in self.best_scores:
            if metric not in metric_dict:
                logger.warning(f"Metric '{metric}' not found (available: {metric_dict.keys()}).")
                continue
            cur_score = metric_dict[metric]
            if self.compare_metric(metric, cur_score):
                self.best_scores[metric]["score"] = cur_score
                self.save_checkpoint(f"best_{metric}.pth", metadata, model)
                logger.info(f"New best '{metric}': {cur_score:.3f}.")
