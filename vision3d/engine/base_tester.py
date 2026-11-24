import json
import os.path as osp
import sys
import time
from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from torch.utils.data import DataLoader

from vision3d.utils.logger import logger
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import get_default_parser

from .checkpoint_manager import load_state_dict
from .utils import setup_engine


class BaseTester(ABC):
    model: Module
    test_loader: DataLoader

    def __init__(self, cfg: DictConfig):
        # parser
        parser = get_default_parser()
        self.args = parser.parse_args()

        # cuda check
        assert torch.cuda.is_available(), "No CUDA devices available."

        # logger
        self.log_file = osp.join(cfg.exp.log_dir, "test-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
        logger.initialize(log_file=self.log_file)

        # find checkpoint
        self.checkpoint = self.args.checkpoint
        if self.checkpoint is None and self.args.test_epoch is not None:
            self.checkpoint = f"epoch-{self.args.test_epoch}.pth"
        if self.checkpoint is not None:
            self.checkpoint = osp.join(cfg.exp.checkpoint_dir, self.checkpoint)

        # print config
        logger.info(f"Command: {sys.argv}")
        logger.info("Args:\n" + json.dumps(vars(self.args), indent=4))
        logger.info("Configs:\n" + json.dumps(OmegaConf.to_object(cfg), indent=4))

        # initialize
        setup_engine(seed=cfg.exp.seed, cudnn_deterministic=self.args.cudnn_deterministic)

        # state
        self.iteration = None

    def load(self, filename: str, strict: bool = True) -> None:
        logger.info('Loading from "{}".'.format(filename))
        state_dict = torch.load(filename, map_location=torch.device("cpu"), weights_only=False)
        assert "model" in state_dict, "No model can be loaded."
        load_state_dict(self.model, state_dict["model"], strict=strict)
        logger.info("Model has been loaded.")
        if "metadata" in state_dict:
            epoch = state_dict["metadata"]["epoch"]
            total_steps = state_dict["metadata"]["total_steps"]
            logger.info(f"Checkpoint metadata: epoch: {epoch}, total_steps: {total_steps}.")

    def register_model(self, model: Module) -> Module:
        """Register model."""
        model = model.cuda()
        self.model = model
        message = "Model description:\n" + str(model)
        logger.info(message)
        return model

    def register_loader(self, test_loader: DataLoader) -> None:
        """Register data loader."""
        self.test_loader = test_loader

    def before_test_epoch(self) -> None:
        pass

    def before_test_step(self, iteration: int, data_dict: dict) -> None:
        pass

    @abstractmethod
    def test_step(self, iteration: int, data_dict: dict) -> dict:
        pass

    @abstractmethod
    def eval_step(self, iteration: int, data_dict: dict, output_dict: dict) -> dict:
        pass

    def after_test_step(self, iteration: int, data_dict: dict, output_dict: dict, result_dict: dict) -> None:
        pass

    def after_test_epoch(self, summary_dict: dict) -> None:
        pass

    def get_log_string(self, iteration: int, data_dict: dict, output_dict: dict, result_dict: dict) -> str:
        return get_log_string(result_dict)

    @abstractmethod
    def test_epoch(self) -> None:
        pass

    def run(self, strict_loading: bool = True) -> None:
        assert hasattr(self, "test_loader"), "testloader is not registered."
        assert hasattr(self, "model"), "model is not registered."

        if self.checkpoint is not None:
            self.load(self.checkpoint, strict=strict_loading)
        else:
            logger.warning("Checkpoint is not specified.")

        self.model.eval()
        torch.set_grad_enabled(False)

        self.test_epoch()
