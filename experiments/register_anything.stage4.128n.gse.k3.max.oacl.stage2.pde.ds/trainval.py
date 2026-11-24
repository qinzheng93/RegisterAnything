from omegaconf import DictConfig

from vision3d.engine import IterBasedTrainer
from vision3d.utils.logger import logger
from vision3d.utils.optimizer import build_optimizer, build_scheduler
from vision3d.utils.parser import add_trainer_args
from vision3d.utils.profiling import profile_cpu_runtime

# isort: split
from config import make_cfg
from dataset import train_valid_data_loader
from loss import EvalFunction, OverallLoss
from model import create_model


class Trainer(IterBasedTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # dataloader
        with profile_cpu_runtime("Data loader create"):
            train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg)
        logger.info(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(train_loader, val_loader)

        # model
        model = create_model(cfg)
        model = self.register_model(model)

        # optimizer, scheduler
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg)
        self.eval_func = EvalFunction(cfg)

    def train_step(self, epoch: int, iteration: int, data_dict: dict) -> tuple[dict, dict]:
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch: int, iteration: int, data_dict: dict) -> tuple[dict, dict]:
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    add_trainer_args()
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
