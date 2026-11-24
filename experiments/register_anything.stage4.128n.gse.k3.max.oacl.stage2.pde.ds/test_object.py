from omegaconf import DictConfig

from vision3d.engine import SampleTester
from vision3d.utils.logger import logger
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args, get_default_parser
from vision3d.utils.profiling import profile_cpu_runtime

# isort: split
from config import make_cfg
from dataset import test_data_loader
from loss import EvalFunction
from model import create_model


def add_custom_args():
    parser = get_default_parser()
    parser.add_argument_group("experiment", "experiment arguments")
    parser.add_argument("--benchmark", choices=["ModelNet", "ShapeNet"], help="test benchmark")


class Tester(SampleTester):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # dataloader
        with profile_cpu_runtime("Data loader created"):
            data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        logger.info(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.eval_func = EvalFunction(cfg).cuda()

    def test_step(self, iteration: int, data_dict: dict) -> dict:
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration: int, data_dict: dict, output_dict: dict) -> dict:
        result_dict = self.eval_func(data_dict, output_dict)
        result_dict["scale"] = data_dict["scale"]
        return result_dict

    def get_log_string(self, iteration: int, data_dict: dict, output_dict: dict, result_dict: dict) -> str:
        message = get_log_string(result_dict=result_dict)
        message += ", nCorr: {}".format(output_dict["corr_scores"].shape[0])
        return message


def main():
    add_tester_args()
    add_custom_args()
    cfg = make_cfg()
    cfg.model.coarse_matching.num_correspondences = 128
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
