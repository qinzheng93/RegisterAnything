from argparse import ArgumentParser, Namespace

_PARSER = None


def get_default_parser() -> ArgumentParser:
    global _PARSER
    if _PARSER is None:
        _PARSER = ArgumentParser()
    return _PARSER


def parse_args() -> Namespace:
    parser = get_default_parser()
    args = parser.parse_args()
    return args


def add_trainer_args() -> None:
    parser = get_default_parser()
    parser.add_argument_group("trainer", "trainer arguments")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from the latest checkpoint",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="logging steps",
    )
    parser.add_argument(
        "--grad_check",
        action="store_true",
        help="check whether the gradients contain inf or nan",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="detect anomaly with autograd",
    )
    parser.add_argument(
        "--save_latest_n_models",
        type=int,
        default=-1,
        help="save latest n models",
    )
    parser.add_argument(
        "--no_cudnn_deterministic",
        action="store_false",
        dest="cudnn_deterministic",
        help="disable cudnn deterministic method",
    )
    parser.add_argument(
        "--no_swanlab",
        action="store_false",
        dest="use_swanlab",
        help="disable swanlab",
    )
    parser.add_argument(
        "--swanlab_id",
        default=None,
        help="resume from swanlab id",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="debug mode (implement debug method for the trainer)",
    )
    parser.add_argument(
        "--only_validation",
        action="store_true",
        help="skip training, only perform validation (for debug use)",
    )


def add_tester_args() -> None:
    parser = get_default_parser()
    parser.add_argument_group("tester", "tester arguments")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="load from checkpoint",
    )
    parser.add_argument(
        "--test_epoch",
        type=int,
        default=None,
        help="test epoch",
    )
    parser.add_argument(
        "--no_cudnn_deterministic",
        action="store_false",
        dest="cudnn_deterministic",
        help="disable cudnn deterministic method",
    )
