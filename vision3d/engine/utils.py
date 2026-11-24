import random

import numpy as np
import torch
from torch.backends import cudnn as cudnn


def setup_engine(seed: int | None = None, cudnn_deterministic: bool = True, detect_anomaly: bool = False) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    torch.autograd.set_detect_anomaly(detect_anomaly)
