import time

import torch
from torch.nn import Module

from .logger import logger


class CudaRuntimeProfiler:
    def __init__(self, description: str):
        self.description = description
        self.last_time = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.last_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        duration = time.time() - self.last_time
        logger.info(f"{self.description}: {duration:.3f}s")


class CpuRuntimeProfiler:
    def __init__(self, description: str):
        self.description = description
        self.last_time = 0.0

    def __enter__(self):
        self.last_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.time() - self.last_time
        logger.info(f"{self.description}: {duration:.3f}s")


def profile_cpu_runtime(description: str) -> CpuRuntimeProfiler:
    """Profile CPU runtime.

    Usage:

    ```
    with profile_cpu_runtime("call function"):
        outputs = function_call(inputs)
    ```

    Args:
        description (str): The description string for the code.

    Returns:
        A CpuRuntimeProfiler object.
    """
    return CpuRuntimeProfiler(description)


def profile_cuda_runtime(description: str) -> CudaRuntimeProfiler:
    """Profile Cuda runtime.

    Usage:

    ```
    with profile_cuda_runtime("call function"):
        outputs = function_call(inputs)
    ```

    The CUDA device is synchronized during profiling.

    Args:
        description (str): The description string for the code.

    Returns:
        A CudaRuntimeProfiler object.
    """
    return CudaRuntimeProfiler(description)


def count_model_parameters(model: Module) -> float:
    """Count model parameters.

    Args:
        model (Module): The model to count parameters.

    Returns:
        Float: The number of parameters in million.
    """
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params / 1e6
