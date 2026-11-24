import importlib
from collections.abc import Callable
from functools import wraps
from types import ModuleType

# deprecate decorator


def deprecated(replacement: str) -> Callable:
    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            raise RuntimeError(f"{func.__name__} is deprecated. Use {replacement} instead.")

        return wrapper

    return decorate


# importlib utilities


def load_ext(name: str, functions: list[str]) -> ModuleType:
    import torch  # handle libc10.so missing
    ext_module = importlib.import_module(name)
    for function in functions:
        assert hasattr(ext_module, function), f"Function '{function}' missing in '{name}'."
    return ext_module


# log utilities


def get_print_format(value: float | int | str) -> str:
    if isinstance(value, (int, str)):
        return ""
    if value == 0:
        return ".3f"
    if value < 1e-5:
        return ".3e"
    if value < 1e-2:
        return ".6f"
    return ".3f"


def get_format_strings(result_dict: dict) -> list[str]:
    """Get format string for a list of key-value pairs."""
    format_strings = []
    if "metadata" in result_dict:
        # handle special key "metadata"
        format_strings.append(result_dict["metadata"])
    for key, value in result_dict.items():
        if key == "metadata":
            continue
        format_string = f"{key}: {value:{get_print_format(value)}}"
        format_strings.append(format_string)
    return format_strings


def get_log_string(
    result_dict: dict,
    epoch: int | None = None,
    max_epoch: int | None = None,
    iteration: int | None = None,
    max_iteration: int | None = None,
    lr: float | None = None,
    time_dict: dict | None = None,
) -> str:
    log_strings = []
    if epoch is not None:
        epoch_string = f"epoch: {epoch}"
        if max_epoch is not None:
            epoch_string += f"/{max_epoch}"
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f"iter: {iteration}"
        if max_iteration is not None:
            iter_string += f"/{max_iteration}"
        log_strings.append(iter_string)
    log_strings += get_format_strings(result_dict)
    if lr is not None:
        log_strings.append("lr: {:.3e}".format(lr))
    if time_dict is not None:
        time_string = "time: " + "/".join([f"{time_dict[key]:.3f}s" for key in time_dict])
        log_strings.append(time_string)
    message = ", ".join(log_strings)
    return message
