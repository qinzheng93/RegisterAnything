import warnings
from collections.abc import Callable
from functools import wraps

from .average_meter import AverageMeter
from .misc import get_format_strings


def check_meter(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, key: str, *args, **kwargs):
        assert key in self.meter_dict, f"No meter for key '{key}'."
        return func(self, key, *args, **kwargs)

    return wrapper


class SummaryBoard:
    """Summary board."""

    def __init__(self, keys: list[str] | None = None, auto_register: bool = False):
        """Instantiate a SummaryBoard.

        Args:
            keys (list[str]=None): create AverageMeter with the keys.
            auto_register (bool=False): whether register basic meters automatically on the fly.
        """
        self.meter_dict: dict[str, AverageMeter] = {}
        self.auto_register = auto_register

        if keys is not None:
            self.register_all(keys)

    def register_meter(self, key: str) -> None:
        """Register a meter for key."""
        assert key not in self.meter_dict, f"Duplicated key '{key}' in SummaryBoard."
        self.meter_dict[key] = AverageMeter()

    def register_all(self, keys: list[str]) -> None:
        """Register a list of meters for the keys."""
        for key in keys:
            self.register_meter(key)

    def reset_meter(self, key: str) -> None:
        """Reset the meter for the key."""
        if key in self.meter_dict:
            self.meter_dict[key].reset()

    def reset_all(self) -> None:
        """Reset all meters."""
        for key in self.meter_dict:
            self.reset_meter(key)

    def update(self, key: str, value: float) -> None:
        """Update the AverageMeter for the kay with the value."""
        if key not in self.meter_dict and self.auto_register:
            self.register_meter(key)
        if key in self.meter_dict:
            self.meter_dict[key].update(value)
        else:
            warnings.warn(f"[SummaryBoard] No meter for key '{key}' to update. Skipped.")

    def update_from_dict(self, metric_dict: dict) -> None:
        """Update the AverageMeter for every (key, value) pair in the metric_dict."""
        for key, value in metric_dict.items():
            self.update(key, value)

    @check_meter
    def sum(self, key: str, last_n: int | None = None) -> float:
        """Return the sum of the meter for the key (only the last n records are used)."""
        return self.meter_dict[key].sum(last_n=last_n)

    @check_meter
    def mean(self, key: str, last_n: int | None = None) -> float:
        """Return the mean value of the meter for the key (only the last n records are used)."""
        return self.meter_dict[key].mean(last_n=last_n)

    @check_meter
    def std(self, key: str, last_n: int | None = None) -> float:
        """Return the standard variance of the meter for the key (only the last n records are used)."""
        return self.meter_dict[key].std(last_n=last_n)

    @check_meter
    def median(self, key: str, last_n: int | None = None) -> float:
        """Return the median value of the meter for the key (only the last n records are used)."""
        return self.meter_dict[key].median(last_n=last_n)

    def tostring(self, keys: list[str] | None = None, last_n: int | None = None) -> str:
        """Return the summary string for the keys (only the last n records are used)."""
        summary_dict = self.summary(keys, last_n=last_n)
        log_strings = get_format_strings(summary_dict)
        summary = ", ".join(log_strings)
        return summary

    def summary(self, keys: list[str] | None = None, last_n: int | None = None) -> dict:
        """Return the summary dict for the keys (only the last n records are used)."""
        if keys is None:
            keys = list(self.meter_dict.keys())
        summary_dict = {key: self.meter_dict[key].mean(last_n=last_n) for key in keys}
        return summary_dict
