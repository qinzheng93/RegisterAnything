import time


class Timer:
    def __init__(self):
        self.total_time = {}
        self.count_time = {}
        self.last_time = {}
        self.keys: list[str] = []

    def register_timer(self, key: str) -> None:
        self.total_time[key] = 0.0
        self.count_time[key] = 0
        self.last_time[key] = None
        self.keys.append(key)

    def tic(self, key: str) -> None:
        if key not in self.keys:
            self.register_timer(key)
        self.last_time[key] = time.time()

    def toc(self, key: str) -> None:
        assert key in self.keys
        last_time = self.last_time[key]
        assert last_time is not None, "'tic' must be called before 'toc'."
        duration = time.time() - last_time
        self.total_time[key] += duration
        self.count_time[key] += 1
        self.last_time[key] = None

    def get_time(self, key: str) -> float:
        assert key in self.keys
        return self.total_time[key] / (self.count_time[key] + 1e-12)

    def tostring(self, keys: list[str] | None = None, verbose: bool = True) -> str:
        if keys is None:
            keys = self.keys
        if verbose:
            log_strings = [f"{key}: {self.get_time(key):.3f}s" for key in keys if key in self.keys]
            format_string = ", ".join(log_strings)
        else:
            log_strings = [f"{self.get_time(key):.3f}s" for key in keys if key in self.keys]
            format_string = "time: " + "/".join(log_strings)
        return format_string

    def summary(self, keys: list[str] | None = None) -> dict:
        if keys is None:
            keys = self.keys
        summary_dict = {key: self.get_time(key) for key in keys}
        return summary_dict
