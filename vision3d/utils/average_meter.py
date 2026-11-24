import numpy as np


class AverageMeter:
    def __init__(self):
        self.records: list[float] = []

    def update(self, result: float) -> None:
        self.records.append(result)

    def reset(self) -> None:
        self.records.clear()

    def get_records(self, last_n: int | None = None) -> list[float]:
        if last_n is not None:
            return self.records[-last_n:]
        return self.records

    def sum(self, last_n: int | None = None) -> float:
        records = self.get_records(last_n=last_n)
        return np.sum(records).item()

    def mean(self, last_n: int | None = None) -> float:
        records = self.get_records(last_n=last_n)
        return np.mean(records).item()

    def std(self, last_n: int | None = None) -> float:
        records = self.get_records(last_n=last_n)
        return np.std(records).item()

    def median(self, last_n: int | None = None) -> float:
        records = self.get_records(last_n=last_n)
        return np.median(records).item()
