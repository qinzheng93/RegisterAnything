import os.path as osp
import uuid

import swanlab

from .distributed import is_master


class Watchboard:
    def __init__(self, cfg, id: str | None = None):
        self.project = cfg.exp.project
        self.description = cfg.exp.name
        self.experiment_name = str(uuid.uuid5(uuid.NAMESPACE_URL, cfg.exp.name))
        self.logdir = osp.join(cfg.exp.event_dir, "swanlog")
        self.id = id

        swanlab.init(
            project=self.project,
            experiment_name=self.experiment_name,
            description=self.description,
            config=cfg,
            logdir=self.logdir,
            resume="allow",
            id=self.id,  # type: ignore
        )

    def log(self, phase: str, value: float, index: int) -> None:
        swanlab.log({phase: value}, step=index)


def get_watchboard(cfg, use_swanlab: bool = True, id: str | None = None) -> Watchboard | None:
    # ONLY use watchboard on master process
    if not is_master():
        return None

    # Skip if watchboard is not enabled
    if not use_swanlab:
        return None

    return Watchboard(cfg, id)
