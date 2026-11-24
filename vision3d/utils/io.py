import gzip
import os
import os.path as osp
import pickle
from typing import Any

import numpy as np
import skimage.io as io
from numpy import ndarray
from tqdm import tqdm

from .distributed import master_only

# pickle utilities


def load_pickle(filename: str) -> Any:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def load_picklez(filename: str) -> Any:
    with gzip.open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(data: Any, filename: str) -> None:
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def dump_picklez(data: Any, filename: str, compresslevel: int = 5) -> None:
    with gzip.open(filename, "wb", compresslevel=compresslevel) as f:
        pickle.dump(data, f)


# reading / writing


def readlines(filename: str) -> list[str]:
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def writelines(lines: list[str], filename: str, mode: str = "w") -> None:
    lines = [line + "\n" if not line.endswith("\n") else line for line in lines]
    with open(filename, mode) as f:
        f.writelines(lines)


def read_image(filename: str, as_gray: bool = False) -> ndarray:
    image = io.imread(filename, as_gray=as_gray)
    if not as_gray:
        image = image.astype(np.float32) / 255.0
    return image


def read_depth_image(filename: str, scaling_factor: float | None = None) -> ndarray:
    image = io.imread(filename).astype(np.float32)
    if scaling_factor is not None:
        image = image / scaling_factor
    return image


# log utilities


def read_log(line: str) -> dict:
    split_line = line.split(", ")
    data_dict = {}
    metadata = []
    for item in split_line:
        if ": " not in item:
            metadata.append(item)
        else:
            key, value = item.split(": ")
            data_dict[key] = value
    data_dict["metadata"] = metadata
    return data_dict


def read_logs(log_file: str) -> list[dict]:
    lines = readlines(log_file)
    log_dicts = [read_log(line) for line in lines]
    return log_dicts


# directory utilities


@master_only
def ensure_dir(path: str) -> None:
    if not osp.exists(path):
        os.makedirs(path)
    else:
        assert osp.isdir(path), f"'{path}' already exists but is not a directory."


# special functions


def write_correspondences(file_name: str, src_corr_points: ndarray, tgt_corr_points: ndarray) -> None:
    if not file_name.endswith(".obj"):
        file_name += ".obj"

    v_lines = []
    l_lines = []

    num_corrs = src_corr_points.shape[0]
    for i in tqdm(range(num_corrs)):
        n = i * 2

        src_point = src_corr_points[i]
        tgt_point = tgt_corr_points[i]

        line = "v {:.6f} {:.6f} {:.6f}\n".format(src_point[0], src_point[1], src_point[2])
        v_lines.append(line)

        line = "v {:.6f} {:.6f} {:.6f}\n".format(tgt_point[0], tgt_point[1], tgt_point[2])
        v_lines.append(line)

        line = "l {} {}\n".format(n + 1, n + 2)
        l_lines.append(line)

    with open(file_name, "w") as f:
        f.writelines(v_lines)
        f.writelines(l_lines)
