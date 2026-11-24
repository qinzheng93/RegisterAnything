import os
import os.path as osp

import numpy as np

from vision3d.utils.io import dump_pickle, ensure_dir


def create_metadata():
    workspace = osp.join(os.environ["HOME"], "workspace")
    dataset_dir = osp.join(workspace, "data", "registration", "ETH")

    metadata_list = []
    scene_names = ["gazebo_summer", "gazebo_winter", "wood_autmn", "wood_summer"]
    for scene_name in scene_names:
        scene_dir = osp.join(dataset_dir, "data", scene_name)
        with open(osp.join(scene_dir, "gt.log"), "r") as f:
            lines = f.readlines()
        while True:
            line = lines.pop(0)
            tgt_frame, src_frame, _ = [int(x) for x in line.split()]
            src_path = osp.join(scene_name, f"Hokuyo_{src_frame}.ply")
            tgt_path = osp.join(scene_name, f"Hokuyo_{tgt_frame}.ply")
            transform = []
            for _ in range(4):
                line = lines.pop(0)
                transform.append([float(x) for x in line.split()])
            transform = np.array(transform).reshape(4, 4)
            metadata = {
                "scene_name": scene_name,
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_pcd": src_path,
                "tgt_pcd": tgt_path,
                "transform": transform,
            }
            if not lines:
                break
            metadata_list.append(metadata)
    ensure_dir(osp.join(dataset_dir, "metadata"))
    dump_pickle(metadata_list, osp.join(dataset_dir, "metadata", "test.pkl"))


if __name__ == "__main__":
    create_metadata()
