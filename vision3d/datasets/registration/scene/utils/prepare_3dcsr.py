import os
import os.path as osp

import numpy as np

from vision3d.utils.io import dump_pickle

dataset_root = osp.join(os.environ["HOME"], "workspace/data/registration/3DCSR")
metadata_dir = osp.join(dataset_root, "metadata")


def prepare_lidar():
    data_dir = osp.join(dataset_root, "data", "kinect_lidar")
    scene_names = sorted(os.listdir(data_dir), key=lambda x: int(x[5:]))
    print(f"Total scenes: {len(scene_names)}")

    metadata_list = []

    except_list = [
        "scene16/pair1",
        "scene16/pair7",
        "scene16/pair5",
        "scene18/pair4",
        "scene18/pair9",
        "scene18/pair5",
        "scene10/pair1",
        "scene13/pair5",
        "scene14/pair6",
        "scene15/pair9",
        "scene19/pair5",
    ]

    for scene_name in scene_names:
        scene_dir = osp.join(data_dir, scene_name)

        pair_names = sorted(os.listdir(scene_dir), key=lambda x: int(x[4:]))
        for pair_name in pair_names:
            pair_dir = osp.join(scene_dir, pair_name)

            src_pcd = osp.join("kinect_lidar", scene_name, pair_name, "kinect.ply")
            tgt_pcd = osp.join("kinect_lidar", scene_name, pair_name, "lidar.ply")
            transform = np.loadtxt(osp.join(pair_dir, "T_gt.txt"))
            transform = np.linalg.inv(transform)

            tag = f"{scene_name}/{pair_name}"

            if tag in except_list:
                continue

            metadata = {
                "scene_name": scene_name,
                "pair_name": pair_name,
                "src_pcd": src_pcd,
                "tgt_pcd": tgt_pcd,
                "transform": transform,
            }

            metadata_list.append(metadata)

    print(f"Total pairs: {len(metadata_list)}")

    dump_pickle(metadata_list, osp.join(metadata_dir, "kinect_lidar_inv.pkl"))


def prepare_sfm():
    data_dir = osp.join(dataset_root, "data", "kinect_sfm")

    level_names = ["easy", "hard"]

    metadata_list = []

    except_list = ["hard/labcorner", "hard/lab6", "hard/lab4", "easy/duck", "easy/duckpart"]

    for level_name in level_names:
        level_dir = osp.join(data_dir, level_name)
        scene_names = sorted(os.listdir(level_dir))
        print(f"Total scenes: {len(scene_names)}")

        for scene_name in scene_names:
            scene_dir = osp.join(level_dir, scene_name)

            src_pcd = osp.join("kinect_sfm", level_name, scene_name, "kinect.ply")
            tgt_pcd = osp.join("kinect_sfm", level_name, scene_name, "sfm.ply")
            transform = np.loadtxt(osp.join(scene_dir, "T_gt.txt"))
            transform = np.linalg.inv(transform)

            tag = f"{level_name}/{scene_name}"

            if tag in except_list:
                continue

            metadata = {
                "level_name": level_name,
                "scene_name": scene_name,
                "src_pcd": src_pcd,
                "tgt_pcd": tgt_pcd,
                "transform": transform,
            }

            metadata_list.append(metadata)

    print(f"Total pairs: {len(metadata_list)}")

    dump_pickle(metadata_list, osp.join(metadata_dir, "kinect_sfm_inv.pkl"))


if __name__ == "__main__":
    prepare_lidar()
    prepare_sfm()
