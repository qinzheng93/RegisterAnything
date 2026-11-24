import os
import os.path as osp
import shutil

from tqdm import tqdm

from vision3d.datasets.registration.threedmatch.threedmatch_utils import read_log_file
from vision3d.utils.io import dump_pickle, ensure_dir


def copy_data(src_path: str, tgt_path: str):
    if osp.exists(tgt_path):
        return
    shutil.copy2(src_path, tgt_path)


def read_overlap_file(file_path: str) -> dict:
    overlap_dict = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    for line in lines:
        split_line = line.split()
        src_frame = int(split_line[0])
        tgt_frame = int(split_line[1])
        overlap = float(split_line[2])
        overlap_dict[f"{src_frame}-{tgt_frame}"] = overlap
    return overlap_dict


def create_metadata():
    workspace = osp.join(os.environ["HOME"], "workspace")
    dataset_dir = osp.join(workspace, "raw_data", "BUFFER-X", "Scannetpp_iphone", "test")
    data_dir = osp.join(workspace, "data", "registration", "ScanNet++i", "data")
    metadata_dir = osp.join(workspace, "data", "registration", "ScanNet++i", "metadata")

    ensure_dir(data_dir)
    ensure_dir(metadata_dir)

    metadata_list = []

    scene_names = sorted(os.listdir(dataset_dir))
    for scene_name in scene_names:
        print(f"Processing {scene_name}...")
        scene_dir = osp.join(dataset_dir, scene_name, "iphone")

        new_scene_dir = osp.join(data_dir, scene_name)
        ensure_dir(new_scene_dir)

        test_dict_list = read_log_file(osp.join(scene_dir, "gt.log"))

        overlap_dict = read_overlap_file(osp.join(scene_dir, "overlap_ratio.txt"))

        print(f"log: {len(test_dict_list)}, overlap: {len(overlap_dict)}")

        for test_dict in tqdm(test_dict_list):
            src_frame, tgt_frame = test_dict["test_pair"]
            transform = test_dict["transform"]
            overlap = overlap_dict[f"{src_frame}-{tgt_frame}"]

            src_filename = f"cloud_bin_{src_frame}.ply"
            tgt_filename = f"cloud_bin_{tgt_frame}.ply"

            # copy data
            copy_data(osp.join(scene_dir, "tsdf", src_filename), osp.join(new_scene_dir, src_filename))
            copy_data(osp.join(scene_dir, "tsdf", tgt_filename), osp.join(new_scene_dir, tgt_filename))

            # create metadata
            metadata = {
                "scene_name": scene_name,
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_pcd": osp.join(scene_name, src_filename),
                "tgt_pcd": osp.join(scene_name, tgt_filename),
                "transform": transform,
                "overlap": overlap,
            }

            metadata_list.append(metadata)

    print(f"Total {len(metadata_list)} pairs.")
    dump_pickle(metadata_list, osp.join(metadata_dir, "test.pkl"))


if __name__ == "__main__":
    create_metadata()
