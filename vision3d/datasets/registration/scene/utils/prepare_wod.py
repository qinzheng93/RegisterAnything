import os
import os.path as osp

import numpy as np
import open3d as o3d

from vision3d.array_ops.se3 import get_transform_from_rotation_translation
from vision3d.utils.io import dump_pickle, ensure_dir
from vision3d.utils.open3d import make_open3d_point_cloud


def save_to_pcd(bin_path, pcd_path):
    if osp.exists(pcd_path):
        return

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
    pcd = make_open3d_point_cloud(points)
    o3d.io.write_point_cloud(pcd_path, pcd)


def generate_metadata():
    distance_threshold = 10.0

    workspace = osp.join(os.environ["HOME"], "workspace")
    dataset_dir = osp.join(workspace, "raw_data", "BUFFER-X", "WOD", "test", "sequences")
    icp_dir = osp.join(workspace, "raw_data", "BUFFER-X", "WOD", "test", "icp")
    data_dir = osp.join(workspace, "data", "registration", "WOD", "data")
    metadata_dir = osp.join(workspace, "data", "registration", "WOD", "metadata")
    ensure_dir(data_dir)
    ensure_dir(metadata_dir)

    metadata_list = []

    scene_names = sorted(os.listdir(dataset_dir))
    scene_names = [x for x in scene_names if osp.isdir(osp.join(dataset_dir, x))]
    for scene_name in scene_names:
        print(f"Processing {scene_name}")

        scene_dir = osp.join(dataset_dir, scene_name)
        new_scene_dir = osp.join(data_dir, scene_name)
        ensure_dir(new_scene_dir)

        pose_file = osp.join(scene_dir, "poses.txt")
        poses = np.genfromtxt(pose_file).reshape(-1, 3, 4)
        rotations = poses[:, :3, :3]
        positions = poses[:, :3, 3]

        scan_dir = osp.join(scene_dir, "scans")
        frame_ids = [x[:-4] for x in sorted(os.listdir(scan_dir))]

        num_frames = poses.shape[0]

        print(num_frames)

        src_index = 0
        while src_index < num_frames:
            src_position = positions[src_index]

            found = False
            tgt_index = src_index + 1
            while tgt_index < num_frames and tgt_index < src_index + 100:
                tgt_position = positions[tgt_index]

                if np.linalg.norm(src_position - tgt_position) > distance_threshold:
                    # select the frame BEFORE the first frame which is at least 2 meters away from the src frame
                    tgt_index -= 1
                    src_pose = get_transform_from_rotation_translation(rotations[src_index], positions[src_index])
                    tgt_pose = get_transform_from_rotation_translation(rotations[tgt_index], positions[tgt_index])

                    src_frame = frame_ids[src_index]
                    tgt_frame = frame_ids[tgt_index]

                    icp_pose_path = osp.join(icp_dir, f"{scene_name}_{int(src_frame)}_{int(tgt_frame)}.npy")
                    transform = np.load(icp_pose_path)

                    metadata = {
                        "scene_name": scene_name,
                        "src_frame": src_frame,
                        "tgt_frame": tgt_frame,
                        "src_pcd": osp.join(scene_name, f"{src_frame}.pcd"),
                        "tgt_pcd": osp.join(scene_name, f"{tgt_frame}.pcd"),
                        "src_pose": src_pose,
                        "tgt_pose": tgt_pose,
                        "transform": transform,
                    }
                    metadata_list.append(metadata)

                    # copy
                    save_to_pcd(osp.join(scan_dir, f"{src_frame}.bin"), osp.join(data_dir, metadata["src_pcd"]))
                    save_to_pcd(osp.join(scan_dir, f"{tgt_frame}.bin"), osp.join(data_dir, metadata["tgt_pcd"]))

                    found = True
                    src_index = tgt_index + 1

                    break

                tgt_index += 1

            if not found:
                src_index += 1

    print(f"{len(metadata_list)} pairs found.")
    dump_pickle(metadata_list, osp.join(metadata_dir, "test.pkl"))


if __name__ == "__main__":
    generate_metadata()
