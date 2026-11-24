import os
import os.path as osp
import shutil

import numpy as np
import open3d as o3d
from tqdm import tqdm

from vision3d.array_ops.se3 import compose_transforms, get_transform_from_rotation_translation, inverse_transform
from vision3d.utils.io import dump_pickle, ensure_dir


def downsample_points():
    workspace = osp.join(os.environ["HOME"], "workspace")
    dataset_dir = osp.join(workspace, "data", "registration", "MIT")
    data_dir = osp.join(dataset_dir, "data")
    downsampled_dir = osp.join(data_dir, "downsampled")
    ensure_dir(downsampled_dir)

    scene_names = sorted(os.listdir(dataset_dir))
    for scene_name in scene_names:
        scene_dir = osp.join(dataset_dir, scene_name)
        new_scene_dir = osp.join(downsampled_dir, scene_name)
        ensure_dir(new_scene_dir)

        filenames = sorted(os.listdir(scene_dir))
        for filename in tqdm(filenames):
            pcd = o3d.io.read_point_cloud(osp.join(scene_dir, filename))
            pcd = pcd.voxel_down_sample(0.3)
            points = np.asarray(pcd.points)
            new_filename = filename.replace("pcd", "npy")
            np.save(osp.join(new_scene_dir, new_filename), points)


def generate_metadata():
    distance_threshold = 5.0

    workspace = osp.join(os.environ["HOME"], "workspace")
    dataset_dir = osp.join(workspace, "raw_data", "BUFFER-X", "newer-college")
    data_dir = osp.join(workspace, "data", "registration", "Oxford", "data")
    metadata_dir = osp.join(workspace, "data", "registration", "Oxford", "metadata")
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

        pose_file = osp.join(scene_dir, "poses_kitti.txt")
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
                    transform = compose_transforms(src_pose, inverse_transform(tgt_pose))

                    src_frame = frame_ids[src_index]
                    tgt_frame = frame_ids[tgt_index]

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
                    shutil.copy2(osp.join(scan_dir, f"{src_frame}.pcd"), osp.join(data_dir, metadata["src_pcd"]))
                    shutil.copy2(osp.join(scan_dir, f"{tgt_frame}.pcd"), osp.join(data_dir, metadata["tgt_pcd"]))

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
    # downsample_points()
