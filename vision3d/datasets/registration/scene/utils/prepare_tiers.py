import os
import os.path as osp

import numpy as np
import open3d as o3d
from tqdm import tqdm

from vision3d.array_ops.se3 import compose_transforms, get_transform_from_rotation_translation, inverse_transform
from vision3d.utils.io import dump_pickle, ensure_dir


def downsample_points():
    workspace = osp.join(os.environ["HOME"], "workspace")
    dataset_dir = osp.join(workspace, "raw_data", "tiers_indoor")
    downsampled_dir = osp.join(workspace, "data", "registration", "TIERS", "downsampled")
    ensure_dir(downsampled_dir)

    scene_names = sorted(os.listdir(dataset_dir))
    for scene_name in scene_names:
        scene_dir = osp.join(dataset_dir, scene_name)
        sensor_types = sorted(os.listdir(scene_dir))
        for sensor_type in sensor_types:
            sensor_dir = osp.join(scene_dir, sensor_type)
            new_sensor_dir = osp.join(downsampled_dir, scene_name, sensor_type)
            ensure_dir(new_sensor_dir)

            scan_dir = osp.join(sensor_dir, "scans")
            filenames = sorted(os.listdir(scan_dir))
            for filename in tqdm(filenames):
                pcd = o3d.io.read_point_cloud(osp.join(scan_dir, filename))
                pcd = pcd.voxel_down_sample(0.025)
                points = np.asarray(pcd.points)
                new_filename = filename.replace("pcd", "npy")
                np.save(osp.join(downsampled_dir, scene_name, sensor_type, new_filename), points)


def generate_metadata():
    distance_threshold = 2.0

    workspace = osp.join(os.environ["HOME"], "workspace")
    dataset_dir = osp.join(workspace, "raw_data", "tiers_indoor")
    metadata_dir = osp.join(workspace, "data", "registration", "TIERS", "metadata")
    ensure_dir(metadata_dir)

    metadata_list = []

    scene_names = sorted(os.listdir(dataset_dir))
    for scene_name in scene_names:
        scene_dir = osp.join(dataset_dir, scene_name)
        sensor_types = sorted(os.listdir(scene_dir))
        for sensor_type in sensor_types:
            print(f"Processing {scene_name}--{sensor_type}")

            sensor_dir = osp.join(scene_dir, sensor_type)
            pose_file = osp.join(sensor_dir, "poses_kitti.txt")
            poses = np.genfromtxt(pose_file).reshape(-1, 3, 4)
            rotations = poses[:, :3, :3]
            positions = poses[:, :3, 3]

            scan_dir = osp.join(sensor_dir, "scans")
            frame_ids = [x[:-4] for x in sorted(os.listdir(scan_dir))]

            num_frames = poses.shape[0]

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
                            "sensor_type": sensor_type,
                            "src_frame": src_frame,
                            "tgt_frame": tgt_frame,
                            "src_path": osp.join(scene_name, sensor_type, f"{src_frame}.npy"),
                            "tgt_path": osp.join(scene_name, sensor_type, f"{tgt_frame}.npy"),
                            "src_pcd": osp.join(scene_name, sensor_type, "scans", f"{src_frame}.pcd"),
                            "tgt_pcd": osp.join(scene_name, sensor_type, "scans", f"{tgt_frame}.pcd"),
                            "src_pose": src_pose,
                            "tgt_pose": tgt_pose,
                            "transform": transform,
                        }
                        metadata_list.append(metadata)

                        found = True
                        src_index = tgt_index + 1

                        break

                    tgt_index += 1

                if not found:
                    src_index += 1

    print(f"{len(metadata_list)} pairs found.")
    dump_pickle(metadata_list, osp.join(metadata_dir, "test.pkl"))


if __name__ == "__main__":
    # downsample_points()
    generate_metadata()
