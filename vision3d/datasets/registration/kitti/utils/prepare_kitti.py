import os
import os.path as osp
import shutil

import numpy as np
import open3d as o3d

from vision3d.array_ops.se3 import (
    apply_transform,
    compose_transforms,
    get_transform_from_rotation_translation,
    inverse_transform,
)
from vision3d.utils.io import dump_pickle, ensure_dir, load_pickle
from vision3d.utils.open3d import make_open3d_point_cloud, voxel_down_sample

dataset_root = osp.join(os.environ["HOME"], "workspace", "data", "registration", "odometryKitti")


def downsample_keyframe(phase: str, voxel_size: float = 0.3):
    old_dir = osp.join(dataset_root, "data", phase)
    new_dir = osp.join(dataset_root, "downsampled", phase)
    ensure_dir(new_dir)

    filenames = os.listdir(old_dir)
    filenames = sorted([x for x in filenames if x.endswith(".npy")])
    for filename in filenames:
        old_filename = osp.join(old_dir, filename)
        new_filename = osp.join(new_dir, filename)
        # if osp.exists(new_filename):
        #     print(f"{new_filename} exists")
        #     continue
        # read old data
        points = np.load(old_filename)
        # downsample
        points = voxel_down_sample(points, voxel_size=voxel_size)
        # save new data
        np.save(new_filename, points)
        print(f"save {old_filename} to {new_filename}")


def get_velo2cam():
    rotation = np.array(
        [
            7.533745e-03,
            -9.999714e-01,
            -6.166020e-04,
            1.480249e-02,
            7.280733e-04,
            -9.998902e-01,
            9.998621e-01,
            7.523790e-03,
            1.480755e-02,
        ]
    ).reshape(3, 3)
    translation = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
    velo2cam = get_transform_from_rotation_translation(rotation, translation)
    return velo2cam


def create_metadata(phase: str):
    match phase:
        case "train":
            scene_names = ["00", "01", "02", "03", "04", "05"]
        case "val":
            scene_names = ["06", "07"]
        case "test":
            scene_names = ["08", "09", "10"]
        case _:
            raise ValueError(f"Unknown phase: {phase}")

    velo2cam = get_velo2cam()

    for distance_threshold in [10, 20, 30]:
        intervals = []
        metadata_list = []

        for scene_name in scene_names:
            scene_dir = osp.join(dataset_root, "all_data", scene_name, "velodyne")
            filenames = os.listdir(scene_dir)
            filenames = sorted([x for x in filenames if x.endswith(".bin")])
            frame_ids = [x[:-4] for x in filenames]

            poses = np.genfromtxt(osp.join(dataset_root, "poses", f"{scene_name}.txt")).reshape(-1, 3, 4)
            rotations = poses[:, :, :3]
            translations = poses[:, :, 3]

            n_frames = len(frame_ids)

            s_index = 0
            while s_index < n_frames:
                s_position = translations[s_index]

                found = False
                for offset in range(0, 100):
                    t_index = s_index + offset

                    if t_index + 1 >= n_frames:
                        break

                    t_position = translations[t_index + 1]

                    # skip if less than 10 meters
                    if np.linalg.norm(s_position - t_position) <= distance_threshold:
                        continue

                    # skip same src and tgt
                    if offset == 0:
                        break

                    found = True
                    intervals.append(offset)

                    # add a pair, from s to t
                    src_pose = get_transform_from_rotation_translation(rotations[s_index], translations[s_index])
                    tgt_pose = get_transform_from_rotation_translation(rotations[t_index], translations[t_index])
                    transform = compose_transforms(src_pose, inverse_transform(tgt_pose))

                    src_frame = frame_ids[s_index]
                    tgt_frame = frame_ids[t_index]

                    print(f"{scene_name}: found {src_frame} to {tgt_frame} with offset {offset}")

                    # refine pose with icp
                    src_points = np.fromfile(osp.join(scene_dir, f"{src_frame}.bin"), dtype=np.float32)
                    tgt_points = np.fromfile(osp.join(scene_dir, f"{tgt_frame}.bin"), dtype=np.float32)
                    src_points = src_points.reshape(-1, 4)[:, :3]
                    tgt_points = tgt_points.reshape(-1, 4)[:, :3]
                    alignment = compose_transforms(velo2cam, transform, inverse_transform(velo2cam))
                    src_points = apply_transform(src_points, alignment)
                    src_pcd = make_open3d_point_cloud(src_points)
                    tgt_pcd = make_open3d_point_cloud(tgt_points)
                    result = o3d.pipelines.registration.registration_icp(
                        src_pcd,
                        tgt_pcd,
                        0.2,
                        np.eye(4),
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
                    )
                    transform = compose_transforms(alignment, result.transformation)

                    src_file = osp.join("downsampled", scene_name, f"{src_frame}.npy")
                    tgt_file = osp.join("downsampled", scene_name, f"{tgt_frame}.npy")

                    metadata = {
                        "scene_name": scene_name,
                        "src_frame": frame_ids[s_index],
                        "tgt_frame": frame_ids[t_index],
                        "src_pcd": src_file,
                        "tgt_pcd": tgt_file,
                        "transform": transform,
                        "src_pose": src_pose,
                        "tgt_pose": tgt_pose,
                    }
                    metadata_list.append(metadata)

                    s_index = t_index + 1

                    break

                if not found:
                    s_index += 1

        print(f"mean offset: {np.mean(intervals)}")

        # save data
        print(f"total_data: {len(metadata_list)}")
        dump_pickle(metadata_list, osp.join(dataset_root, "metadata", f"{phase}-v2-{distance_threshold}.pkl"))


def copy_selected_data():
    def copy_file(filename):
        dirname = osp.dirname(filename)
        ensure_dir(osp.join(dataset_root, "data", dirname))
        new_path = osp.join(dataset_root, "data", filename)
        if osp.exists(osp.join(dataset_root, "data", filename)):
            return
        old_path = osp.join(dataset_root, "all_data", filename)
        shutil.copy2(old_path, new_path)

    filenames = os.listdir(osp.join(dataset_root, "metadata"))
    for filename in filenames:
        metadata_list = load_pickle(osp.join(dataset_root, "metadata", filename))
        for metadata in metadata_list:
            src_file = metadata["src_file"]
            tgt_file = metadata["tgt_file"]
            copy_file(src_file)
            copy_file(tgt_file)


if __name__ == "__main__":
    # create_metadata("train")
    # create_metadata("val")
    create_metadata("test")
    # process_keyframe()
    # copy_selected_data()
    # downsample_keyframe("train")
