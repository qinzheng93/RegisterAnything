import json
import os
import os.path as osp
import shutil

import numpy as np

from vision3d.array_ops.se3 import (
    compose_transforms,
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
)
from vision3d.array_ops.so3 import quaternion_to_rotation_matrix
from vision3d.utils.io import ensure_dir, load_pickle
from vision3d.utils.open3d import voxel_down_sample

dataset_root = osp.join(os.getcwd(), "raw_data")
phase = "trainval"
distance_threshold = 10.0
data_root = osp.join(dataset_root, phase, f"v1.0-{phase}")

output_root = os.getcwd()
ensure_dir(osp.join(output_root, "all_data"))
ensure_dir(osp.join(output_root, "data"))


def read_pose():
    # load pose data
    pose_metadata_file = osp.join(data_root, "ego_pose.json")
    with open(pose_metadata_file, "r") as f:
        pose_metadata_list = json.load(f)
        print("Read ego_pose.json finish.")

    pose_dict = {}
    for pose_metadata in pose_metadata_list:
        token = pose_metadata["token"]
        pose_dict[token] = pose_metadata

    return pose_dict


def read_calibrated_sensor():
    # load pose data
    sensor_metadata_file = osp.join(data_root, "calibrated_sensor.json")
    with open(sensor_metadata_file, "r") as f:
        sensor_metadata_list = json.load(f)
        print("Read calibrated_sensor.json finish.")

    sensor_dict = {}
    for sensor_metadata in sensor_metadata_list:
        token = sensor_metadata["token"]
        sensor_dict[token] = sensor_metadata

    return sensor_dict


def read_metadata():
    pose_dict = read_pose()

    sensor_dict = read_calibrated_sensor()

    # load frame data
    frame_metadata_file = osp.join(data_root, "sample_data.json")
    with open(frame_metadata_file, "r") as f:
        frame_metadata_list = json.load(f)
        print("Read sample_data.json finish.")
    frame_metadata_list = [x for x in frame_metadata_list if x["fileformat"] == "pcd" and "LIDAR_TOP" in x["filename"]]

    frame_dict = {}
    for frame_metadata in frame_metadata_list:
        if not frame_metadata["is_key_frame"]:
            continue
        keyframe_token = frame_metadata["sample_token"]
        ego_pose_token = frame_metadata["ego_pose_token"]
        sensor_token = frame_metadata["calibrated_sensor_token"]
        pose_metadata = pose_dict[ego_pose_token]
        sensor_metadata = sensor_dict[sensor_token]
        # check timestamp alignment
        assert frame_metadata["timestamp"] == pose_metadata["timestamp"]
        # compute pose
        sensor_rotation = quaternion_to_rotation_matrix(np.asarray(sensor_metadata["rotation"]))
        sensor_translation = np.asarray(sensor_metadata["translation"])
        sensor_transform = get_transform_from_rotation_translation(sensor_rotation, sensor_translation)
        ego_rotation = quaternion_to_rotation_matrix(np.asarray(pose_metadata["rotation"]))
        ego_translation = np.asarray(pose_metadata["translation"])
        ego_transform = get_transform_from_rotation_translation(ego_rotation, ego_translation)
        transform = compose_transforms(sensor_transform, ego_transform)
        rotation, translation = get_rotation_translation_from_transform(transform)
        frame_metadata["rotation"] = rotation
        frame_metadata["translation"] = translation
        # add frame metadata
        frame_dict[keyframe_token] = frame_metadata

    return frame_dict


def read_keyframe():
    # load keyframe data
    keyframe_metadata_file = osp.join(data_root, "sample.json")
    with open(keyframe_metadata_file, "r") as f:
        keyframe_metadata_list = json.load(f)
        print("Read sample.json finish.")

    # merge keyframe, frame, pose metadata
    keyframe_dict = {}
    for keyframe_metadata in keyframe_metadata_list:
        token = keyframe_metadata["token"]
        keyframe_dict[token] = keyframe_metadata

    return keyframe_dict


def convert_keyframe_data():
    metadata_dict = read_metadata()

    keyframe_dict = read_keyframe()

    output_dir = osp.join(output_root, "all_data", phase)
    ensure_dir(output_dir)

    for keyframe_token in keyframe_dict.keys():
        filename = metadata_dict[keyframe_token]["filename"]
        filename = osp.join(dataset_root, phase, filename)
        new_basename = osp.basename(filename).replace(".pcd.bin", ".npy")
        new_filename = osp.join(output_dir, new_basename)
        # if osp.exists(new_filename):
        #     print(f"{new_filename} exists")
        #     continue
        # read old data
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:, :3]
        # downsample
        # points = voxel_down_sample(points, voxel_size=voxel_size)
        # save new data
        np.save(new_filename, points)
        print(f"save {filename} to {new_filename}")


def downsample_keyframe(voxel_size: float = 0.3):
    old_dir = osp.join(output_root, "data", phase)
    new_dir = osp.join(output_root, "downsampled", phase)
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


def copy_selected_data():
    def copy_file(filename):
        dirname = osp.dirname(filename)
        ensure_dir(osp.join(output_root, "data", dirname))
        new_path = osp.join(output_root, "data", filename)
        if osp.exists(osp.join(output_root, "data", filename)):
            return
        old_path = osp.join(output_root, "all_data", filename)
        shutil.copy2(old_path, new_path)

    filenames = os.listdir(osp.join(output_root, "metadata"))
    for filename in filenames:
        metadata_list = load_pickle(osp.join(output_root, "metadata", filename))
        for metadata in metadata_list:
            src_file = metadata["src_file"]
            tgt_file = metadata["tgt_file"]
            copy_file(src_file)
            copy_file(tgt_file)


if __name__ == "__main__":
    convert_keyframe_data()
    copy_selected_data()
    downsample_keyframe()
