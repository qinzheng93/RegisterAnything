import os
import os.path as osp

import numpy as np
import open3d as o3d
from tqdm import tqdm

from vision3d.utils.io import ensure_dir


def compute_triangle_areas(vertices, triangles):
    tri_vert_a = vertices[triangles[:, 0]]
    tri_vert_b = vertices[triangles[:, 1]]
    tri_vert_c = vertices[triangles[:, 2]]
    vec_ab = tri_vert_b - tri_vert_a
    vec_ac = tri_vert_c - tri_vert_a
    cross_products = np.cross(vec_ab, vec_ac)
    triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    return triangle_areas


def sample_points_with_normals_from_obj(filepath, num_points=2048):
    mesh = o3d.io.read_triangle_mesh(filepath)

    if not mesh.has_triangle_normals():
        print("No normals found in the mesh, computing triangle normals...")
        mesh.compute_triangle_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0 or len(vertices) == 0:
        print("The mesh does not have valid vertices or triangles.")
        return None, None

    triangle_normals = np.asarray(mesh.triangle_normals)  # 形状 (F, 3)

    triangle_areas = compute_triangle_areas(vertices, triangles)

    if np.any(triangle_areas == 0):
        non_zero_area_indices = np.where(triangle_areas > 0)[0]
        if len(non_zero_area_indices) == 0:
            print("All triangles have zero area, cannot sample.")
            return None, None
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[non_zero_area_indices])
        mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(mesh.triangle_normals)[non_zero_area_indices])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    triangle_areas = compute_triangle_areas(vertices, triangles)

    total_area = np.sum(triangle_areas)
    probs = triangle_areas / total_area
    sampled_triangle_indices = np.random.choice(np.arange(len(triangles)), size=num_points, p=probs)

    chosen_triangles = triangles[sampled_triangle_indices]
    chosen_normals = triangle_normals[sampled_triangle_indices]

    chosen_vert_a = vertices[chosen_triangles[:, 0]]
    chosen_vert_b = vertices[chosen_triangles[:, 1]]
    chosen_vert_c = vertices[chosen_triangles[:, 2]]

    r1 = np.random.rand(num_points)
    r2 = np.random.rand(num_points)

    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    points = u[:, np.newaxis] * chosen_vert_a + v[:, np.newaxis] * chosen_vert_b + w[:, np.newaxis] * chosen_vert_c

    return points, chosen_normals


def sample_points():
    num_points = 8192

    model_dir = "ShapeNetCore.v2"
    output_dir = "data"
    ensure_dir(output_dir)

    filenames = sorted(os.listdir(model_dir))
    category_ids = [x for x in filenames if osp.isdir(osp.join(model_dir, x))]
    for category_id in category_ids:
        print(f"Processing {category_id}...")
        old_category_dir = osp.join(model_dir, category_id)
        new_category_dir = osp.join(output_dir, category_id)
        ensure_dir(new_category_dir)
        model_ids = sorted(os.listdir(old_category_dir))
        pbar = tqdm(model_ids)
        for model_id in pbar:
            model_path = osp.join(old_category_dir, model_id, "models", "model_normalized.obj")
            if not osp.exists(model_path):
                print(f"{model_path} does not exist.")
                continue
            npz_path = osp.join(new_category_dir, model_id + ".npz")
            if osp.exists(npz_path):
                # print(f"{npz_path} already exists.")
                continue
            points, normals = sample_points_with_normals_from_obj(model_path, num_points)
            if points is None or normals is None:
                print(f"Skip model: {model_id}.")
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            # o3d.io.write_point_cloud(osp.join(new_category_dir, model_id + ".ply"), pcd)
            pbar.set_description(f"{model_id}, #Points: {points.shape[0]}")
            np.savez_compressed(npz_path, points=points, normals=normals)


def count_models():
    workspace = osp.join(os.environ["HOME"], "workspace")
    all_data_dir = osp.join(workspace, "data", "registration", "ShapeNet", "all_data")
    category_ids = sorted(os.listdir(all_data_dir))
    for category_id in category_ids:
        category_dir = osp.join(all_data_dir, category_id)
        model_files = os.listdir(category_dir)
        model_files = sorted([x for x in model_files if x.endswith(".npz")])
        num_models = len(model_files)
        print(f"{category_id}: {num_models} models.")


if __name__ == "__main__":
    sample_points()
    count_models()
