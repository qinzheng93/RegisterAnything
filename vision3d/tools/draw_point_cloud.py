import sys

import numpy as np

from vision3d.utils.open3d import draw_geometries, get_color, make_open3d_point_cloud

if __name__ == "__main__":
    filename = sys.argv[1]
    points = np.load(filename)
    pcd = make_open3d_point_cloud(points)
    pcd.paint_uniform_color(get_color("src_color"))
    pcd.estimate_normals()
    draw_geometries(pcd)
