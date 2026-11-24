from numpy import ndarray
from scipy.spatial import KDTree


def ball_query(q_points: ndarray, s_points: ndarray, radius: float) -> list[ndarray]:
    s_tree = KDTree(s_points)
    indices_list = s_tree.query_ball_point(q_points, radius)
    return indices_list
