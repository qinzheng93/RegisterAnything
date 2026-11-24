#include "knn_neighbors.h"

namespace vision3d {

py::array_t<long> knn_neighbors(
    const py::array_t<float>& q_points,
    const py::array_t<float>& s_points,
    const py::array_t<long>& q_lengths,
    const py::array_t<long>& s_lengths,
    std::size_t num_neighbors) {
  std::size_t total_q_points = q_points.shape(0);
  std::size_t total_s_points = s_points.shape(0);
  std::size_t batch_size = q_lengths.shape(0);

  // read array
  std::vector<std::size_t> q_lengths_vec(batch_size);
  std::vector<std::size_t> s_lengths_vec(batch_size);
  for (std::size_t i = 0; i < batch_size; ++i) {
    q_lengths_vec[i] = *q_lengths.data(i);
    s_lengths_vec[i] = *s_lengths.data(i);
  }

  nanoflann::SearchParams search_params;
  search_params.sorted = true;
  nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10);

  std::vector<std::vector<std::size_t>> neighbor_indices_vec(total_q_points);
  std::vector<std::vector<float>> neighbor_distances_vec(total_q_points);

  std::size_t q_start_index = 0;
  std::size_t s_start_index = 0;
  for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    // read array
    auto cur_q_length = q_lengths_vec[batch_index];
    auto cur_s_length = s_lengths_vec[batch_index];

    PointCloud cur_s_point_cloud;
    auto& cur_s_points_vec = cur_s_point_cloud.pts;
    cur_s_points_vec.resize(cur_s_length);
    for (std::size_t i = 0; i < cur_s_length; ++i) {
      auto s_index = s_start_index + i;
      cur_s_points_vec[i].x = *s_points.data(s_index, 0);
      cur_s_points_vec[i].y = *s_points.data(s_index, 1);
      cur_s_points_vec[i].z = *s_points.data(s_index, 2);
    }

    // build kdtree
    kd_tree_t* index = new kd_tree_t(3, cur_s_point_cloud, tree_params);
    index->buildIndex();

    // kdtree query
    for (std::size_t i = 0; i < cur_q_length; ++i) {
      std::size_t q_index = q_start_index + i;
      float cur_q_point[3] = {*q_points.data(q_index, 0), *q_points.data(q_index, 1), *q_points.data(q_index, 2)};
      neighbor_indices_vec[q_index].reserve(num_neighbors);
      neighbor_distances_vec[q_index].reserve(num_neighbors);
      std::size_t cur_num_neighbors = index->knnSearch(
          cur_q_point, num_neighbors, neighbor_indices_vec[q_index].data(), neighbor_distances_vec[q_index].data());
      if (cur_num_neighbors < num_neighbors) {
        for (std::size_t j = cur_num_neighbors; j < num_neighbors; ++j) {
          neighbor_indices_vec[q_index][j] = total_s_points;
        }
      }
    }

    delete index;

    q_start_index += cur_q_length;
    s_start_index += cur_s_length;
  }

  // write array
  py::array_t<long> neighbor_indices = py::array_t<long>({total_q_points, num_neighbors});
  q_start_index = 0;
  s_start_index = 0;
  for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    auto cur_q_length = q_lengths_vec[batch_index];
    auto cur_s_length = s_lengths_vec[batch_index];
    for (std::size_t i = 0; i < cur_q_length; ++i) {
      auto q_index = q_start_index + i;
      const auto& cur_neighbor_indices_vec = neighbor_indices_vec[q_index];
      for (std::size_t j = 0; j < num_neighbors; ++j) {
        if (cur_neighbor_indices_vec[j] < total_s_points) {
          *neighbor_indices.mutable_data(q_index, j) = cur_neighbor_indices_vec[j] + s_start_index;
        } else {
          *neighbor_indices.mutable_data(q_index, j) = cur_neighbor_indices_vec[j];
        }
      }
    }
    q_start_index += cur_q_length;
    s_start_index += cur_s_length;
  }

  return neighbor_indices;
}

}  // namespace vision3d
