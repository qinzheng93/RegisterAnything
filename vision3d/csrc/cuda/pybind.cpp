#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "ops/ball_query/ball_query.h"
#include "ops/furthest_point_sample/furthest_point_sample.h"
#include "ops/gather/gather.h"
#include "ops/group_gather/group_gather.h"
#include "ops/knn_points/knn_points.h"
#include "ops/three_interpolate/three_interpolate.h"
#include "ops/three_nn/three_nn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUDA extensions
  m.def(
      "ball_query",
      &vision3d::ball_query,
      "Ball Query",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("indices"),
      py::arg("max_radius"),
      py::arg("num_samples"));

  m.def(
      "group_gather_forward",
      &vision3d::group_gather_forward,
      "Group Gather Points By Index (Forward)",
      py::arg("sources"),
      py::arg("indices"),
      py::arg("targets"));
  m.def(
      "group_gather_backward",
      &vision3d::group_gather_backward,
      "Group Gather Points By Index (Backward)",
      py::arg("target_grads"),
      py::arg("indices"),
      py::arg("source_grads"),
      py::arg("num_sources"));

  m.def(
      "furthest_point_sample",
      &vision3d::furthest_point_sample,
      "Furthest Point Sampling",
      py::arg("points"),
      py::arg("distances"),
      py::arg("indices"),
      py::arg("num_samples"));

  m.def(
      "gather_forward",
      &vision3d::gather_forward,
      "Gather Points By Index (Forward)",
      py::arg("sources"),
      py::arg("indices"),
      py::arg("targets"));
  m.def(
      "gather_backward",
      &vision3d::gather_backward,
      "Gather Points By Index (Backward)",
      py::arg("target_grads"),
      py::arg("indices"),
      py::arg("source_grads"),
      py::arg("num_sources"));

  m.def(
      "three_nn",
      &vision3d::three_nn,
      "Three Nearest Neighbors",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("tnn_distances"),
      py::arg("tnn_indices"));

  m.def(
      "three_interpolate_forward",
      &vision3d::three_interpolate_forward,
      "Three Nearest Neighbors Interpolate (Forward)",
      py::arg("sources"),
      py::arg("indices"),
      py::arg("weights"),
      py::arg("targets"));
  m.def(
      "three_interpolate_backward",
      &vision3d::three_interpolate_backward,
      "Three Nearest Neighbors Interpolate (Backward)",
      py::arg("target_grads"),
      py::arg("indices"),
      py::arg("weights"),
      py::arg("source_grads"),
      py::arg("num_sources"));

  m.def(
      "knn_points",
      &vision3d::knn_points,
      "k Nearest Neighbors",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("knn_distances"),
      py::arg("knn_indices"),
      py::arg("num_neighbors"));
}
