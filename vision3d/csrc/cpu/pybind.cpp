#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "ops/deformation_graph/deformation_graph.h"
#include "ops/graph_proc/graph_proc.h"
#include "ops/grid_subsampling/grid_subsampling.h"
#include "ops/knn_neighbors/knn_neighbors.h"
#include "ops/node_sampling/node_sampling.h"
#include "ops/radius_neighbors/radius_neighbors.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // NumPy Extensions
  m.def("depth_to_mesh", &graph_proc::depthToMesh, "Convert Depth Image to Mesh");
  m.def("erode_mesh", &graph_proc::erode_mesh, "Erode Mesh");
  m.def("sample_nodes", &graph_proc::sample_nodes, "Sample Graph Nodes");
  m.def("compute_edges_geodesic", &graph_proc::compute_edges_geodesic, "Compute Graph Edges via Geodesic Distance");
  m.def("compute_edges_euclidean", &graph_proc::compute_edges_euclidean, "Compute Graph Edges via Euclidean Distance");
  m.def(
      "compute_pixel_anchors_geodesic",
      &graph_proc::compute_pixel_anchors_geodesic,
      "Compute Pixel Anchors via Geodesic Distance");
  m.def("node_and_edge_clean_up", &graph_proc::node_and_edge_clean_up, "Remove Nodes and Edges with Too Few Neighbors");
  m.def("update_pixel_anchors", &graph_proc::update_pixel_anchors, "Update Pixel Anchor IDs");
  m.def("compute_clusters", &graph_proc::compute_clusters, "Compute Graph Clusters");

  m.def(
      "radius_neighbors",
      &vision3d::radius_neighbors,
      "Radius Neighbors",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("q_lengths"),
      py::arg("s_lenghts"),
      py::arg("radius"));

  m.def(
      "knn_neighbors",
      &vision3d::knn_neighbors,
      "K-Nearest Neighbors",
      py::arg("q_points"),
      py::arg("s_points"),
      py::arg("q_lengths"),
      py::arg("s_lenghts"),
      py::arg("num_neighbors"));

  m.def(
      "grid_subsampling",
      &vision3d::grid_subsampling,
      "Grid Subsampling",
      py::arg("points"),
      py::arg("lengths"),
      py::arg("voxel_size"));

  m.def(
      "build_deformation_graph_from_point_cloud",
      &vision3d::build_deformation_graph_from_point_cloud,
      "Build Deformation Graph from Point Cloud",
      py::arg("vertices"),
      py::arg("node_indices"),
      py::arg("num_neighbors"),
      py::arg("num_anchors"),
      py::arg("max_distance"),
      py::arg("node_coverage"));

  m.def(
      "sample_nodes_with_fps",
      &vision3d::sample_nodes_with_fps,
      "Sample Nodes with Furthest Point Sampling.",
      py::arg("points"),
      py::arg("min_distance"),
      py::arg("num_samples"));
}
