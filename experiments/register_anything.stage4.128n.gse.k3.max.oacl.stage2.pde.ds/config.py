import argparse
import os
import os.path as osp

from omegaconf import DictConfig, OmegaConf

from vision3d.utils.io import ensure_dir

_C = OmegaConf.create()

# system
_C.sys = OmegaConf.create()
_C.sys.workspace = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))

# exp
_C.exp = OmegaConf.create()
_C.exp.name = osp.basename(osp.dirname(osp.realpath(__file__)))
_C.exp.project = _C.exp.name.split(".")[0]
_C.exp.working_dir = osp.dirname(osp.realpath(__file__))
_C.exp.output_dir = osp.join(_C.sys.workspace, "output", _C.exp.name)
_C.exp.checkpoint_dir = osp.join(_C.exp.output_dir, "checkpoints")
_C.exp.log_dir = osp.join(_C.exp.output_dir, "logs")
_C.exp.event_dir = osp.join(_C.exp.output_dir, "events")
_C.exp.cache_dir = osp.join(_C.exp.output_dir, "cache")
_C.exp.result_dir = osp.join(_C.exp.output_dir, "results")
_C.exp.seed = 7351

ensure_dir(_C.exp.output_dir)
ensure_dir(_C.exp.checkpoint_dir)
ensure_dir(_C.exp.log_dir)
ensure_dir(_C.exp.event_dir)
ensure_dir(_C.exp.cache_dir)
ensure_dir(_C.exp.result_dir)

# data
_C.data = OmegaConf.create()
_C.data.dataset_root = osp.join(_C.sys.workspace, "data")
_C.data.dataset_names = ["3DMatch", "odometryKitti", "nuScenes", "ModelNet"]

# data - 3DMatch
_C.data.threedmatch = OmegaConf.create()
_C.data.threedmatch.dataset_dir = osp.join(_C.data.dataset_root, "3DMatch")

_C.data.threedmatch.train = OmegaConf.create()
_C.data.threedmatch.train.max_points = 30000
_C.data.threedmatch.train.use_augmentation = True
_C.data.threedmatch.train.augmentation_noise = 0.005
_C.data.threedmatch.train.augmentation_rotation = 1.0
_C.data.threedmatch.train.use_weak_augmentation = False
_C.data.threedmatch.train.use_random_crop = True
_C.data.threedmatch.train.keep_ratio = 0.7
_C.data.threedmatch.train.use_random_scaling = True
_C.data.threedmatch.train.augmentation_min_scale = 0.8
_C.data.threedmatch.train.augmentation_max_scale = 1.2
_C.data.threedmatch.train.use_raw_data = False
_C.data.threedmatch.train.voxel_size = 0.025

_C.data.threedmatch.test = OmegaConf.create()
_C.data.threedmatch.test.max_points = None
_C.data.threedmatch.test.use_raw_data = True
_C.data.threedmatch.test.voxel_size = 0.025
_C.data.threedmatch.test.geometry_norm = True
_C.data.threedmatch.test.rotated = False

# data - odometryKitti
_C.data.kitti = OmegaConf.create()
_C.data.kitti.dataset_dir = osp.join(_C.data.dataset_root, "odometryKitti")

_C.data.kitti.train = OmegaConf.create()
_C.data.kitti.train.max_points = 30000
_C.data.kitti.train.use_augmentation = True
_C.data.kitti.train.augmentation_noise = 0.01
_C.data.kitti.train.augmentation_min_scale = 0.8
_C.data.kitti.train.augmentation_max_scale = 1.2
_C.data.kitti.train.augmentation_shift = 2.0
_C.data.kitti.train.augmentation_rotation = 1.0
_C.data.kitti.train.use_raw_data = False
_C.data.kitti.train.voxel_size = 0.3

_C.data.kitti.test = OmegaConf.create()
_C.data.kitti.test.max_points = None
_C.data.kitti.test.use_raw_data = True
_C.data.kitti.test.voxel_size = 0.3
_C.data.kitti.test.geometry_norm = True

# data - nuScenes
_C.data.nuscenes = OmegaConf.create()
_C.data.nuscenes.dataset_dir = osp.join(_C.data.dataset_root, "nuScenes")

_C.data.nuscenes.train = OmegaConf.create()
_C.data.nuscenes.train.max_points = 30000
_C.data.nuscenes.train.use_augmentation = True
_C.data.nuscenes.train.augmentation_noise = 0.01
_C.data.nuscenes.train.augmentation_min_scale = 0.8
_C.data.nuscenes.train.augmentation_max_scale = 1.2
_C.data.nuscenes.train.augmentation_shift = 2.0
_C.data.nuscenes.train.augmentation_rotation = 1.0
_C.data.nuscenes.train.use_raw_data = False
_C.data.nuscenes.train.voxel_size = 0.3

_C.data.nuscenes.test = OmegaConf.create()
_C.data.nuscenes.test.max_points = None
_C.data.nuscenes.test.use_raw_data = True
_C.data.nuscenes.test.voxel_size = 0.3
_C.data.nuscenes.test.geometry_norm = True
_C.data.nuscenes.test.max_distance = 200.0

# data - ETH
_C.data.eth = OmegaConf.create()
_C.data.eth.dataset_dir = osp.join(_C.data.dataset_root, "ETH")
_C.data.eth.subset = "test"
_C.data.eth.max_points = None
_C.data.eth.voxel_size = 0.2
_C.data.eth.geometry_norm = True

# data - TIERS
_C.data.tiers = OmegaConf.create()
_C.data.tiers.dataset_dir = osp.join(_C.data.dataset_root, "TIERS")
_C.data.tiers.subset = "test"
_C.data.tiers.max_points = None
_C.data.tiers.voxel_size = 0.2
_C.data.tiers.geometry_norm = True
_C.data.tiers.max_range = 250.0

# data - MIT
_C.data.mit = OmegaConf.create()
_C.data.mit.dataset_dir = osp.join(_C.data.dataset_root, "MIT")
_C.data.mit.subset = "test"
_C.data.mit.max_points = None
_C.data.mit.voxel_size = 0.3
_C.data.mit.geometry_norm = True
_C.data.mit.max_range = 250.0

# data - WOD
_C.data.wod = OmegaConf.create()
_C.data.wod.dataset_dir = osp.join(_C.data.dataset_root, "WOD")
_C.data.wod.subset = "test"
_C.data.wod.max_points = None
_C.data.wod.voxel_size = 0.3
_C.data.wod.geometry_norm = True
_C.data.wod.max_range = 250.0

# data - ScanNet++i
_C.data.scannetppi = OmegaConf.create()
_C.data.scannetppi.dataset_dir = osp.join(_C.data.dataset_root, "ScanNet++i")
_C.data.scannetppi.subset = "test"
_C.data.scannetppi.max_points = None
_C.data.scannetppi.voxel_size = 0.025
_C.data.scannetppi.geometry_norm = True
_C.data.scannetppi.max_range = 50.0

# data - 3DCSR-L
_C.data.csreg_l = OmegaConf.create()
_C.data.csreg_l.dataset_dir = osp.join(_C.data.dataset_root, "3DCSR")
_C.data.csreg_l.subset = "kinect_lidar_inv"
_C.data.csreg_l.max_points = 30000
_C.data.csreg_l.voxel_size = 0.025
_C.data.csreg_l.geometry_norm = True
_C.data.csreg_l.max_range = None

# data - 3DCSR-S
_C.data.csreg_s = OmegaConf.create()
_C.data.csreg_s.dataset_dir = osp.join(_C.data.dataset_root, "3DCSR")
_C.data.csreg_s.subset = "kinect_sfm_inv"
_C.data.csreg_s.max_points = 30000
_C.data.csreg_s.voxel_size = 0.025
_C.data.csreg_s.geometry_norm = True
_C.data.csreg_s.max_range = None

# data - Oxford
_C.data.oxford = OmegaConf.create()
_C.data.oxford.dataset_dir = osp.join(_C.data.dataset_root, "Oxford")
_C.data.oxford.subset = "test"
_C.data.oxford.max_points = None
_C.data.oxford.voxel_size = 0.25
_C.data.oxford.geometry_norm = True
_C.data.oxford.max_range = 250.0

# data - ModelNet
_C.data.modelnet = OmegaConf.create()
_C.data.modelnet.dataset_dir = osp.join(_C.data.dataset_root, "ModelNet")

_C.data.modelnet.base = OmegaConf.create()
_C.data.modelnet.base.num_points = 717
_C.data.modelnet.base.voxel_size = None
_C.data.modelnet.base.rotation_magnitude = 45.0
_C.data.modelnet.base.translation_magnitude = 0.5
_C.data.modelnet.base.keep_ratio = 0.7
_C.data.modelnet.base.crop_method = "plane"
_C.data.modelnet.base.asymmetric = True
_C.data.modelnet.base.twice_sample = True
_C.data.modelnet.base.twice_transform = False
_C.data.modelnet.base.return_normals = False
_C.data.modelnet.base.return_occupancy = True

_C.data.modelnet.train = OmegaConf.create()
_C.data.modelnet.train.noise_magnitude = 0.05
_C.data.modelnet.train.class_indices = "seen"
_C.data.modelnet.train.use_random_scaling = True
_C.data.modelnet.train.augmentation_min_scale = 0.8
_C.data.modelnet.train.augmentation_max_scale = 1.2
_C.data.modelnet.train.deterministic = False

_C.data.modelnet.test = OmegaConf.create()
_C.data.modelnet.test.noise_magnitude = 0.05
_C.data.modelnet.test.class_indices = "unseen"
_C.data.modelnet.test.deterministic = True

# data - ShapeNet
_C.data.shapenet = OmegaConf.create()
_C.data.shapenet.dataset_dir = osp.join(_C.data.dataset_root, "ShapeNet")

_C.data.shapenet.base = OmegaConf.create()
_C.data.shapenet.base.num_points = 1434
_C.data.shapenet.base.voxel_size = None
_C.data.shapenet.base.rotation_magnitude = 45.0
_C.data.shapenet.base.translation_magnitude = 0.5
_C.data.shapenet.base.keep_ratio = 0.7
_C.data.shapenet.base.crop_method = "plane"
_C.data.shapenet.base.asymmetric = True
_C.data.shapenet.base.twice_sample = True
_C.data.shapenet.base.twice_transform = False
_C.data.shapenet.base.return_normals = False
_C.data.shapenet.base.return_occupancy = True

_C.data.shapenet.train = OmegaConf.create()
_C.data.shapenet.train.noise_magnitude = 0.05
_C.data.shapenet.train.deterministic = False

_C.data.shapenet.test = OmegaConf.create()
_C.data.shapenet.test.noise_magnitude = 0.05
_C.data.shapenet.test.subset = "test"
_C.data.shapenet.test.deterministic = True
_C.data.shapenet.test.voxel_size = 0.025
_C.data.shapenet.test.geometry_norm = True

# train data
_C.train = OmegaConf.create()
_C.train.batch_size = 1
_C.train.num_workers = 8

# test data
_C.test = OmegaConf.create()
_C.test.batch_size = 1
_C.test.num_workers = 8

# evaluation
_C.eval = OmegaConf.create()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = {
    "3DMatch": 0.1,
    "ScanNet++i": 0.1,
    "3DCSR-L": 0.1,
    "3DCSR-S": 0.1,
    "odometryKitti": 1.0,
    "nuScenes": 1.0,
    "ETH": 0.8,
    "TIERS": 0.8,
    "MIT": 1.0,
    "WOD": 1.0,
    "Oxford": 1.0,
    "ModelNet": 0.1,
    "ShapeNet": 0.1,
}
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rre_threshold = {
    "ScanNet++i": 5.0,
    "3DCSR-L": 15.0,
    "3DCSR-S": 15.0,
    "odometryKitti": 5.0,
    "nuScenes": 5.0,
    "ETH": 2.0,
    "TIERS": 5.0,
    "MIT": 5.0,
    "WOD": 5.0,
    "Oxford": 5.0,
    "ModelNet": 5.0,
    "ShapeNet": 5.0,
}
_C.eval.rte_threshold = {
    "ScanNet++i": 0.3,
    "3DCSR-L": 0.3,
    "3DCSR-S": 0.3,
    "odometryKitti": 0.6,  # 2.0
    "nuScenes": 0.6,  # 2.0
    "ETH": 0.3,
    "TIERS": 0.6,  # 2.0
    "MIT": 0.6,  # 2.0
    "WOD": 0.6,  # 2.0
    "Oxford": 0.6,  # 2.0
    "ModelNet": 0.1,
    "ShapeNet": 0.1,
}
_C.eval.rmse_threshold = {"3DMatch": 0.2}

# ransac
_C.ransac = OmegaConf.create()
_C.ransac.distance_threshold = {"3DMatch": 0.05, "odometryKitti": 0.3, "nuScenes": 0.3, "ETH": 0.4}
_C.ransac.num_points = {"3DMatch": 3, "odometryKitti": 4, "nuScenes": 4, "ETH": 4}
_C.ransac.num_iterations = 50000

# trainer
_C.trainer = OmegaConf.create()
_C.trainer.max_epoch = 60
_C.trainer.num_iters_per_epoch = 50000
_C.trainer.grad_acc_steps = 1
_C.trainer.use_amp = False

# EMA
_C.ema = OmegaConf.create()
_C.ema.use_ema = True
_C.ema.decay = 0.99
_C.ema.start_epoch = -1
_C.ema.use_buffers = False
_C.ema.update_bn = False

# optim
_C.optimizer = OmegaConf.create()
_C.optimizer.type = "AdamW"
_C.optimizer.lr = 1e-4
_C.optimizer.weight_decay = 1e-4

# scheduler
_C.scheduler = OmegaConf.create()
_C.scheduler.type = "Cosine"
_C.scheduler.total_steps = _C.trainer.max_epoch * _C.trainer.num_iters_per_epoch
_C.scheduler.warmup_steps = 25000
_C.scheduler.eta_init = 0.1
_C.scheduler.eta_min = 0.1

# model - Global
_C.model = OmegaConf.create()
_C.model.matching_radius = {
    "3DMatch": 0.05,
    "ScanNet++i": 0.05,
    "3DCSR-L": 0.05,
    "3DCSR-S": 0.05,
    "odometryKitti": 0.6,
    "nuScenes": 0.6,
    "ETH": 0.4,
    "TIERS": 0.4,
    "MIT": 0.6,
    "WOD": 0.6,
    "Oxford": 0.6,
    "ModelNet": 0.05,
    "ShapeNet": 0.05,
}
_C.model.num_points_in_patch = 128

# model - backbone
_C.model.backbone = OmegaConf.create()
_C.model.backbone.num_stages = 4
_C.model.backbone.kernel_size = 15
_C.model.backbone.base_voxel_size = 0.025
_C.model.backbone.base_radius = 2.5 * _C.model.backbone.base_voxel_size
_C.model.backbone.base_sigma = 2.0 * _C.model.backbone.base_voxel_size
_C.model.backbone.input_dim = 1
_C.model.backbone.init_dim = 64
_C.model.backbone.output_dim = 256

# model - Coarse Matching
_C.model.coarse_matching = OmegaConf.create()
_C.model.coarse_matching.num_targets = 128
_C.model.coarse_matching.overlap_threshold = 0.1
_C.model.coarse_matching.num_correspondences = 256
_C.model.coarse_matching.dual_normalization = True

# model - Coarse GeoTransformer
_C.model.c_geotransformer = OmegaConf.create()
_C.model.c_geotransformer.input_dim = 1024
_C.model.c_geotransformer.hidden_dim = 256
_C.model.c_geotransformer.output_dim = 256
_C.model.c_geotransformer.num_heads = 4
_C.model.c_geotransformer.num_neighbors = 128
_C.model.c_geotransformer.blocks = [
    "geometric_self",
    "ordered_cross",
    "geometric_self",
    "ordered_cross",
    "geometric_self",
    "ordered_cross",
]
_C.model.c_geotransformer.sigma_d = 0.2
_C.model.c_geotransformer.sigma_a = 15
_C.model.c_geotransformer.angle_k = 3
_C.model.c_geotransformer.angle_reduction = "max"
_C.model.c_geotransformer.embedding = "GSEv1"
_C.model.c_geotransformer.norm_cfg = "RMSNorm"
_C.model.c_geotransformer.act_cfg = "SiLU"

# model - Fine GeoTransformer
_C.model.f_geotransformer = OmegaConf.create()
_C.model.f_geotransformer.input_dim = 256
_C.model.f_geotransformer.hidden_dim = 256
_C.model.f_geotransformer.output_dim = 256
_C.model.f_geotransformer.num_heads = 4
_C.model.f_geotransformer.blocks = ["geometric_self", "ordered_cross"]
_C.model.f_geotransformer.sigma_d = 0.05
_C.model.f_geotransformer.embedding = "PDE"
_C.model.f_geotransformer.norm_cfg = "RMSNorm"
_C.model.f_geotransformer.act_cfg = "SiLU"

# model - Dual Softmax
_C.model.dual_softmax = OmegaConf.create()
_C.model.dual_softmax.input_dim = 256
_C.model.dual_softmax.output_dim = 256

# model - Fine Matching
_C.model.fine_matching = OmegaConf.create()
_C.model.fine_matching.topk = 3
_C.model.fine_matching.acceptance_radius = 0.1
_C.model.fine_matching.mutual = True
_C.model.fine_matching.confidence_threshold = 0.0
_C.model.fine_matching.use_dustbin = False
_C.model.fine_matching.use_global_score = False
_C.model.fine_matching.use_logits = False
_C.model.fine_matching.use_nms = False
_C.model.fine_matching.nms_radius = 0.01
_C.model.fine_matching.min_local_correspondences = 3
_C.model.fine_matching.max_global_correspondences = None
_C.model.fine_matching.filter_correspondences = False
_C.model.fine_matching.num_refinement_steps = 5

_C.model.sinkhorn = OmegaConf.create()
_C.model.sinkhorn.num_iterations = 100

# loss
_C.loss = OmegaConf.create()

# loss - Coarse level
_C.loss.coarse_loss = OmegaConf.create()
_C.loss.coarse_loss.positive_margin = 0.1
_C.loss.coarse_loss.negative_margin = 1.4
_C.loss.coarse_loss.positive_optimal = 0.1
_C.loss.coarse_loss.negative_optimal = 1.4
_C.loss.coarse_loss.log_scale = 24
_C.loss.coarse_loss.positive_overlap = 0.1
_C.loss.coarse_loss.weight = 1.0

# loss - Fine level
_C.loss.fine_loss = OmegaConf.create()
_C.loss.fine_loss.positive_radius = {"3DMatch": 0.05, "odometryKitti": 0.6, "nuScenes": 0.6, "ModelNet": 0.05}
_C.loss.fine_loss.weight = 1.0


def make_cfg() -> DictConfig:
    return _C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()

    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.exp.output_dir, "output")


if __name__ == "__main__":
    main()
