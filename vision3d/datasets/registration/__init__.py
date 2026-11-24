from .cape import CapePairDataset
from .collected_dataset import CollectedPairDataset
from .deepdeform import DeepDeformPairDataset
from .flyingthings import FlyingThingsPairDataset
from .fourdmatch import FourDMatchPairDataset
from .kitti import OdometryKittiPairDataset
from .modelnet import ModelNetPairDataset
from .mvp import MVPRegistrationDataset
from .nuscenes import NuScenesPairDataset
from .rgbdscenes import RGBDScenes2D3DHardPairDataset
from .scannet import ScanNet2D3DPairDataset, ScanNetPairURRDataset
from .scene import (
    CSRegLidarPairDataset,
    CSRegSfmPairDataset,
    EthPairDataset,
    MitPairDataset,
    OxfordPairDataset,
    ScanNetPPiPhonePairDataset,
    TiersPairDataset,
    WodPairDataset,
)
from .sevenscenes import SevenScenes2D3DEasyPairDataset, SevenScenes2D3DHardPairDataset
from .shapenet import ShapeNetPairDataset
from .threedmatch import ThreeDMatchPairDataset, ThreeDMatchPairURRDataset, ThreeDMatchRgbPairDataset
from .twodthreedmatch import TwoDThreeDMatchPairDataset
