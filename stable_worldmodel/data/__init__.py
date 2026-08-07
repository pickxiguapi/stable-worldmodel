# Importing the formats subpackage registers all built-in formats whose
# optional deps are installed.
from . import formats as _formats  # noqa: F401
from .buffer import ReplayBuffer, classic_filter
from .dataset import *
from .format import (
    EPISODE_DATA_KEY,
    FORMATS,
    WRITE_MODES,
    Format,
    Writer,
    detect_format,
    get_format,
    list_formats,
    register_format,
    split_episode_data,
    validate_write_mode,
)
from .formats.folder import FolderDataset, FolderWriter, ImageDataset

# Re-export concrete readers/writers from their format modules so existing
# imports like `from stable_worldmodel.data import LanceDataset` keep working.
# Optional formats (hdf5, video) are re-exported only when their extras are
# installed; absent ones are simply not bound at module level.
from .formats.lance import LanceDataset, LanceWriter
from .formats.lerobot import LeRobotAdapter
from .normalization import (
    IdentityScaler,
    PercentileScaler,
    ZScoreScaler,
    get_scaler,
)
from .utils import *
from .utils import column_normalizer

try:
    from .formats.hdf5 import HDF5Dataset, HDF5Writer  # noqa: F401
except ImportError:
    pass

try:
    from .formats.video import VideoDataset, VideoWriter  # noqa: F401
except ImportError:
    pass

try:
    from .formats.lance_video import (  # noqa: F401
        LanceVideoDataset,
        LanceVideoWriter,
    )
except ImportError:
    pass


__all__ = [
    'EPISODE_DATA_KEY',
    'FORMATS',
    'WRITE_MODES',
    'FolderDataset',
    'FolderWriter',
    'Format',
    'GoalDataset',
    'HierarchicalGoalDataset',
    'IdentityScaler',
    'ImageDataset',
    'LanceDataset',
    'LanceWriter',
    'LeRobotAdapter',
    'PercentileScaler',
    'ReplayBuffer',
    'Writer',
    'ZScoreScaler',
    'classic_filter',
    'column_normalizer',
    'detect_format',
    'get_format',
    'get_scaler',
    'list_formats',
    'register_format',
    'split_episode_data',
    'validate_write_mode',
]
