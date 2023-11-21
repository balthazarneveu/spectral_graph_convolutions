import logging
from pathlib import Path
from scipy.io import loadmat
from typing import List
import numpy as np
DEFAULT_ABIDE_NAME = "__ABIDE_dataset"
root_folder = Path(__file__).parent.parent.parent.parent.absolute()
DEFAULT_ABIDE_ROOT_LOCATION = root_folder/DEFAULT_ABIDE_NAME
ABIDE_PCP = 'ABIDE_pcp'
PIPELINE = 'cpac'
STRATEGY = 'filt_noglobal'
ATLAS_NAME = 'ho'

DEFAULT_ABIDE_LOCATION = DEFAULT_ABIDE_ROOT_LOCATION/ ABIDE_PCP / PIPELINE/ STRATEGY

if not DEFAULT_ABIDE_LOCATION.exists():
    logging.warning(f"No ABIDE DATA found at {DEFAULT_ABIDE_LOCATION}")


class AbideData():
    def __init__(self, folder_root: Path = DEFAULT_ABIDE_LOCATION) -> None:
        assert folder_root.exists()
        self.folder_root = folder_root
        self.indexes_path = sorted(list(self.folder_root.glob("5*")))

    def get_connectivity_matrix(self, index: int) -> np.ndarray:
        subject_path = self.indexes_path[index]
        subject_index = subject_path.name[:5]
        con = loadmat(str(subject_path/f"{subject_index}_ho_correlation.mat"))["connectivity"]
        return con
