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
        self.n_patients = len(self.indexes_path)

    def get_connectivity_matrix(self, index: int) -> np.ndarray:
        subject_path = self.indexes_path[index]
        subject_index = subject_path.name[:5]
        con = loadmat(str(subject_path/f"{subject_index}_ho_correlation.mat"))["connectivity"]
        return con

    def get_connectivity_features(self, index: int) -> np.ndarray:
        """Retrieve upper triangular coefficients (~ 6000 components)
        without any fancy pre-processing

        Args:
            index (int): patient index

        Returns:
            np.ndarray: connectivity vector
        """
        mat = self.get_connectivity_matrix(index)
        return mat[np.triu_indices_from(mat)]
    
    def get_input_feature_map(self) -> np.ndarray:
        """Retrieve feature maps

        Returns:
            np.ndarray: N inviduals x C features
        """
        mat_feat = []
        for patient_index in range(self.n_patients):
            mat_feat.append(self.get_connectivity_features(patient_index))
        return np.array(mat_feat)
