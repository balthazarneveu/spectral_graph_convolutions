import logging
from pathlib import Path
from scipy.io import loadmat
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.spatial import distance

DEFAULT_ABIDE_NAME = "__ABIDE_dataset"
root_folder = Path(__file__).parent.parent.parent.parent.absolute()
DEFAULT_ABIDE_ROOT_LOCATION = root_folder/DEFAULT_ABIDE_NAME
ABIDE_PCP = 'ABIDE_pcp'
PIPELINE = 'cpac'
STRATEGY = 'filt_noglobal'
ATLAS_NAME = 'ho'

DEFAULT_ABIDE_LOCATION = DEFAULT_ABIDE_ROOT_LOCATION / ABIDE_PCP / PIPELINE / STRATEGY

if not DEFAULT_ABIDE_LOCATION.exists():
    logging.warning(f"No ABIDE DATA found at {DEFAULT_ABIDE_LOCATION}")


class AbideData():
    """ABIDE dataset wrapper (load data from disk, preprocess and provide a graph representation)
    """

    def __init__(self, folder_root: Path = DEFAULT_ABIDE_ROOT_LOCATION) -> None:
        """Assumes that data is stored in the folder_root
        Use ABIDE_dataset/download_preprocess.py to download the data
        """
        assert folder_root.exists()
        self.folder_root = folder_root / ABIDE_PCP / PIPELINE / STRATEGY
        self.indexes_path = sorted(list(self.folder_root.glob("5*")))
        self.n_patients = len(self.indexes_path)
        self.metadata_path = folder_root / ABIDE_PCP / 'Phenotypic_V1_0b_preprocessed1.csv'
        self.subject_indices = [int(subject_path.name[:5]) for subject_path in self.indexes_path]
        assert self.metadata_path.exists()

    def get_connectivity_matrix(self, index: int) -> np.ndarray:
        """Retrieve connectivity matrix for a given patient from disk

        Args:
            index (int): Patient index

        Returns:
            np.ndarray: Connectivity matrix [111 x 111]
        """
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

    def get_labels(self) -> np.ndarray:
        """Retrieve labels for each patient (DX_GROUP)

        Returns:
            np.ndarray: 1D array of labels
        """
        self.get_metadata()
        labels = []
        for subject_id in self.subject_indices:
            labels.append(self.df.loc[self.df.SUB_ID == subject_id].iloc[0].DX_GROUP)
        return np.array(labels)-1

    def get_input_feature_map(self) -> np.ndarray:
        """Retrieve feature maps

        Returns:
            np.ndarray: N inviduals x C features
        """
        mat_feat = []
        for patient_index in range(self.n_patients):
            mat_feat.append(self.get_connectivity_features(patient_index))
        return np.array(mat_feat)

    def get_metadata(self) -> pd.DataFrame:
        if not hasattr(self, "df"):
            self.df = pd.read_csv(self.metadata_path)

    def get_metadata_mask(self) -> np.ndarray:
        self.get_metadata()
        metadata_mask = np.zeros((self.n_patients, self.n_patients))

        for k, subject_id in tqdm(enumerate(self.subject_indices), total=self.n_patients, desc="Metadata mask"):
            ref = self.df.loc[self.df.SUB_ID == subject_id].iloc[0]
            for j in range(k+1, self.n_patients):
                # No self loop included (k+1)
                cand = self.df.loc[self.df.SUB_ID == self.subject_indices[j]].iloc[0]
                score = 0.
                # Link if patients have same sex
                if ref.SEX == cand.SEX:
                    score += 1.
                # Link if patients measurements were done on the same site / hospital
                if ref.SITE_ID == cand.SITE_ID:
                    score += 1.
                # constructing an undirected graph
                metadata_mask[k, j] = metadata_mask[j, k] = score

        return metadata_mask

    def get_graph_adjacency(self) -> np.ndarray:
        """Construct graph from
        - features similarity
        - metadata

        Returns:
            np.ndarray: Dense adajcency matrix
        """
        inp_feat = self.get_input_feature_map()

        distv = distance.pdist(inp_feat, metric='correlation')
        dist = distance.squareform(distv)
        cov_mean = np.mean(dist)
        sim_graph = np.exp(- dist ** 2 / (2 * cov_mean ** 2))

        mask = self.get_metadata_mask()

        return mask * sim_graph