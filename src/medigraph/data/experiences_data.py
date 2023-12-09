from pathlib import Path
from medigraph.data.io import Dump
from medigraph.data.abide import AbideData
from medigraph.data.properties import ADJ, INPUTS, LABELS, TRAIN_MASK, VAL_MASK, TEST_MASK
from medigraph.data.preprocess import sanitize_data, whiten
import numpy as np
import torch

root_folder = Path(__file__).parent.parent.parent.parent.absolute()
DEFAULT_ABIDE_ROOT_LOCATION = root_folder/ "__ABIDE_dataset"

def get_training_dict_exp1(data : AbideData,
             nb_train: int = 800, 
             nb_val: int = 50,
             override: bool=False,
             folder_root: Path = DEFAULT_ABIDE_ROOT_LOCATION):
    
    training_dict_path = folder_root / "data_dict_exp1.pkl"

    if training_dict_path.exists() and not override:
        data_dict = Dump.load_pickle(training_dict_path)
    
    else:

        signals, labels, adj = data.get_training_data(override=override)  

        clean_inp = sanitize_data(torch.tensor(signals))
        inp = np.array(whiten(clean_inp))

        # get random masks
        shuffle_nodes = np.random.permutation(range(inp.shape[0]))
        train_mask = shuffle_nodes[:nb_train]
        val_mask = shuffle_nodes[nb_train:nb_train+nb_val]
        test_mask = shuffle_nodes[nb_train+nb_val:]

        data_dict = {
            INPUTS : inp,
            LABELS : labels,
            TRAIN_MASK : train_mask,
            VAL_MASK : val_mask,
            TEST_MASK : test_mask,
            ADJ : adj
        }
        Dump.save_pickle(data_dict, training_dict_path)

    return data_dict

def get_training_dict_exp2(data : AbideData,
             nb_train: int = 800, 
             nb_val: int = 50,
             nbFeatures: int = 2000,
             override: bool=False,
             folder_root: Path = DEFAULT_ABIDE_ROOT_LOCATION):   

    training_dict_path = folder_root / "data_dict_exp2.pkl"

    if training_dict_path.exists() and not override:
        data_dict = Dump.load_pickle(training_dict_path)

    else:
        N = data.n_patients
        assert nb_train + nb_val < N, f"Not enough patients to split the {nb_train+nb_val} data"
        
        shuffle_nodes = np.random.permutation(N)
        train_mask = shuffle_nodes[:nb_train]
        val_mask = shuffle_nodes[nb_train:nb_train+nb_val]
        test_mask = shuffle_nodes[nb_train+nb_val:]

        adj = data.get_graph_adjacency()  # [V, V]
        labels = data.get_labels()

        mask_classifier = np.random.choice(train_mask, 300, replace=False)
        selected_feat = data.get_selectedRidge_features(mask_classifier, n_features_to_select=nbFeatures)

        data_dict = {
            ADJ : adj,
            INPUTS : selected_feat,
            LABELS : labels,
            TRAIN_MASK : train_mask,
            VAL_MASK : val_mask,
            TEST_MASK : test_mask
            }
        Dump.save_pickle(data_dict, training_dict_path)
    return data_dict



