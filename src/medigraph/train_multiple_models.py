from medigraph.train import training_loop

from medigraph.data.abide import AbideData
from medigraph.model.gcn import GCN, ChebGCN
from medigraph.model.baseline import DenseNN, DenseNNSingle
from medigraph.data.io import Dump
from pathlib import Path
from medigraph.data.properties import (
    INPUTS, LABELS, ADJ, RFE_DIM_REDUCTION, TRAIN_MASK, VAL_MASK, TEST_MASK
)
import logging
import torch
from itertools import product
from typing import List
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_training_data(device=DEVICE, dimension_reduction=None, keep_frozen_masks=True, seed=42):
    dat = AbideData()
    data_dict = dat.get_training_data(dimension_reduction=dimension_reduction)
    adj_np = data_dict[ADJ]
    lab_np = data_dict[LABELS]
    inp_np = data_dict[INPUTS]
    logging.info(f"Adjacency matrix : {adj_np.shape} [VxV]")
    logging.info(f"Labels {lab_np.shape} : [V]")
    logging.info(f"Input feature vector {inp_np.shape} : [VxF]")

    adj = torch.tensor(adj_np, dtype=torch.float32).to(device)
    inp = torch.tensor(inp_np, dtype=torch.float32).to(device)  # [V=871,  F6216]
    lab = torch.tensor(lab_np, dtype=torch.float32).to(device)  # for binary classification

    training_data = {
        INPUTS: inp,
        LABELS: lab,
        TRAIN_MASK: data_dict[TRAIN_MASK],
        VAL_MASK: data_dict[VAL_MASK],
        TEST_MASK: data_dict[TEST_MASK]
    }
    if not keep_frozen_masks:
        train_mask, val_mask, test_mask = dat.get_mask(train_ratio=0.8, val_ratio=0.1, seed=seed)
        training_data[TRAIN_MASK] = train_mask
        training_data[VAL_MASK] = val_mask
        training_data[TEST_MASK] = test_mask
    return training_data, adj


def train_multiple_configurations(
    models_list: List[str] = ["Dense", "GCN"],
    device: str = DEVICE,
    n_epochs: int = 1000,
    features_selection_list: str = [RFE_DIM_REDUCTION,],
    output_folder: Path = Path("results")
) -> dict:
    """Train multiple models for further comparison.
    Allows saving results to disk to later compare and plot
    Args:
        models_list (List[str], optional): list of models defined by a string.
        Dense-dr=0.2+h=8 means a Dense NN with a 20% dropout.
        Defaults to ["Dense", "GCN"].
        device (str, optional): cuda|cpu . Defaults to DEVICE.
        n_epochs (int, optional): number of epochs. Defaults to 1000.
        output_folder (Path, optional): results folder to cache metrics. Defaults to Path("results").


    Returns:
        dict: metrics dictionary (ordered by [model_name])
    """
    output_folder.mkdir(exist_ok=True, parents=True)
    metric_dict = {}
    optimizer_params = {
        "lr": 1.E-4,
        "weight_decay": 0.1
    }
    # noise_levels_list = [0.1, None]
    noise_levels_list = [None]
    seeds_list = [42, 43, 81, 53, 19, 708, 901, 844, 98, 55]
    for feat_kind, model_name, noise_level in product(
        features_selection_list, models_list, noise_levels_list
    ):
        exp_name = f"{model_name} {feat_kind}"
        if noise_level is not None:
            exp_name += f" noise={noise_level:.2f}"
        metric_dict[exp_name] = {}
        out_exp_path = output_folder/f"{exp_name}.pkl"
        if out_exp_path.exists():
            logging.info(f"Skipping {exp_name}")
            metric_dict[exp_name] = Dump.load_pickle(out_exp_path)
            continue
        else:
            logging.warning(f"Training {exp_name}")

        for seed in seeds_list:
            # exp_name += f"seed={seed}"
            training_data, adj = prepare_training_data(
                device=device,
                dimension_reduction=feat_kind,
                keep_frozen_masks=False,
                seed=seed
            )
            feat_dim = training_data[INPUTS].shape[1]
            if "-h=" in model_name:
                hdim = int(model_name.split("-h=")[1].split("-")[0])
            else:
                hdim = 64
            if "-dr=" in model_name:
                dropout = float(model_name.split("-dr=")[1].split("-")[0])
                logging.info(f"Dropout {dropout:.2f}")
            else:
                dropout = 0.

            if "gcn" in model_name.lower():
                model = GCN(feat_dim, adj, hdim=hdim, p_dropout=dropout)
            elif "cheb" in model_name.lower():
                model = ChebGCN(feat_dim, 1, adj.cpu().numpy(), K=3, device=device, proba_dropout=dropout)
                # model = ChebGCN(feat_dim, 1, adj.cpu().numpy(), K=3, device=device, proba_dropout=dropout, decimate=8)
            elif "dense" in model_name.lower():
                model = DenseNN(feat_dim, hdim=hdim, p_dropout=dropout)
            elif "single" in model_name.lower():
                model = DenseNNSingle(feat_dim, hdim=hdim, p_dropout=dropout)
            else:
                raise ValueError(f"Unknown model name {model_name}")
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"Total number of parameters : {total_params}")

            model.to(device)
            model, metrics = training_loop(
                model, training_data, device=device, n_epochs=n_epochs,
                noise_level=noise_level, optimizer_params=optimizer_params
            )
            metric_dict[exp_name][f"seed={seed}"] = metrics
        Dump.save_pickle(metric_dict[exp_name], out_exp_path)
    return metric_dict
