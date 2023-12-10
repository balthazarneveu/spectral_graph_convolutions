from medigraph import root_dir
from medigraph.data.io import Dump
from pathlib import Path
from medigraph.data.properties import (RFE_DIM_REDUCTION, NORMALIZED_INPUTS)
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm


def retrieve_metrics(
    config_list: List[Tuple[str, str, Optional[float]]] = [("Dense", NORMALIZED_INPUTS), ("GCN", NORMALIZED_INPUTS)],
    results_folder: Path = root_dir/"results"
) -> dict:
    """"
    config_list

    Returns:
        dict: metrics dictionary (ordered by [model_name])
    """
    metric_dict = {}
    assert results_folder.exists()
    for config in tqdm(config_list):
        if len(config) >= 1:
            model_name = config[0]
            feat_kind, noise_level = RFE_DIM_REDUCTION, None
        if len(config) >= 2:
            feat_kind = config[1]
        if len(config) >= 3:
            noise_level = config[2]
        exp_name = f"{model_name} {feat_kind}"
        if noise_level is not None:
            exp_name += f" noise={noise_level:.2f}"
        out_exp_path = results_folder/f"{exp_name}.pkl"
        if out_exp_path.exists():
            logging.info(f"Skipping {exp_name}")
            metric_dict[exp_name] = Dump.load_pickle(out_exp_path)
        else:
            logging.warning(f"Missing {exp_name}")
    return metric_dict
