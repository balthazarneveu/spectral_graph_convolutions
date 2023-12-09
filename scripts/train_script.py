from medigraph.train import training_loop

from medigraph.data.abide import AbideData
from medigraph.model.gcn import GCN, ChebGCN
from medigraph.model.baseline import DenseNN, DenseNNSingle
import torch
from medigraph.data.properties import INPUTS, LABELS, ADJ, RFE_DIM_REDUCTION, TRAIN_MASK, VAL_MASK, TEST_MASK, NORMALIZED_INPUTS
from medigraph.model.metrics import plot_metrics
import logging
from itertools import product

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_training_data(device=DEVICE, dimension_reduction=None):
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
    return training_data, adj


def train(device=DEVICE):

    metric_dict = {}
    # for feat_kind, model_name in product([RFE_DIM_REDUCTION, NORMALIZED_INPUTS], ["Dense", "GCN"]):
    # for feat_kind, model_name in product([NORMALIZED_INPUTS], ["Dense", "Single"]):
    # for feat_kind, model_name in product([NORMALIZED_INPUTS], ["GCN",]):
    # for feat_kind, model_name in product([NORMALIZED_INPUTS], ["Single-1", "Single-4", "Single-8", "Single-16"]):

    for feat_kind, model_name in product([NORMALIZED_INPUTS], ["Single", "Dense"]):
        exp_name = f"{model_name} {feat_kind}"
        training_data, adj = prepare_training_data(
            device=device,
            dimension_reduction=feat_kind
        )
        feat_dim = training_data[INPUTS].shape[1]
        if model_name == "GCN":
            model = GCN(feat_dim, adj, hdim=64)
        elif model_name == "ChebNet":
            model = ChebGCN(feat_dim, 1, adj.cpu().numpy(), K=3, device=device)
        elif model_name == "Dense":
            model = DenseNN(feat_dim, hdim=16)
        elif "Single" in model_name:
            if "-" in model_name:
                hdim = int(model_name.split("-")[1])
            else:
                hdim = 64
            model = DenseNNSingle(feat_dim, hdim=hdim)
        else:
            raise ValueError(f"Unknown model name {model_name}")
        model.to(device)
        model, metrics = training_loop(model, training_data, device=device, n_epochs=1000)
        metric_dict[exp_name] = metrics
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters : {total_params}")
        logging.info(total_params)
    plot_metrics(metric_dict)


if __name__ == "__main__":
    train()
