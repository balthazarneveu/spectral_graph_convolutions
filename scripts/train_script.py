from medigraph.train import training_loop

from medigraph.data.abide import AbideData
from medigraph.model.gcn import GCN
from medigraph.model.baseline import DenseNN
import torch
from medigraph.data.preprocess import sanitize_data, whiten
from medigraph.data.properties import INPUTS, LABELS, ADJ
from medigraph.model.metrics import plot_metrics
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_training_data(device=DEVICE):
    dat = AbideData()
    data_dict = dat.get_training_data(dimension_reduction="rfe")
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
        LABELS: lab
    }
    return training_data, adj


def train(device=DEVICE):
    training_data, adj = prepare_training_data(device=device)
    metric_dict = {}
    feat_dim = training_data[INPUTS].shape[1]
    for model_name in ["Dense", "GCN"]:
        if model_name == "GCN":
            model = GCN(feat_dim, adj, hdim=64)
        else:
            model = DenseNN(feat_dim, hdim=64)
        model.to(device)
        model, metrics = training_loop(model, training_data, device=device, n_epochs=1000)
        metric_dict[model_name] = metrics
    plot_metrics(metric_dict)


if __name__ == "__main__":
    train()
