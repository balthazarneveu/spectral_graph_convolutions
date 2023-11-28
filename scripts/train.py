from medigraph.train import training_loop

from medigraph.data.abide import AbideData
from medigraph.model.gcn import GCN
from medigraph.model.baseline import DenseNN
import torch
from medigraph.data.preprocess import sanitize_data, whiten
from medigraph.data.properties import INPUTS, LABELS
from medigraph.model.metrics import plot_metrics
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_training_data(device=DEVICE):
    dat = AbideData()
    inp_np, lab_np, adj_np = dat.get_training_data()
    logging.info(f"Adjacency matrix : {adj_np.shape} [VxV]")
    logging.info(f"Labels {lab_np.shape} : [V]")
    logging.info(f"Input feature vector {inp_np.shape} : [VxF]")

    adj = torch.tensor(adj_np, dtype=torch.float32).to(device)
    inp_raw = torch.tensor(inp_np, dtype=torch.float32).to(device)  # [V=871,  F6216]
    lab = torch.tensor(lab_np, dtype=torch.float32).to(device)  # for binary classification
    clean_inp = sanitize_data(inp_raw)
    inp = whiten(clean_inp)
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
