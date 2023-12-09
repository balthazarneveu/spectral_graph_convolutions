"""Training loop
"""
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# try:
#     __IPYTHON__  # noqa: F821
#     from tqdm.notebook import tqdm
# except Exception as _e:  # noqa: F841
#     from tqdm import tqdm

from medigraph.data.properties import INPUTS, LABELS, TRAIN_MASK, VAL_MASK, TEST_MASK
from typing import Tuple, Optional


def training_loop(
    model: torch.nn.Module,
    train_data: dict, device,
    valid_data: Optional[dict] = None,
    optimizer_params={
        "lr": 1.E-4,
        "weight_decay": 0.1
    },
    n_epochs: Optional[int] = 1000
) -> Tuple[torch.nn.Module, dict]:
    criterion = torch.nn.BCEWithLogitsLoss()
    train_input = train_data[INPUTS]
    train_label = train_data[LABELS]
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), **optimizer_params)
    training_losses = []
    training_accuracies = []
    for ep in tqdm(range(n_epochs)):
        model.train()
        optim.zero_grad()
        logit = model(train_input)
        loss = criterion(logit, train_label)
        loss.backward()
        optim.step()
        with torch.no_grad():
            predicted_prob = torch.sigmoid(logit).squeeze()  # Apply sigmoid and remove extra dimensions if any
            predicted = (predicted_prob >= 0.5).long()  # Convert probabilities to 0 or 1
            correct = (predicted == train_label).sum().item()
            total = train_label.shape[0]
            accuracy = correct / total
            training_accuracies.append(accuracy)
        if ep % 100 == 0:
            print(f"Epoch {ep} loss: {loss.item():10f} - accuracy: {accuracy:.2%}")
        training_losses.append(loss.detach().cpu())
    metrics = {
        "training_losses": training_losses,
        "training_accuracies": training_accuracies
    }
    return model, metrics

# ---- METRICS AND TEST ----


def compute_accuracy(output, labels):

    pred_label = (output >= 0.5).long()
    correct = pred_label.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def compute_metrics(model, data, criterion=torch.nn.BCEWithLogitsLoss(), mask: str = TEST_MASK):

    model.eval()
    input_feat = data[INPUTS]
    labels = data[LABELS]
    test_mask = data[mask]
    print(f'=== Computing metrics on {len(test_mask)} test nodes ===')
    output_feat = model(input_feat)
    loss_test = criterion(output_feat[test_mask], labels[test_mask])
    acc_test = compute_accuracy(output_feat[test_mask], labels[test_mask])
    return loss_test, acc_test

# ---- PLOTTING ----


def plot_learning_curves(train_log, val_log, title="Training"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    metrics = ["losses", "accuracies"]
    for i in range(len(metrics)):
        ax[i].plot(train_log[metrics[i]], label="train")
        ax[i].plot(val_log[metrics[i]], label="val")
        ax[i].legend()
        ax[i].set_title(metrics[i].upper())
        ax[i].set_xlabel("Epochs")

    fig.suptitle(title)

# --- TRAIN LOOP FOR BINARY CLASSIFIER ---


def train(model: torch.nn.Module,
          data: dict,
          nEpochs: Optional[int] = 150,
          criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
          optimizer_params: Optional[dict] = {"lr": 5.E-3,
                                              "weight_decay": 5.E-4}
          ) -> Tuple[torch.nn.Module, dict, dict]:

    optim = torch.optim.Adam(model.parameters(), **optimizer_params)
    train_log = {
        "losses": [],
        "accuracies": []
    }
    val_log = {
        "losses": [],
        "accuracies": []
    }

    input_feat = data[INPUTS]
    labels = data[LABELS]
    train_mask = data[TRAIN_MASK]
    val_mask = data[VAL_MASK]

    model.train()
    for _ in tqdm(range(nEpochs), total=nEpochs, desc="Training"):
        optim.zero_grad()

        output_feat = model(input_feat)
        assert output_feat[train_mask].requires_grad == True

        loss_train = criterion(output_feat[train_mask], labels[train_mask])

        loss_train.backward()

        # debugging training
        # for name, param in model.named_parameters():
        #     print(f" Paremeter {name}, gradient norm : {param.grad.norm()}")

        optim.step()

        with torch.no_grad():
            # model.eval()
            acc_train = compute_accuracy(output_feat[train_mask], labels[train_mask])
            loss_val = criterion(output_feat[val_mask], labels[val_mask])
            acc_val = compute_accuracy(output_feat[val_mask], labels[val_mask])
            train_log["losses"].append(loss_train.item())
            train_log['accuracies'].append(acc_train.item())
            val_log['losses'].append(loss_val.item())
            val_log['accuracies'].append(acc_val.item())

    return model, train_log, val_log
