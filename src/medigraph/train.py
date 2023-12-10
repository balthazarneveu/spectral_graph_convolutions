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
from medigraph.model.metrics import TRAIN, VALIDATION, TEST
from typing import Tuple, Optional


def training_loop(
    model: torch.nn.Module,
    train_data: dict, device,
    noise_level: Optional[float] = None,
    optimizer_params={
        "lr": 1.E-4,
        "weight_decay": 0.1
    },
    n_epochs: Optional[int] = 1000
) -> Tuple[torch.nn.Module, dict]:

    criterion = torch.nn.BCEWithLogitsLoss()
    train_input = train_data[INPUTS]
    train_label = train_data[LABELS]

    train_mask = train_data[TRAIN_MASK]
    val_mask = train_data[VAL_MASK]
    test_maks = train_data[TEST_MASK]

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), **optimizer_params)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.999)
    losses = {TRAIN: [], VALIDATION: [], TEST: []}
    accuracies = {TRAIN: [], VALIDATION: [], TEST: []}
    for ep in tqdm(range(n_epochs)):
        model.train()
        optim.zero_grad()
        if noise_level is not None:
            train_input_noisy = train_input + noise_level * torch.randn_like(train_input)
        else:
            train_input_noisy = train_input
        logit = model(train_input_noisy)
        loss = criterion(logit[train_mask], train_label[train_mask])
        loss.backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            for mode, mask in zip([TRAIN, VALIDATION, TEST], [train_mask, val_mask, test_maks]):
                if mask is None:
                    continue
                loss = criterion(logit[mask], train_label[mask])
                predicted_prob = torch.sigmoid(logit[mask]).squeeze()  # Apply sigmoid and remove extra dimensions
                predicted = (predicted_prob >= 0.5).long()  # Convert probabilities to 0 or 1
                correct = (predicted == train_label[mask]).sum().item()
                total = len(mask)
                accuracy = correct / total
                losses[mode].append(loss.detach().cpu())
                accuracies[mode].append(accuracy)
        if ep % 100 == 0:
            print(
                f"Epoch {ep}\nloss: train {losses[TRAIN][-1]:.5f} | validation {losses[VALIDATION][-1]:.5f}" +
                f"\naccuracy: train {accuracies[TRAIN][-1]:.2%} | validation {accuracies[VALIDATION][-1]:.2%}")
            print("Learning rate", optim.param_groups[0]['lr'])
        # scheduler.step(losses[VALIDATION][-1])
        # scheduler.step()
        
    metrics = {"loss": losses, "accuracy": accuracies}
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
