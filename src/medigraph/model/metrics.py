import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Tuple

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
LOSS = "loss"
ACCURACY = "accuracy"


def analyze_metrics(
    metric_dict: dict,
    plot_flag: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = "Comparison of Model Performances"
) -> dict:
    """Extract best metric for each run on the point with the lowest validation loss
    and optionally create a "Moustache plot" aka Tukey box plot
    of the best test accuracies.
    """
    results = {}
    all_test_acc = []
    mean_acc_labels = []
    for model_name, metric in metric_dict.items():
        best_test_acc_list = []

        for seed, current_metric in metric.items():
            best_val_loss_idx = np.argmin(current_metric[LOSS][VALIDATION])
            best_test_acc = current_metric[ACCURACY][TEST][best_val_loss_idx]
            best_test_acc_list.append(best_test_acc)

        mean_test_acc = np.mean(best_test_acc_list)
        std_test_acc = np.std(best_test_acc_list)
        results[model_name] = {"mean_test_accuracy": mean_test_acc, "std_test_accuracy": std_test_acc}
        mean_acc_labels.append(f"{model_name} ({mean_test_acc:.1%})")
        all_test_acc.append(best_test_acc_list)

    if plot_flag:
        plt.figure(figsize=figsize)
        sns.boxplot(data=all_test_acc, palette="Set2")  # Using Seaborn's palette for colors
        xlabels = [m.replace(" ", "\n") for m in metric_dict.keys()]
        plt.xticks(ticks=range(len(metric_dict)), labels=xlabels)  # Setting model names as labels
        plt.ylabel("Best Test Accuracy")
        plt.title(title)
        # Creating custom legend
        colors = sns.color_palette("Set2")
        patches = [plt.Line2D([0], [0], color=colors[i % len(colors)], marker='o', linestyle='',
                              label=label)
                   for i, label in enumerate(mean_acc_labels)]
        plt.plot(
            [-0.5, len(metric_dict.keys())-1+0.5],
            [0.695, 0.695],
            color="m", linestyle="--", alpha=0.3, label="GCN author's accuracy")
        plt.legend(handles=patches, title="Mean Accuracy", bbox_to_anchor=(0.95, 1), loc='upper left')
        
        plt.ylim(0.4, 0.75)
        plt.show()

    return results


def plot_metrics(metric_dict: dict, figsize=(10, 6)) -> None:
    """Compare training metrics of different models

    Args:
        metric_dict (dict): Dictionary of metrics for each model
        ```
        {
            "model1": {
                "seed1: {
                    "loss": {
                        "train": [float],
                        "validation": [float],
                        "test": [float]
                    },
                    "accuracy": {
                        "train": [float],
                        "validation": [float],
                        "test": [float]
                    }
                },
                "seed2: {
                    "loss": {
                        "train": [float],
                        "validation": [float],
                        "test": [float]
                    },
                    "accuracy": {
                        "train": [float],
                        "validation": [float],
                        "test": [float]
                    }
                },
            "model2": {
                "seed1: {
                    "loss": {
                        "train": [float]
                        "validation: [float]
                    },
                    "accuracy": {
                        "train": [float],
                        "validation: [float]
                    }
                },
            }
        }
        ```
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    colors = ["b", "g", "r", "y", "m", "c", "k"]
    # colors  = [""]
    for idx, (model_name, metric) in enumerate(metric_dict.items()):
        color = colors[idx % len(colors)]
        for seed_idx, seed in enumerate(metric.keys()):
            current_metric = metric[seed]
            axs[0].plot(current_metric[LOSS][TRAIN], color+"--",
                        label=None if seed_idx >= 1 else (model_name + " TRAIN"))
            axs[0].plot(current_metric[LOSS][VALIDATION], color+"-.",
                        alpha=0.8,
                        label=None if seed_idx >= 1 else (model_name + " VALIDATION"))
            axs[0].plot(current_metric[LOSS][TEST], color+"-", linewidth=2,
                        label=None if seed_idx >= 1 else (model_name + " TEST"))
            axs[1].plot(current_metric[ACCURACY][TRAIN], color+"--",
                        label=None if seed_idx >= 1 else (f"{model_name} TRAIN accuracy"))
            axs[1].plot(current_metric[ACCURACY][VALIDATION], color+"-.",
                        alpha=0.8,
                        label=None if seed_idx >= 1 else (f"{model_name} VALIDATION accuracy"))
            axs[2].plot(current_metric[ACCURACY][TEST], color+"-",
                        # linewidth=2,
                        alpha=0.1,
                        label=None if seed_idx >= 1 else (f"{model_name} TEST accuracy"))
    for idx, (model_name, metric) in enumerate(metric_dict.items()):
        color = colors[idx % len(colors)]
        acc_test = np.array([metric[seed][ACCURACY][TEST]for seed in metric.keys()]).mean(axis=0)
        axs[2].plot(
            acc_test,
            color+"-",
            linewidth=3,
            alpha=1.,
            label=f"{model_name} average TEST accuracy")
    for ax in axs:
        ax.legend()
        ax.grid()
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Binary Cross Entropy Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[2].set_ylabel("Test accuracy")
    axs[0].set_title("Losses")
    axs[1].set_title("Accuracy")
    axs[2].set_title("Test accuracy")

    plt.show()
