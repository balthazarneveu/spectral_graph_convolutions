import matplotlib.pyplot as plt


def plot_metrics(metric_dict: dict) -> None:
    """Compare training metrics of different models

    Args:
        metric_dict (dict): Dictionary of metrics for each model
        ```
        {
            "model1": {
                "training_losses": [float],
                "training_accuracies": [float]
                },
            "model2": {
                "training_losses": [float],
                "training_accuracies": [float]
            }
        }
        ```
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    for model_name, metric in metric_dict.items():
        axs[0].plot(metric["training_losses"], label=model_name)
        axs[1].plot(metric["training_accuracies"], label=f"{model_name} accuracy")
    for ax in axs:
        ax.legend()
        ax.grid()
    axs[0].set_title("Training loss (Binary Cross Entropy)")
    axs[1].set_title("Accuracy")

    plt.show()
