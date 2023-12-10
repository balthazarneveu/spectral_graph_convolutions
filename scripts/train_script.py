from pathlib import Path
from medigraph.data.properties import (
    RFE_DIM_REDUCTION, RAW_INP, NORMALIZED_INPUTS
)
import argparse
from medigraph.model.metrics import plot_metrics, analyze_metrics
from medigraph.train_multiple_models import train_multiple_configurations as train
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    models_list = ["Dense", "Dense-dr=0.1", "Dense-dr=0.2", "Dense-dr=0.3"]
    models_list = ["GCN-dr=0.1",]
    # models_list = ["GCN", "GCN-dr=0.3"]
    models_list = ["Single-h=1", "Single-h=4", "Single-h=8", "Single-h=16", "Dense", "Single-h=128"]
    # models_list+= ["GCN-dr=0.1",]
    # models_list += ["Cheb-dr=0.3"]

    parser = argparse.ArgumentParser(description="Train classification models on Abide dataset - compare performances")
    parser.add_argument("-d", "--device", type=str,
                        choices=["cpu", "cuda"], default=str(DEVICE), help="Training device")
    parser.add_argument("-n", "--n-epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("-o", "--output-folder", type=str, default="results", help="Output results folder"),
    parser.add_argument("-m", "--models-list", nargs="+", default=models_list, help="List of models to train")
    parser.add_argument("-f", "--features-selection", nargs="+", default=[RFE_DIM_REDUCTION],
                        choices=[RFE_DIM_REDUCTION, NORMALIZED_INPUTS, RAW_INP],
                        help="List of input feature reduction methods")
    args = parser.parse_args()
    metric_dict = train(
        models_list=args.models_list,
        device=args.device,
        n_epochs=args.n_epochs,
        features_selection_list=args.features_selection,
        output_folder=Path(args.output_folder)
    )
    analyze_metrics(metric_dict, plot_flag=True)
    plot_metrics(metric_dict)


if __name__ == "__main__":
    main()
