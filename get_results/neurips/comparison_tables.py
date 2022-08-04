import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

from typing import Dict, List

import numpy as np
import pandas as pd
import wandb_utils


def main(
    project: str,
    model_type: str,
    filters: Dict,
    config_keys: List[str],
    x_axis: str = "epoch",
    normalize: bool = True,
):
    """
    Gather metrics from wandb runs given filters.

    Args:
        project: Name of project in wandb such as "mnist", "cifar10", "tiny_imagenet", "imagenet"
        model_type: Name of model type such as "MLP", "LeNet"
        filters: Filters to apply on wandb to retrieve runs
        config_keys: Entries in config to retrieve
        x_axis: Choice of horizontal axis. One of "epoch" (deafult) or "_step".
        normalize: Whether to normalize with respect to baseline.
    """
    config_keys.append("seed")

    keys_epoch = ["val/params", "val/macs", "val/top1"]
    keys_batch = ["train/batch/reg/l0_model"]
    metric_names = keys_batch + keys_epoch
    metrics_df = wandb_utils.epoch_batch_runs(
        project, filters, keys_epoch, keys_batch, config_keys
    )

    # Best (min) and last top1 error for each combination in config_keys AND each seed
    agg_metrics = metrics_df.agg(
        {
            "train/batch/reg/l0_model": "last",
            # "val/l0_full": "last",
            "val/macs": "last",
            "val/params": "last",
            "val/top1": ["min", "last"],
        }
    )

    # Aggregate the summary across seeds
    keys_minus_seed = config_keys[:-1]
    summary = agg_metrics.groupby(keys_minus_seed).agg(
        {
            ("train/batch/reg/l0_model", "last"): ["mean", "std", "count"],
            # ("val/l0_full", "last"): ["mean", "std", "count"],
            ("val/params", "last"): ["mean", "std", "count"],
            ("val/macs", "last"): ["mean", "std", "count"],
            ("val/top1", "min"): "min",  # absolute best value ever, for any seed
            ("val/top1", "last"): ["count", "mean", "min", "max", "median", "std"],
        }
    )

    for key in metric_names:
        # Compute 95% CIs
        std = summary[(key, "last", "std")]
        num_samples = summary[(key, "last", "count")]
        conf95_width = 1.96 * std / np.sqrt(num_samples)
        summary[(key, "last", "conf95_width")] = conf95_width

    extract_columns = [("val/top1", "min", "min")]
    for key in metric_names:
        extract_columns.append((key, "last", "mean"))
        extract_columns.append((key, "last", "conf95_width"))

    if normalize:
        # Normalize params and macs with respect to the baseline model
        for key in metric_names:
            if key in ["val/macs", "val/params"]:
                baselines = pd.read_csv(
                    "get_results/saved_dataframes/all_models_stats.csv"
                )
                this_baseline = baselines[baselines["model_type"] == model_type]

                summary[(key, "last", "mean")] /= this_baseline[key[4:]].item()
                summary[(key, "last", "conf95_width")] /= this_baseline[key[4:]].item()

            if key in ["train/batch/reg/l0_model", "val/macs", "val/params"]:
                # Report in percentages
                summary[(key, "last", "mean")] *= 100
                summary[(key, "last", "conf95_width")] *= 100

    return summary[extract_columns]


if __name__ == "__main__":

    # # ----------------------------------------------------------------------------------
    # #                            MLP Comparison Table
    # # ----------------------------------------------------------------------------------
    # project = "mnist"
    # model_type = "MLP"
    # print_df_path = os.path.join(BASE_PATH, "saved_dataframes/mlp_table")
    # filters = {
    #     "$and": [
    #         {"tags": "neurips"},
    #         {"config.run_group": "neurips_table"},
    #         {"config.model_type": model_type},
    #     ]
    # }
    # config_keys = ["lmbdas", "target_density"]
    # x_axis = "epoch"
    # # ----------------------------------------------------------------------------------

    # # ----------------------------------------------------------------------------------
    # #                            LeNet Comparison Table
    # # ----------------------------------------------------------------------------------
    # project = "mnist"
    # model_type = "LeNet"
    # print_df_path = os.path.join(BASE_PATH, "saved_dataframes/lenet_control")
    # filters = {
    #     "$and": [
    #         {"tags": "neurips"},
    #         {"config.run_group": "neurips_control"},
    #         {"config.model_type": model_type},
    #     ]
    # }
    # config_keys = ["lmbdas", "target_density"]
    # x_axis = "epoch"
    # # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    #                            CIFAR10 Comparison Table
    # ----------------------------------------------------------------------------------
    # project = "cifar10"
    # model_type = "ResNet-28-10"
    # print_df_path = os.path.join(BASE_PATH, "saved_dataframes/cifar10_control")
    # filters = {
    #     "$and": [
    #         {"tags": "neurips"},
    #         {"config.run_group": "control"},
    #     ]
    # }
    # config_keys = ["lmbdas", "target_density", "gates_lr"]
    # x_axis = "epoch"
    # # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    #                            CIFAR100 Comparison Table
    # ----------------------------------------------------------------------------------
    # project = "cifar100"
    # model_type = "ResNet-28-10"
    # print_df_path = os.path.join(BASE_PATH, "saved_dataframes/cifar100_table")
    # filters = {
    #     "$and": [
    #         {"tags": "neurips"},
    #         {"config.run_group": "cifar_table"},
    #     ]
    # }
    # config_keys = ["lmbdas", "target_density", "gates_lr"]
    # x_axis = "epoch"
    # # ----------------------------------------------------------------------------------

    # # ----------------------------------------------------------------------------------
    # #                            TinyImageNet Table
    # # ----------------------------------------------------------------------------------
    # project = "tiny_imagenet"
    # model_type = "L0ResNet18"
    # print_df_path = os.path.join(BASE_PATH, "saved_dataframes/tiny_imagenet_control")
    # filters = {
    #     "$and": [
    #         {"tags": "neurips"},
    #         {"config.run_group": "resnet18"},
    #     ]
    # }
    # config_keys = ["lmbdas", "target_density"]
    # x_axis = "epoch"
    # # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    #                            ImageNet Table
    # ----------------------------------------------------------------------------------
    project = "imagenet"
    model_type = "L0ResNet50"
    print_df_path = os.path.join(BASE_PATH, "saved_dataframes/imagenet_table")
    filters = {
        "$and": [
            {"tags": "rebuttal_comparison"},
        ]
    }
    config_keys = ["target_density"]
    x_axis = "epoch"
    # ----------------------------------------------------------------------------------

    # Remember to uncomment one of the projects above!
    normalize = True

    # normalize = False
    res_df = main(project, model_type, filters, config_keys, x_axis, normalize)
    if normalize:
        print_df_path += "_normalized"
    res_df.to_csv(print_df_path + ".csv")
