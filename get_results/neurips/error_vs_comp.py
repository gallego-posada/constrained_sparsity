import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import copy

import pandas as pd
from plot_style import *
from wandb_utils import get_metrics


def generate_plots(metrics, task, metric_keys, project, model_type, x_axis):

    # sep = True for layer-wise, False for model-wise
    sep = task == "layer"

    error_key, val_key = metric_keys

    # Select all sep or all non-sep experiments. tdst is a string, not a list of floats
    mask_tdst = [tdst.count(",") > 0 for tdst in metrics["target_density"].values]
    mask_lmbdas = [lmbdas.count(",") > 0 for lmbdas in metrics["lmbdas"].values]
    mask = [a or b for a, b in zip(mask_tdst, mask_lmbdas)]
    if not sep:
        mask = [not m for m in mask]
    metrics = metrics[mask]

    # -------------------------------------------- Agg across seeds
    # Get density at last epoch
    summary_logic = {key: "mean" for key in metric_keys}

    co_metrics = metrics.loc[metrics["target_density"].ne("None")]
    co_summary = co_metrics.groupby(["target_density"]).agg(summary_logic)

    pen_metrics = metrics.loc[metrics["lmbdas"].ne("None")]
    pen_summary = pen_metrics.groupby(["lmbdas"]).agg(summary_logic)

    # Numeric index for plot from string tdsts and lmbdas
    if sep:
        tdsts = [i.split(",")[0][1:] for i in co_summary.index]
        lmbdas = [i.split(",")[0][1:] for i in pen_summary.index]
    else:
        tdsts = [i.split(",")[0][1:-1] for i in co_summary.index]
        lmbdas = [i.split(",")[0][1:-1] for i in pen_summary.index]

    co_summary["target_density"] = [float(elem) for elem in tdsts]
    pen_summary["lmbdas"] = [float(elem) for elem in lmbdas]

    # -------------------------------------------- Normalize macs and params with baseline
    # Get baseline
    baselines = pd.read_csv("get_results/saved_dataframes/all_models_stats.csv")
    this_baseline = baselines[baselines["model_type"] == model_type]

    # Normalize
    co_summary[val_key] /= this_baseline[x_axis].item()
    pen_summary[val_key] /= this_baseline[x_axis].item()
    # In percentage
    co_summary[val_key] *= 100
    pen_summary[val_key] *= 100

    # # Increase values of size_key to make sizes reasonable
    # augment_factor = 40.0
    # co_summary[size_key] *= augment_factor
    # # Penalization uses starts which appear smaller than dots by default, hence
    # # we augment some more.
    # pen_summary[size_key] *= augment_factor * 2

    # -------------------------------------------- Plot
    # loop over all density measures: across layers and at the model lev

    # Filename

    filename = f"figs/error_vs_{x_axis}/{project}"
    os.makedirs(filename, exist_ok=True)

    constraint_type = "layerwise" if sep else "modelwise"
    filename += "/" + model_type + "_" + constraint_type

    helper_plot(co_summary, pen_summary, error_key, val_key, filename)


def main(project, filters, model_type, task_types, x_axis):
    # x_axis (str): "macs" or "params"

    print(f"Running project: {project}")

    config_keys = ["target_density", "lmbdas", "seed"]
    metric_keys = ["val/top1", "val/" + x_axis]
    metrics = get_metrics(project, filters, metric_keys, config_keys, "epoch")
    metrics = metrics.groupby(config_keys)

    metrics = metrics.tail(1)
    metrics = metrics.reset_index()

    for task in task_types:
        generate_plots(
            copy.deepcopy(metrics), task, metric_keys, project, model_type, x_axis
        )


def helper_plot(co_summary, pen_summary, error_key, axis_key, filename, size_key=None):

    tdst_color = "royalblue"
    lam_color = "firebrick"

    xlabel = "Parameters (%)" if axis_key == "val/params" else "MACs (%)"

    fig, ax = plt.subplots()
    co_summary.plot(
        figsize=(2, 2),
        ax=ax,
        x=axis_key,
        y=error_key,
        s=35,
        # s=size_key,
        kind="scatter",
        color=tdst_color,
        # label="Constrained",
    )

    pen_summary.plot(
        ax=ax,
        x=axis_key,
        y=error_key,
        s=45,
        # s=size_key,
        marker="*",
        color=lam_color,
        kind="scatter",
        # label="Penalized",
    )

    ax.set_xlabel(xlabel)
    ax.tick_params(
        axis="x", which="both", bottom=True, colors="black", labelsize=TICK_SIZE
    )
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlim(-5, 105)

    ax.tick_params(axis="y", which="both", colors="black", labelsize=TICK_SIZE)
    ax.set_ylabel("Error (%)")

    plt.grid("on", alpha=0.15)

    plt.savefig(filename + ".pdf", bbox_inches="tight", dpi=1000)
    plt.savefig(filename + ".png", bbox_inches="tight", transparent=True, dpi=1000)
    plt.close()


if __name__ == "__main__":

    task_types = ["layer", "model"]

    for x_axis in ["params", "macs"]:

        # ----------------------------------------------------------------------------------
        #                                    MNIST
        # ----------------------------------------------------------------------------------
        for model_type in ["LeNet", "MLP"]:
            filters_mnist = {
                "$and": [
                    {"config.model_type": model_type},
                    {"config.run_group": "neurips_control"},
                ]
            }
            main("mnist", filters_mnist, model_type, task_types, x_axis)

        # ----------------------------------------------------------------------------------
        #                                    CIFAR 10/100
        # ----------------------------------------------------------------------------------

        for project in ["cifar10", "cifar100"]:
            filters_cifar = {
                "$and": [
                    {"config.run_group": "control"},
                    {"tags": "neurips"},
                ]
            }
            main(project, filters_cifar, "ResNet-28-10", task_types, x_axis)

        # ----------------------------------------------------------------------------------
        #                                   Tiny ImageNet
        # ----------------------------------------------------------------------------------

        filters_tiny_imagenet = {
            "$and": [
                {"config.run_group": "resnet18"},
                {"tags": "neurips"},
            ]
        }
        main("tiny_imagenet", filters_tiny_imagenet, "L0ResNet18", task_types, x_axis)
