import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import copy

import numpy as np
import wandb_utils
from plot_style import *


def generate_plots(metrics, task, metric_key, project, model_type):
    # sep = True for layer-wise, False for model-wise
    sep = task == "layer"

    # Select all sep or all non-sep experiments. tdst is a string, not a list of floats
    mask_tdst = [tdst.count(",") > 0 for tdst in metrics["target_density"].values]
    mask_lmbdas = [lmbdas.count(",") > 0 for lmbdas in metrics["lmbdas"].values]
    mask = [a or b for a, b in zip(mask_tdst, mask_lmbdas)]
    if not sep:
        mask = [not m for m in mask]

    metrics = metrics[mask]

    # Get density at last epoch
    summary_logic = {metric_key: "mean"}

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

    # Filename
    constraint_type = "layerwise" if sep else "modelwise"

    filename = f"figs/control/{project}"
    os.makedirs(filename, exist_ok=True)
    filename += "/" + model_type + "_" + constraint_type

    helper_plot(co_summary, pen_summary, metric_key, filename)


def main(project, filters, model_type, task_types):

    print(f"Running project: {project}")

    # Get runs
    config_keys = ["target_density", "lmbdas", "seed"]
    metric_key = "train/batch/reg/l0_model"

    # DataFrame with the historic of the metrics across runs
    metrics = wandb_utils.get_metrics(
        project, filters, [metric_key], config_keys, "epoch"
    )

    # last epoch per run
    metrics = metrics.groupby(["target_density", "lmbdas", "seed"]).tail(1)

    for task in task_types:
        generate_plots(copy.deepcopy(metrics), task, metric_key, project, model_type)


def helper_plot(co_summary, pen_summary, key, filename):

    tdst_color = "royalblue"
    tdst_ax_color = "navy"
    lam_color = "firebrick"

    fig, ax = plt.subplots()
    co_summary.plot(
        figsize=(2, 2),
        ax=ax,
        x=key,
        y="target_density",
        s=35,
        kind="scatter",
        color=tdst_color,
        ylabel=r"Target $L_0$-density $\epsilon$",
        xlabel=r"Achieved $L_0$-density",
    )

    ax.tick_params(axis="y", colors=tdst_ax_color, labelsize=TICK_SIZE)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="x", which="both", bottom=True, colors="black", labelsize=TICK_SIZE
    )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.yaxis.label.set_color(tdst_ax_color)

    plt.plot(
        np.linspace(0, 1),
        np.linspace(0, 1),
        color=tdst_color,
        linestyle="--",
        alpha=0.4,
    )

    secax = ax.twinx()

    x_data = pen_summary[key].values
    y_data = pen_summary["lmbdas"].values

    _, stemlines, _ = plt.stem(x_data, y_data, markerfmt=" ", basefmt=" ", bottom=10)
    plt.setp(stemlines, linestyle=":", color=lam_color, linewidth=1, alpha=0.5)

    pen_summary.plot(
        ax=secax,
        x=key,
        y="lmbdas",
        marker="*",
        s=45,
        color=lam_color,
        kind="scatter",
        ylabel=r"Penalty coef. $\lambda_{pen}$",
    )

    secax.invert_yaxis()
    secax.set_yscale("log", base=10)
    secax.set_ylim(2, 5e-5)
    secax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    secax.tick_params(axis="y", colors=lam_color, labelsize=TICK_SIZE)
    secax.yaxis.label.set_color(lam_color)

    secax.tick_params(axis="x", colors="black", labelsize=TICK_SIZE)
    secax.xaxis.set_ticks_position("bottom")

    secax.spines["left"].set_color(tdst_color)
    secax.spines["right"].set_color(lam_color)

    plt.savefig(filename + ".pdf", bbox_inches="tight", dpi=1000)
    plt.savefig(filename + ".png", bbox_inches="tight", transparent=True, dpi=1000)

    plt.close()


if __name__ == "__main__":

    task_types = ["layer", "model"]

    # ----------------------------------------------------------------------------------
    #                                     MNIST
    # ----------------------------------------------------------------------------------
    for model_type in ["LeNet", "MLP"]:
        filters_mnist = {
            "$and": [
                {"config.run_group": "neurips_control"},
                {"config.model_type": model_type},
            ]
        }
        main("mnist", filters_mnist, model_type, task_types)

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
        main(project, filters_cifar, "ResNet-28-10", task_types)

    # ----------------------------------------------------------------------------------
    #                                   Tiny ImageNet
    # ----------------------------------------------------------------------------------

    filters_tiny_imagenet = {
        "$and": [
            {"config.run_group": "resnet18"},
            {"tags": "neurips"},
        ]
    }
    main("tiny_imagenet", filters_tiny_imagenet, "L0ResNet18", task_types)
