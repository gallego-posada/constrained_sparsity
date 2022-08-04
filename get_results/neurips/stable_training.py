import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import numpy as np
import wandb_utils
from plot_style import *


def main(project, filters, model_type, task_type, metric_key):

    print(f"Running project: {project}")

    # Get runs
    config_keys = ["target_density", "seed"]

    # DataFrame with the historic of the metrics across runs
    metrics = wandb_utils.get_metrics(
        project, filters, [metric_key], config_keys, "epoch"
    )

    # Remove penalized runs
    metrics = metrics[metrics["target_density"] != "None"]

    # Select seed = 1
    metrics = metrics[metrics["seed"] == "1"]

    # sep = True for layer-wise, False for model-wise
    sep = task_type == "layer"

    # Select all sep or all non-sep experiments. tdst is a string, not a list of floats
    mask = [tdst.count(",") > 0 for tdst in metrics["target_density"].values]
    if not sep:
        mask = [not m for m in mask]
    metrics = metrics[mask]

    # Show only the first num_epoch epochs
    num_epochs = 75
    metrics["epoch"] = metrics["epoch"].apply(lambda x: float(x))
    metrics = metrics[metrics["epoch"] <= num_epochs]

    metrics = metrics.groupby(["target_density"])

    # Numeric index for plot from string tdsts and lmbdas
    tdsts = metrics.groups.keys()
    if sep:
        tdsts = [i.split(",")[0][1:] for i in tdsts]
    else:
        tdsts = [i.split(",")[0][1:-1] for i in tdsts]

    # Plot
    filename = f"figs/stable_training"
    os.makedirs(filename, exist_ok=True)
    constraint_type = "layerwise" if sep else "modelwise"
    filename += "/" + model_type + "_" + constraint_type

    helper_plot(project, metrics, metric_key, filename, tdsts, num_epochs)


def helper_plot(
    project, metrics, metric_key, filename, tdsts, num_epochs, annotate_tdst=False
):

    colors = ["lightblue", "cornflowerblue", "blue", "mediumblue", "darkblue"]

    _, ax = plt.subplots(figsize=(3, 1.8))

    for i, (_, group) in enumerate(metrics):
        group.plot(
            ax=ax,
            x="epoch",
            y=metric_key,
            color=colors[i],
            label=tdsts[i],
            linewidth=1.5,
            alpha=ALPHA + 0.3,
            zorder=1,
        )

        clean_tdst = float(tdsts[i])

        ax.hlines(
            y=clean_tdst,
            xmin=0,
            xmax=num_epochs,
            linewidth=1,
            alpha=ALPHA,
            color="black",
            linestyles="dashed",
            zorder=0,
        )
        if annotate_tdst:
            ax.annotate(
                r"$\epsilon = $" + tdsts[i],
                (num_epochs * 0.75, clean_tdst - 0.08),
                fontsize=TICK_SIZE,
            )

    # Legend location, size
    if project == "cifar10":
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[::-1],
            labels[::-1],
            title=r"$\epsilon$",
            loc="upper left",
            bbox_to_anchor=(1.05, 1.06),
            fontsize=TICK_SIZE,
        )
    else:
        ax.get_legend().remove()

    ax.set_ylabel(r"$L_0$-density")
    ax.set_xlabel("Epoch")

    ax.set_ylim(0, 1)

    xticks = [
        x for x in [0, 25, 50, 75, 100, 125, 150, 175, 200] if x <= num_epochs + 5
    ]
    ax.set_xticks(xticks)
    ax.tick_params(
        axis="x", which="both", bottom=True, colors="black", labelsize=TICK_SIZE
    )
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    plt.grid("on", alpha=0.15)

    plt.savefig(filename + ".pdf", bbox_inches="tight", dpi=1000)
    plt.savefig(filename + ".png", bbox_inches="tight", transparent=True, dpi=1000)


if __name__ == "__main__":

    # ----------------------------------------------------------------------------------
    #                                    CIFAR 10
    # ----------------------------------------------------------------------------------

    # project = "cifar10"
    # filters = {
    #     "$and": [
    #         {"config.run_group": "control"},
    #         {"tags": "neurips"},
    #     ]
    # }
    # model = "ResNet-28-10"
    # task_type = "layer"
    # key = "train/batch/reg/l0_layer_11"

    # ----------------------------------------------------------------------------------
    #                                   Tiny ImageNet
    # ----------------------------------------------------------------------------------

    # project = "tiny_imagenet"
    # filters = {
    #     "$and": [
    #         {"config.run_group": "resnet18"},
    #         {"tags": "neurips"},
    #     ]
    # }
    # model = "L0ResNet18"
    # task_type = "model"
    # key = "train/batch/reg/l0_model"

    # Uncomment one above
    main(project, filters, model, task_type, key)
