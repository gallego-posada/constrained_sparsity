from cmath import nan
import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import matplotlib.pyplot as plt
import numpy as np
from plot_style import *
from wandb_utils import get_metrics


def main(
    project,
    filters,
    metric_keys,
    config_keys,
    x_axis,
    file_name,
    ylabels,
):

    metrics = get_metrics(project, filters, metric_keys, config_keys, x_axis)

    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))

    metrics = metrics.fillna("0.3")
    groups = metrics.groupby("droprate_init")

    for droprate, group in groups:
        for i in range(len(metric_keys)):

            if droprate == "0.5":
                label = "Dense Baseline"
                color = "#AFD5AA"
            elif droprate == "0.3":
                label = r"$\rho_{\mathrm{init}} = 0.3$"
                color = "#FF8C42"
            else:
                label = r"$\rho_{\mathrm{init}} = 0.05$"
                color = "royalblue"

            group.plot(
                ax=axs[i],
                x=x_axis,
                y=metric_keys[i],
                ylabel=ylabels[i],
                label=label,
                color=color,
                legend=False,
                alpha=0.8,
                linewidth=LINEWIDTH / 1.5,
            )

            axs[i].set_xlabel("Epoch")
            axs[i].grid("on", alpha=0.15)
            axs[i].set_ylim(20, 80)

    fig.tight_layout(pad=0.9)

    # Legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles=handles[::-1], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.01)
    )

    try:
        os.mkdir("figs/droprate/")
    except FileExistsError:
        pass

    file_path = "figs/droprate/" + file_name
    plt.savefig(file_path + ".png", bbox_inches="tight", dpi=1000, transparent=True)
    plt.savefig(file_path + ".pdf", bbox_inches="tight", dpi=1000)
    plt.close()


if __name__ == "__main__":

    # Dataset/wandb project
    project = "imagenet"

    # Data generating configs in configs/dynamics
    filters = {
        "$and": [
            {"tags": "droprate"},
        ]
    }
    # Metrics to plot.
    keys_epoch = ["train/epoch/top1", "val/top1"]
    config_keys = ["droprate_init"]
    x_axis = "epoch"

    ylabels = [
        "Training Error (%)",
        "Validation Error (%)",
    ]
    file_name = "droprate"

    main(
        project,
        filters,
        keys_epoch,
        config_keys,
        x_axis,
        file_name,
        ylabels,
    )
