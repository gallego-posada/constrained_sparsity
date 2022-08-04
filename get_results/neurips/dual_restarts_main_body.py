import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import matplotlib.pyplot as plt
import numpy as np
from plot_style import *
from wandb_utils import epoch_batch_runs, get_metrics


def main(
    project,
    filters,
    tdst,
    keys_batch,
    keys_epoch,
    config_keys,
    x_axis,
    file_name,
    ylabels,
):

    metrics = epoch_batch_runs(project, filters, keys_epoch, keys_batch, config_keys)

    fig, axs = plt.subplots(1, 3, figsize=(9, 2))
    metric_keys = [*keys_batch, *keys_epoch]

    labels = ["No Dual Restarts", "Dual Restarts"]
    colors = ["#FF8C42", "royalblue"]

    for keys, group in metrics:
        # Get the keys
        ndr = keys[1] == "True"
        label = labels[0] if ndr else labels[1]
        color = colors[0] if ndr else colors[1]
        order = 0 if ndr else 1  # keep ndr in background

        shared_kwargs = {
            "label": label,
            "x": x_axis,
            "legend": False,
            "alpha": 0.8,
            "linewidth": LINEWIDTH,
            "color": color,
            "zorder": order,
        }

        for i in range(len(metric_keys)):

            group.plot(ax=axs[i], y=metric_keys[i], ylabel=ylabels[i], **shared_kwargs)

            axs[i].set_xlabel("Epoch")

            # Feasibility vline
            feas_value = 49 if tdst == 0.7 else 56
            ymax = np.quantile(metrics.obj[metric_keys[i]], 0.97) * 1.2
            ymin = -0.5
            axs[i].vlines(
                x=feas_value,
                ymin=ymin,
                ymax=ymax,
                label="First feasibility",
                linestyles="-.",
                color="black",
                alpha=0.4,
                linewidth=LINEWIDTH,
            )
            axs[i].grid("on", alpha=0.15)

        # Target density hline
        axs[0].hlines(
            y=tdst,
            xmin=0,
            xmax=max(metrics.obj[x_axis]),
            linestyles="--",
            color="black",
            alpha=0.4,
            linewidth=LINEWIDTH,
        )

        axs[0].set_ylim(0.0, 1.0)
        axs[1].set_ylim(-0.1, 1.0)
        axs[2].set_ylim(-0.01, 0.1)

    fig.tight_layout(pad=0.3)

    # Legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles=handles[:-1], loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01)
    )

    try:
        os.mkdir("figs/restarts/")
    except FileExistsError:
        pass

    file_path = "figs/restarts/" + file_name
    plt.savefig(file_path + ".png", bbox_inches="tight", dpi=1000, transparent=True)
    plt.savefig(file_path + ".pdf", bbox_inches="tight", dpi=1000)
    plt.close()


if __name__ == "__main__":

    # Dataset/wandb project
    project = "mnist"

    # targeted density
    tdst = 0.3

    # Data generating configs in configs/dynamics
    filters = {
        "$and": [
            {"config.run_group": "dr_vs_no_dr"},
            {"config.model_type": "LeNet"},
            {"config.target_density": [tdst]},
        ]
    }
    # Metrics to plot. Disentangle those logged every step from those every epoch
    # because wandb api does funny things when mixing them.
    keys_batch = ["train/batch/reg/l0_model", "train/batch/reg/ineq_lambda_00"]
    keys_epoch = ["train/epoch/loss"]
    config_keys = ["target_density", "no_dual_restart"]
    x_axis = "epoch"

    ylabels = [
        r"$L_0$-density",
        r"Multiplier $\lambda_{co}$",
        "Train loss",
    ]
    file_name = "dual_restarts"

    main(
        project,
        filters,
        tdst,
        keys_batch,
        keys_epoch,
        config_keys,
        x_axis,
        file_name,
        ylabels,
    )
