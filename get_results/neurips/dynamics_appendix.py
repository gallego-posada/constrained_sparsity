import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import matplotlib.pyplot as plt
import numpy as np
from plot_style import *
from wandb_utils import epoch_batch_runs, get_metrics


def main(
    project, filters, keys_batch, keys_epoch, config_keys, x_axis, file_name, ylabels
):

    metrics = epoch_batch_runs(project, filters, keys_epoch, keys_batch, config_keys)

    fig, axs = plt.subplots(5, 2, figsize=(5, 7))
    colors = ["#FF8C42", "royalblue"]
    metric_keys = [*keys_batch, *keys_epoch]
    labels = ["No Dual Restarts", "Dual Restarts"]

    for keys, group in metrics:
        # Get the keys
        tdst = float(keys[0][2:4])
        ndr = keys[1] == "True"
        column = (
            1 if tdst == 0.7 else 0
        )  # first column for tdst=0.3, second for tdst=0.7

        label = labels[0] if ndr else labels[1]
        color = colors[0] if ndr else colors[1]
        order = 0 if ndr else 1

        shared_kwargs = {
            "label": label,
            "x": x_axis,
            "legend": False,
            "alpha": 0.8,
            "color": color,
            "zorder": order,
        }

        for i in range(len(metric_keys)):
            group.plot(
                ax=axs[i, column],
                y=metric_keys[i],
                ylabel=ylabels[i] if column == 0 else "",
                linewidth=LINEWIDTH if i < 3 else 1.5,
                **shared_kwargs
            )

            if i != 3:
                axs[i, column].set_xlabel("")
            else:
                axs[3, column].set_xlabel("Epoch")

            # Feasibility vline
            feas_value = 49 if tdst == 0.7 else 56
            ymax = np.quantile(metrics.obj[metric_keys[i]], 0.97) * 1.2
            ymin = -0.5
            axs[i, column].vlines(
                x=feas_value,
                ymin=ymin,
                ymax=ymax,
                label="First feasibility",
                linestyles="-.",
                color="black",
                alpha=0.4,
                linewidth=LINEWIDTH,
            )

            axs[i, column].grid("on", alpha=0.15)

        for idx in [1, 2]:
            # Target density hline
            axs[idx, column].hlines(
                y=tdst,
                xmin=0,
                xmax=max(metrics.obj[x_axis]),
                linestyles="--",
                color="black",
                alpha=0.4,
                linewidth=LINEWIDTH,
            )
            axs[idx, column].annotate(
                r"$\epsilon$",
                xy=(10, tdst - 0.2),
                ha="center",
                va="bottom",
                alpha=ALPHA,
                size=9,
                color="black",
            )

        axs[0, column].set_title(r"$\epsilon=$" + str(tdst))
        axs[0, column].set_ylim(-0.1, 1.0)
        axs[1, column].set_ylim(0.0, 1.0)
        axs[2, column].set_ylim(0.0, 1.1)
        axs[2, column].set_yticks([0.0, 0.5, 1.0])
        axs[3, column].set_ylim(-0.01, 0.1)
        axs[4, column].set_ylim(0.4, 1.3)

    axs[0, 1].set_ylim(-0.03, 0.33)
    axs[0, 0].set_yticks([0.0, 0.5, 1.0])
    axs[0, 1].set_yticks([0.0, 0.3])

    fig.tight_layout(pad=0.1)

    # Legend
    handles, labels = axs[1, 0].get_legend_handles_labels()
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

    # Data generating configs in configs/dynamics
    filters = {
        "$and": [
            {"config.run_group": "dr_vs_no_dr"},
            {"config.model_type": "LeNet"},
        ]
    }
    # Metrics to plot. Disentangle those logged every step from those every epoch
    # because wandb api does funny things when mixing them.
    keys_batch = ["train/batch/reg/ineq_lambda_00", "train/batch/reg/l0_model"]
    keys_epoch = ["val/l0_model", "train/epoch/loss", "val/purged_top1"]
    config_keys = ["target_density", "no_dual_restart"]
    x_axis = "epoch"

    ylabels = [
        r"Multiplier $\lambda_{co}$",
        r"Expected $L_0$",
        "Purged density",
        "Train loss",
        "Test error (%)",
    ]
    file_name = "dynamics"

    main(
        project,
        filters,
        keys_batch,
        keys_epoch,
        config_keys,
        x_axis,
        file_name,
        ylabels,
    )
