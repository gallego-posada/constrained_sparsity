import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from wandb_utils import get_metrics

sns.set_style("white")

FONT_SIZE = 10
plt.rcParams.update({"font.family": "Times New Roman", "font.size": FONT_SIZE})


def main(project, filters_dict):

    hist_keys = ["gates_medians/layer_00"]  # first layer
    config_keys = ["target_density", "gates_lr"]
    x_axis = "epoch"

    # DataFrame with the historic of the hists
    metrics_dict = {}
    if project == "cifar10":
        snapshots = [0, 1, 2, 5, 10, 50, 100, 199]
    elif project == "mnist":
        snapshots = [0, 1, 5, 10, 50, 199]

    for name, filters in filters_dict.items():
        aux_metrics = get_metrics(project, filters, hist_keys, config_keys, x_axis)
        metrics_dict[name] = aux_metrics.iloc[snapshots]  # Filter specific epochs

    if project == "cifar10":
        fig, axs = plt.subplots(
            len(snapshots),
            len(metrics_dict),
            figsize=(len(metrics_dict), len(snapshots) / 1.5),
        )
    else:
        # Transpose
        fig, axs = plt.subplots(
            len(metrics_dict),
            len(snapshots),
            figsize=(len(snapshots), len(metrics_dict)),
        )

    for col, (name, metrics) in enumerate(metrics_dict.items()):
        for row, elem in enumerate(metrics.iterrows()):

            if project == "mnist":
                # "Transpose" subplots for mnist
                _col = row
                _row = col
            else:
                _col = col
                _row = row
            this_axis = axs[_row, _col]

            # Extract numpy histogram
            hist = elem[1][3]
            vals, bins = hist["values"], hist["bins"]
            bincentres = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)]

            # Normalize values to density
            data_range = max(bins) - min(bins)
            vals = [v * data_range / (len(bins) - 1) for v in vals]

            # Standardize bins
            num_bins = 40
            linspace = np.linspace(0, 1, num_bins)
            dig_bins = np.digitize(bincentres, linspace, right=True)

            # Count number of entries in each of the new bins
            new_vals = np.zeros(len(linspace))
            for idx, bin_idx in enumerate(dig_bins):
                new_vals[bin_idx] += vals[idx]

            if name[0] == "0.1":
                color = "#FF8C42"
            elif name[0] == "1":
                color = "#AFD5AA"
            else:
                color = "royalblue"

            this_axis.bar(
                linspace,
                new_vals,
                width=1.05 / num_bins,
                color=color,
                linestyle="-",
                label=name,
                linewidth=0.0,
                alpha=1,
            )

            # Hline reference to the size of peaks at 0 or 1
            mid_v = max(new_vals) / 2  # half the tallest bin
            this_axis.axhline(
                mid_v, 0, 1, color="black", linewidth=0.7, linestyle="--", alpha=0.2
            )
            to_annotate = str(int(mid_v * 100)) + "%"
            this_axis.annotate(
                to_annotate,
                (0.05, mid_v * 1.05),
                color="black",
                fontsize=FONT_SIZE * 0.7,
                alpha=0.5,
            )

            # Plot titles
            if _row == 0:
                if project == "cifar10":
                    if name[1] == "No Detach":
                        title = r"WD w/o $\it{sg}(\mathbf{z}$)"
                    elif name[1] == "Detach":
                        title = r"WD with $\it{sg}(\mathbf{z})$"
                else:
                    title = "Ep. " + str(snapshots[_col])

                this_axis.set_title(title, fontsize=FONT_SIZE * 0.9)

            # y axis
            this_axis.set_yticks([])
            if _col == 0:
                if project == "cifar10":
                    ylab = "Ep. " + str(snapshots[_row])
                else:
                    ylab = name[1]

                this_axis.set_ylabel(ylab, fontsize=FONT_SIZE * 0.9)

            # x axis
            this_axis.set_xticks([])
            this_axis.set_xlim(-0.05, 1.05)
            if _row + 1 == max(len(snapshots), len(metrics_dict)):
                this_axis.set_xticks([0, 0.5, 1])
                this_axis.xaxis.set_tick_params(labelsize=FONT_SIZE * 0.9)

    fig.supxlabel("Medians of stochastic gates", fontsize=FONT_SIZE)
    fig.tight_layout(pad=0.3)

    # Legend
    if name[1] != "MLP" and name[1] != "LeNet":
        custom_lines = [
            Line2D([0], [0], color="#FF8C42", lw=4),
            Line2D([0], [0], color="#AFD5AA", lw=4),
            Line2D([0], [0], color="royalblue", lw=4),
        ]
        names = ["0.1", "1", "6"]
        legend = axs[0, 0].legend(
            title=r"$\eta_{primal}^{\phi}$",
            handles=custom_lines,
            labels=names,
            loc="upper left",
            ncol=3,
            prop={"size": FONT_SIZE * 0.8},
            bbox_to_anchor=(2.6, 2.5),
            frameon=False,
        )
        legend.get_title().set_fontsize(FONT_SIZE * 0.8)
        legend.get_title().set_position((-85, -12))

    try:
        os.mkdir("figs/hists")
    except FileExistsError:
        pass

    filename = (
        "figs/hists/hist_baseline"
        if name[1] in ["MLP", "LeNet"]
        else "figs/hists/hist_series"
    )
    plt.savefig(filename + ".pdf", bbox_inches="tight", dpi=1000)
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=1000, transparent=True)

    plt.close()


if __name__ == "__main__":

    # LeNet vs MLP
    project = "mnist"
    filters_dict = {
        ("None", "MLP"): {
            "$and": [
                {"config.model_type": "MLP"},
                {"config.run_group": "gates_baseline"},
            ]
        },
        ("None", "LeNet"): {
            "$and": [
                {"config.model_type": "LeNet"},
                {"config.run_group": "gates_baseline"},
            ]
        },
    }

    # # ResNet, only changing gates_lr nor detach
    project = "cifar10"
    filters_dict = {
        ("0.1", "No Detach"): {
            "$and": [
                {"config.run_group": "gates_detach"},
                {"config.gates_lr": 0.1},
                {"config.l2_detach_gates": False},
            ]
        },
        ("0.1", "Detach"): {
            "$and": [
                {"config.run_group": "gates_detach"},
                {"config.gates_lr": 0.1},
                {"config.l2_detach_gates": True},
            ]
        },
        ("1", "No Detach"): {
            "$and": [
                {"config.run_group": "gates_detach"},
                {"config.gates_lr": 1},
                {"config.l2_detach_gates": False},
            ]
        },
        ("1", "Detach"): {
            "$and": [
                {"config.run_group": "gates_detach"},
                {"config.gates_lr": 1},
                {"config.l2_detach_gates": True},
            ]
        },
        ("6", "No Detach"): {
            "$and": [
                {"config.run_group": "gates_detach"},
                {"config.gates_lr": 6},
                {"config.l2_detach_gates": False},
            ]
        },
        ("6", "Detach"): {
            "$and": [
                {"config.run_group": "gates_detach"},
                {"config.gates_lr": 6},
                {"config.l2_detach_gates": True},
            ]
        },
    }

    main(project, filters_dict)
