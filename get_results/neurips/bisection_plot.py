import sys

sys.path.append(".")

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

from adjustText import adjust_text

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})
my_cmap = cm.cividis.copy()
my_cmap.set_over("navy")
my_cmap.set_under("gold")


def main(training_level):

    tdst = 0.5  # target_density

    # Results at 0.01 significante, obtained by running sparse/bisection.py with
    # mnist_bisection.yml config.
    if training_level == "model":
        name = "model_level"
        log_lam = [-3.0, 0.0, -1.5, -0.75, -0.375, -0.5625]
        density = [
            0.9228633046150208,
            0.2802237272262573,
            0.7473313212394714,
            0.5584232807159424,
            0.4263841211795807,
            0.5006365180015564,
        ]

    if training_level == "layer":
        name = "layer_level"
        log_lam = [-3.0, 0.0, -1.5, -0.75, -0.375, -0.5625, -0.65625]
        density = [
            0.9228633046150208,
            0.2683962881565094,
            0.7141464948654175,
            0.5378358960151672,
            0.4170256555080414,
            0.4773602783679962,
            0.5060443878173828,
        ]

    lam = [10**elem for elem in log_lam]  # actual lam for plot

    # Iteration numbers. We have two initial values (iter 0)
    iters = [i - 1 for i in range(len(lam))]
    iters[0] = 0

    # all points black
    c_kwargs = {"color": "black"}

    size = 14  # to be used for markers and annotations

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    plt.scatter(
        x=lam, y=density, marker="*", s=size * 5, linewidth=0.5, alpha=0.7, **c_kwargs
    )

    # Iteration number to each point
    texts = []
    for i in range(len(lam)):
        texts.append(plt.text(lam[i], density[i], str(iters[i])))

    adjust_text(texts)

    # Hline to indicate target density
    ax.hlines(
        y=tdst, xmin=min(lam), xmax=max(lam), linestyles=":", color="black", alpha=0.4
    )
    ax.annotate(
        "Target density",
        xy=(5 * min(lam), tdst + 0.01),
        ha="center",
        va="bottom",
        alpha=0.6,
        size=size,
        color="black",
    )

    ax.set_xscale("log", base=10)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_xlabel(r"Penalty coef. $\lambda_{pen}$")
    ax.set_ylabel("Achieved density")
    ax.set_ylim(0, 1)

    folder = "figs/bisection"
    os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/{name}_plot"
    plt.savefig(filename + ".png", bbox_inches="tight", transparent=True, dpi=1000)
    plt.savefig(filename + ".pdf", bbox_inches="tight", dpi=1000)
    plt.close()


if __name__ == "__main__":

    # Plot when using one lambda to penalize model L0 and when using various
    # lambdas (all of the same value) for penalizing each layer's normalized L0.
    for training_level in ["model", "layer"]:
        main(training_level)
