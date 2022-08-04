import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(BASE_PATH))

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

for dataset, model_type in [
    # ("mnist", "MLP"),
    # ("mnist", "LeNet"),
    ("cifar10", "ResNet-28-10"),
    ("cifar100", "ResNet-28-10"),
    ("tiny_imagenet", "L0ResNet18"),
]:

    print(dataset, model_type)

    plot_list = []

    fig = plt.figure(figsize=(10, 6))
    ax = [fig.add_subplot(2, 3, i + 1) for i in range(6)]

    row_names = ["layerwise", "modelwise"]
    col_names = ["control", "error_vs_params", "error_vs_macs"]
    ax_id = 0
    for row_id, task_type in enumerate(row_names):
        for col_id, plot_type in enumerate(col_names):
            fig_name = f"figs/{plot_type}/{dataset}/{model_type}_{task_type}.png"
            ax[ax_id].axis("off")
            ax[ax_id].imshow(plt.imread(fig_name))
            ax_id += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
