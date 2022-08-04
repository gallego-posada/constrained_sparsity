import csv
import os
import sys
from argparse import Namespace

BASE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(os.path.join(BASE_PATH, os.path.pardir)))

import torch

import utils


def get_model_stats():

    # Creating a Namespace to simulate execution of core_exp
    args = Namespace(
        weight_decay=0.0,
        l2_detach_gates=True,
        temp=2.0 / 3.0,
        act_fn="ReLU",
        bn_type="L0",
        task_type="gated",
    )

    model_types = ["MLP", "LeNet", "L0ResNet18", "L0ResNet50", "ResNet-28-10"]
    stats_dicts = []

    for model_type in model_types:
        if model_type in ["MLP", "LeNet"]:
            input_shape = (1, 28, 28)
            num_classes = 10
            args.use_bias = True
        elif model_type == "L0ResNet18":
            input_shape = (3, 64, 64)
            num_classes = 200
            args.use_bias = False
            # l0_conv_ix = ["conv1", "conv2"] # This is the hard-coded default
        elif model_type == "L0ResNet50":
            input_shape = (3, 224, 224)
            num_classes = 1000
            args.use_bias = False
            # l0_conv_ix = ["conv1", "conv2"] # This is the hard-coded default
        elif model_type == "ResNet-28-10":
            input_shape = (3, 32, 32)
            num_classes = 100
            args.use_bias = False
        else:
            raise ValueError("Model type not understood")

        args.model_type = model_type

        model = utils.exp_utils.construct_model(args, num_classes, input_shape)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        # Turn on all the gates in the model to make it fully dense
        for layer in model.layers_dict["l0"]:
            layer.weight_log_alpha.data = 10.0 * torch.ones_like(
                layer.weight_log_alpha.data
            )

        # Purge the model
        purged_model = utils.exp_utils.purge_model(model)
        if torch.cuda.is_available():
            purged_model = purged_model.cuda()
        purged_model.eval()

        model_stats = utils.exp_utils.get_macs_and_params(purged_model)
        model_stats.update({"model_type": model_type})
        stats_dicts.append(model_stats)

    return stats_dicts


if __name__ == "__main__":
    stats_dicts = get_model_stats()

    cvs_path = os.path.join(BASE_PATH, "saved_dataframes/all_models_stats.csv")

    with open(cvs_path, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["model_type", "macs", "params"])
        writer.writeheader()
        for data in stats_dicts:
            writer.writerow(data)
