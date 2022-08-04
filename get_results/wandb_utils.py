"""
Extract information logged to wandb in order to plot/analyze.
WandB help: https://docs.wandb.ai/guides/track/public-api-guide
"""
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import pandas as pd
import torch

import utils
import wandb

# Fix entity
ENTITY = "constrained_l0"


def get_metrics(
    project, filters, metric_keys, config_keys=None, x_axis="_step", sample_size=10_000
):
    """
    Extract metric_keys from wandb runs given filters. Keep config_keys for reference
    Args:
        filters: Example {"$and": [{"config.run_group": "control"},
                            {"config.dual_optim": "SGD"}]}
        metric_keys: Example ["val/top1", "val/macs", "val/params"]
        config_keys: config elements to return: ["seed", "model_type"]
        x_axis: one of "_step" or "epoch"
    Returns:
        DataFrame with metrics, config list
    """
    api = wandb.Api(overrides={"entity": ENTITY, "project": project}, timeout=20)
    runs = api.runs(path=ENTITY + "/" + project, filters=filters, order="-created_at")
    print("Number of runs:", len(runs))

    all_frames = []
    for run in runs:
        # samples param: without replacement, if too large returns all.
        metrics = run.history(samples=sample_size, keys=metric_keys, x_axis=x_axis)

        # Do not keep the whole config, only config_keys if provided by user
        filtered_config = {
            key: run.config[key] for key in config_keys if key in run.config
        }

        for key, val in filtered_config.items():
            metrics.insert(0, key, str(val))

        all_frames.append(metrics)

    return pd.concat(all_frames)


def epoch_batch_runs(project, filters, keys_epoch, keys_batch, config_keys=None):
    # DataFrame with the historic of the metrics
    metrics_epoch = get_metrics(project, filters, keys_epoch, config_keys, "epoch")
    metrics_batch = get_metrics(project, filters, keys_batch, config_keys, "epoch")

    # Aggregate metrics_batch to an epoch level
    idx = [*config_keys, "epoch"]
    metrics_batch = metrics_batch.groupby(idx).agg(["mean"]).reset_index()
    metrics_batch.columns = metrics_batch.columns.droplevel(1)  # drop "mean" col name

    # Join both dataframes
    metrics_batch = metrics_batch.set_index(idx)
    metrics_epoch = metrics_epoch.set_index(idx)
    metrics = metrics_batch.join(metrics_epoch)
    metrics = metrics.reset_index().groupby(config_keys)  # not x_axis

    return metrics


def get_models(project, filters, purge=True):
    """
    Get the models of runs according to filters. If purge, the model is purged
    Args:
        filters = {"$and": [{"config.run_group": "4 oct"}, {"config.dual_optim": "SGD"},
                {"config.primal_optim": "Adam"}, {"config.dual_restart": True}]}
    Returns:
        list of models, list of configs
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    api = wandb.Api(overrides={"entity": ENTITY, "project": project})
    runs = api.runs(path=ENTITY + "/" + project, filters=filters, order="-created_at")
    print("Number of runs:", len(runs))

    model_list, config_list = [], []
    for i, run in enumerate(runs):
        print("Model", i, "of", len(runs))

        # Get last iterate model
        run.file("model.h5").download(root="wandb/models/", replace=True)
        model = torch.load("wandb/models/model.h5", map_location=device)

        if purge:
            model = utils.exp_utils.purge_model(model)

        config_list.append(run.config)
        model_list.append(model)

    return model_list, config_list
