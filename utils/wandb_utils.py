import os
import random
import string
from typing import List, Optional

import numpy as np
import torch
from cooper import LagrangianFormulation

import wandb
from sparse import BaseL0Layer


def create_wandb_subdir(subdir_name: str):
    """
    Create a subdirectory in wandb.
    """
    try:
        os.mkdir(os.path.join(wandb.run.dir, subdir_name))
    except:
        pass


def prepare_wandb(run_id, args):
    """
    Disable WandB logging or make logging offline.
    """

    if not args.use_wandb:
        wandb.setup(
            wandb.Settings(
                mode="disabled",
                program=__name__,
                program_relpath=__name__,
                disable_code=True,
            )
        )
    else:
        if args.use_wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=args.dataset_name,
        entity="constrained_l0",
        dir=args.wandb_dir,
        resume="allow",
        id=run_id,
    )
    wandb.config.update(args, allow_val_change=True)
    print("WANDB_RUN_ID", wandb.run.id, run_id)


def collect_l0_stats(reg_stats):
    model_reg = reg_stats.l0_model
    layer_regs = reg_stats.l0_layer
    reg_dict = {
        "reg/l0_layer_" + str(i).zfill(2): reg for i, reg in enumerate(layer_regs)
    }
    reg_dict["reg/l0_model"] = model_reg.item()
    return reg_dict


def collect_multipliers(formulation):

    lmbda_dict = {}
    for idx, mult_vals in enumerate(formulation.state()):
        # The first element of formulation.state() corresponds to
        # ineq_multipliers, the second to eq_multipliers
        mult_type = "ineq" if idx == 0 else "eq"
        prefix = "reg/" + mult_type + "_lambda_"

        if mult_vals is not None:
            if len(mult_vals.shape) == 0:
                # if scalar, convert to iterable for comprehension
                mult_vals = mult_vals.unsqueeze(0)

            for mult_ix, lmbda in enumerate(mult_vals):
                lmbda_dict[prefix + str(mult_ix).zfill(2)] = lmbda

    return lmbda_dict


def collect_gates_hist(gated_layers):
    wandb_dict = {}
    for layer_ix, layer in enumerate(gated_layers):
        for is_training in [True, False]:
            # This logs "soft" or "hard thresholded" gates depending on is_training
            wandb_dict.update(log_gates_histogram(layer, layer_ix, is_training))

    return wandb_dict


def log_gates_histogram(layer, idx, do_sample):
    """
    Log a histogram of the gates of layer to wandb summary.
    is_training=True for training; sampling once for each of the gate's distributions.
    is_training=False for validation; select the median of each gate's distribution.
    """

    if do_sample:
        weight_gates, bias_gates = layer.sample_gates()
    else:
        weight_gates, bias_gates = layer.evaluation_gates()

    fix = "samples" if do_sample else "medians"
    log_name = "gates_" + fix + "/layer_" + str(idx + 1).zfill(2)
    weight_gates = weight_gates.detach().cpu().numpy()
    weight_hist = np.histogram(weight_gates, bins=50)

    hist_dict = {"weight_" + log_name: wandb.Histogram(np_histogram=weight_hist)}
    if bias_gates is not None:
        bias_hist = np.histogram(bias_gates.detach().cpu().numpy(), bins=50)
        hist_dict["bias_" + log_name] = bias_hist

    return hist_dict


def get_baseline_model(entity, project, filters):
    """
    Get the best compressed models of runs according to filters.
    """

    api = wandb.Api(overrides={"entity": entity, "project": project})
    runs = api.runs(path=entity + "/" + project, filters=filters, order="-created_at")
    assert len(runs) <= 1, "More than one 'baseline' run found, revise TAGS"

    for one_run in runs:

        # Get best compressed model in terms of psnr
        wandb_file_name = "final_model_state.pt"
        root = "./models"
        one_run.file(wandb_file_name).download(root=root, replace=True)
        model_state = torch.load(os.path.join(root, wandb_file_name))

        return model_state, one_run.config, one_run.summary
