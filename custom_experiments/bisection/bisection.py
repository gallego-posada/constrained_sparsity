import sys

sys.path.append(".")

import logging
import os

import yaml
from scipy import optimize

import core_exp
import wandb

logging.basicConfig(level=logging.ERROR)


def bisection_search(args, model_level=True, tdst=0.5):

    visited = []

    def eval_fn(log_lmbda):
        if model_level:
            args.lmbdas = [10**log_lmbda]
        else:
            args.lmbdas = [10**log_lmbda, 10**log_lmbda, 10**log_lmbda]

        model, cmp = core_exp.main(args)

        l0_model = cmp.state.misc["reg_stats"].l0_model.item()
        visited.append([log_lmbda, l0_model])

        print("Log_lmbda:", log_lmbda, "-- L0-model:", l0_model)

        return l0_model - tdst

    x0, r = optimize.bisect(eval_fn, -3, 0, disp=True, maxiter=6, full_output=True)
    print(r)

    print("***** Visited sequence *****")
    print(visited)


if __name__ == "__main__":

    try_args = core_exp.parse_arguments()
    try_args.verbose = True

    # Import default parameters from YAML
    try_args.yaml_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bisection_defaults.yml"
    )

    if try_args.yaml_file != "":
        opt = yaml.load(open(try_args.yaml_file), Loader=yaml.FullLoader)
        try_args.__dict__.update(opt)

    # Manually disable WandB
    wandb.setup(
        wandb.Settings(
            mode="disabled",
            program=__name__,
            program_relpath=__name__,
            disable_code=True,
        )
    )

    print("---------- LAYER-WISE ------------")
    bisection_search(try_args, model_level=False)
    print("------------------------")

    print("---------- MODEL-WISE ------------")
    bisection_search(try_args, model_level=True)
