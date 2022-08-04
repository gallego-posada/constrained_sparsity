import argparse
import logging
import os

import cooper
import torch
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler

import wandb
from utils import (
    basic_utils,
    constraints,
    datasets,
    distributed,
    exp_utils,
    lr_scheduling,
    wandb_utils,
)


def prepare_metadata(checkpoint_dir):

    checkpoint_path = os.path.join(checkpoint_dir, "metadata.pt")
    if os.path.exists(checkpoint_path):
        loaded_metadata = torch.load(checkpoint_path)

        # Make sure we are resuming the correct run
        assert loaded_metadata["run_id"] == wandb.run.id

        init_epoch = loaded_metadata["epoch"]
        step_id = loaded_metadata["step_id"]

        logging.info("Resuming from epoch {}".format(init_epoch))
    else:
        init_epoch, step_id = 0, 0

    return init_epoch, step_id


def prepare_constraint_scheduler(args):

    reg_config = constraints.ConstraintScheduler(
        args.target_density,
        args.lmbdas,
        const_update="fix",
    )

    return reg_config


def prepare_model(args, num_classes, input_shape, checkpoint_dir):

    model = exp_utils.construct_model(args, num_classes, input_shape)

    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(checkpoint_path):
        logging.info("Loading model from checkpoint: {}".format(checkpoint_path))
        model.load_state_dict(torch.load(checkpoint_path))

    num_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info("Number of model parameters: {}".format(num_params))

    return model


def prepare_cmp_formulation(args, checkpoint_dir, loss_module_str="CrossEntropyLoss"):

    if loss_module_str == "CrossEntropyLoss":
        loss_module = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif loss_module_str == "KLDivLoss":
        loss_module = nn.KLDivLoss(reduction="batchmean", log_target=True)
    else:
        loss_module = nn.__dict__[loss_module_str]()

    cmp = exp_utils.construct_problem(
        task_type=args.task_type,
        target_density=args.target_density,
        lmbdas=args.lmbdas,
        loss_module=loss_module,
        weight_decay=args.weight_decay,
    )

    formulation = cooper.LagrangianFormulation(cmp)

    checkpoint_path = os.path.join(checkpoint_dir, "formulation.pt")
    if os.path.exists(checkpoint_path):
        logging.info("Loading formulation from checkpoint: {}".format(checkpoint_path))
        formulation.load_state_dict(torch.load(checkpoint_path))

    return cmp, formulation


def prepare_constrained_optimizer(
    args, formulation, primal_optimizer, dual_optimizer, checkpoint_dir
):

    checkpoint_path = os.path.join(checkpoint_dir, "constrained_optimizer.pt")
    if os.path.exists(checkpoint_path):
        logging.info(
            "Loading constrained optimizer from checkpoint: {}".format(checkpoint_path)
        )
        const_optim_checkpoint = torch.load(checkpoint_path)

        return cooper.ConstrainedOptimizer.load_from_state_dict(
            const_optim_state=const_optim_checkpoint,
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer_class=dual_optimizer,
            dual_scheduler_class=None,  # Not using dual scheduler in this project
        )

    else:
        return cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
            dual_restarts=not (args.no_dual_restart),  # Note the double negation
        )


def prepare_lr_scheduler(args, constrained_optimizer, checkpoint_dir):

    if not args.use_lr_schedule:
        return None

    else:

        # This scheduler is a custom version of MultiStepLR
        primal_lr_scheduler = lr_scheduling.GroupMultiStepLR(
            constrained_optimizer.primal_optimizer,
            milestones=args.sch_epoch_drop,
            # Schedule learning rate of first parameter group (weights) but not
            # of second parameter group (gates).
            groups=[True, args.use_gates_schedule],
            gamma=args.lr_decay_ratio,
        )

        checkpoint_path = os.path.join(checkpoint_dir, "primal_lr_schedule.pt")
        if os.path.exists(checkpoint_path):
            logging.info(
                "Loading primal learning rate scheduler from checkpoint: {}".format(
                    checkpoint_path
                )
            )
            sch_checkpoint = torch.load(checkpoint_path)
            primal_lr_scheduler.load_state_dict(sch_checkpoint)

        return primal_lr_scheduler


def generate_checkpoint(
    args,
    epoch,
    step_id,
    model,
    constrained_optimizer,
    primal_lr_schedule,
    checkpoint_dir,
):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Only save on main process
    if not distributed.is_main_process():
        return

    # Save checkpoints for the epoch, model, formulation and constrained_optimizer
    meta_dict = {"run_id": wandb.run.id, "epoch": epoch, "step_id": step_id}
    torch.save(meta_dict, os.path.join(checkpoint_dir, "metadata.pt"))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

    torch.save(
        constrained_optimizer.formulation.state_dict(),
        os.path.join(checkpoint_dir, "formulation.pt"),
    )

    torch.save(
        constrained_optimizer.state_dict(),
        os.path.join(checkpoint_dir, "constrained_optimizer.pt"),
    )

    if args.use_lr_schedule:
        torch.save(
            primal_lr_schedule.state_dict(),
            os.path.join(checkpoint_dir, "primal_lr_schedule.pt"),
        )


def main(args):

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if "LOCAL_RANK" in os.environ:
        # This gets populated by torchrun
        local_rank = int(os.environ["LOCAL_RANK"])
        is_master = local_rank == 0
    else:
        is_master = True

    run_id = os.environ["WANDB_RUN_ID"] if "WANDB_RUN_ID" in os.environ.keys() else None

    dist_dict = distributed.init_distributed_mode()

    batch_size_ = args.batch_size
    if dist_dict["distributed"]:
        # In distributed mode, each GPU has its own dataloader with its own
        # batch_size. When consolidating across GPUs, the effective batch size
        # becomes Num_GPUs * batch_size. Therefore, dataloaders on each process
        # must be created with a fraction of args.batch_size.
        # Eg: If desired batch size is 256 and employing 4 GPUs, create
        # dataloaders with 64 batch_size
        batch_size_ = int(args.batch_size / dist_dict["world_size"])

    # Initialize WandB (log directory, import run_id from env variables, etc.)
    if is_master:
        wandb_utils.prepare_wandb(run_id, args)
        wandb_utils.create_wandb_subdir("models")
        run_checkpoint_dir = os.path.join(args.checkpoint_dir, wandb.run.id)
        if run_id is not None:
            assert run_id == wandb.run.id
    else:
        run_checkpoint_dir = "NON_EXISTENT_PATH"

    # init_epoch and step_id are loaded from a checkpoint. If there is no
    # checkpoint, both are set to 0.
    init_epoch, step_id = prepare_metadata(run_checkpoint_dir)

    basic_utils.change_random_seed(args.seed)

    foo = datasets.load_dataset(
        args.dataset_name,
        args.augment,
        train_batch_size=batch_size_,
        val_batch_size=256,
        distributed=dist_dict["distributed"],
    )
    train_loader, val_loader, num_classes, input_shape = foo

    if args.task_type == "gated":
        # TODO: Currently only using "fixed" schedule (for simpler checkpoints)
        reg_config = prepare_constraint_scheduler(args)

    model = prepare_model(args, num_classes, input_shape, run_checkpoint_dir)
    model, model_module = distributed.distributed_wrapper(model, dist_dict)

    primal_optim_class = exp_utils.OPTIM_DICT[args.primal_optim]
    if args.task_type == "gated":
        net_params = model_module.params_dict["net"]
        gate_params = model_module.params_dict["gates"]

        # If not explicitly provided gates_lr, we use primal_lr (as for the weights)
        gates_lr = args.primal_lr if args.gates_lr is None else args.gates_lr

        # TODO: Pytorch training adds '"weight_decay": args.primal_wd'. Why?
        primal_optimizer = primal_optim_class(
            [
                {"params": net_params, "lr": args.primal_lr},
                {"params": gate_params, "lr": gates_lr},
            ]
        )
    else:
        primal_optimizer = primal_optim_class(model.parameters(), lr=args.primal_lr)

    cmp, formulation = prepare_cmp_formulation(args, run_checkpoint_dir)

    if cmp.is_constrained:
        dual_optimizer = cooper.optim.partial_optimizer(
            exp_utils.OPTIM_DICT[args.dual_optim], lr=args.dual_lr
        )
    else:
        dual_optimizer = None

    constrained_optimizer = prepare_constrained_optimizer(
        args, formulation, primal_optimizer, dual_optimizer, run_checkpoint_dir
    )

    primal_lr_schedule = prepare_lr_scheduler(
        args, constrained_optimizer, run_checkpoint_dir
    )

    if is_master:
        # Print model performance at initialization
        val_log = exp_utils.validation_loop(
            epoch=-1,
            val_loader=val_loader,
            model=model,
            model_module=model_module,
            cmp=cmp,
            val_prop=args.val_prop,
            do_purge_model=True,
        )
        logging.info("Initial validation performance: %s", val_log["val/top1"])

    # Training loop
    # step_id and init_epoch come form prepare_metadata
    for epoch in range(init_epoch, args.epochs):

        if is_master:
            logging.info("Epoch %s - StepId %s", epoch, step_id)

        if dist_dict["distributed"]:
            train_loader.sampler.set_epoch(epoch)

        step_id, wandb_step_dict, epoch_log_dict = exp_utils.train(
            cmp,
            model,
            model_module,
            formulation,
            train_loader,
            constrained_optimizer,
            step_id,
            epoch,
            reg_config if args.task_type == "gated" else None,
            args.use_wandb,
            args.debug_batches,
        )

        if is_master:
            # Validation statistics only get logged by master process
            val_log = {}
            do_validate = (args.val_freq > 0) and ((epoch + 1) % args.val_freq == 0)
            if do_validate or (epoch + 1 == args.epochs):
                logging.info("Validation loop on epoch: %s", epoch)

                # This generates the logs for panel "val/"
                val_log = exp_utils.validation_loop(
                    epoch=epoch,
                    val_loader=val_loader,
                    model=model,
                    model_module=model_module,
                    cmp=cmp,
                    val_prop=args.val_prop,
                    do_purge_model=True,
                )

            # Log each of the steps to wandb
            for logged_step_id in wandb_step_dict:
                wandb.log(
                    wandb_step_dict[logged_step_id], step=logged_step_id, commit=False
                )

            # Log validation stats
            epoch_log_dict.update(val_log)

            # commit=True here also transmits batch-level logs to wandb
            wandb.log(epoch_log_dict, step=step_id, commit=True)

        # Update current learning rate
        if args.use_lr_schedule:
            primal_lr_schedule.step()

        # Generate checkpoint
        do_checkpoint = (args.checkpoint_every_n_epochs > 0) and (
            (epoch + 1) % args.checkpoint_every_n_epochs == 0
        )
        if do_checkpoint and is_master:
            generate_checkpoint(
                args,
                epoch + 1,  # We are actually done with this epoch, should add 1
                step_id,
                model_module,
                constrained_optimizer,
                primal_lr_schedule,
                run_checkpoint_dir,
            )

    # Save model and checkpoints at end of training
    if is_master and args.save_final_model:

        generate_checkpoint(
            args,
            epoch + 1,
            step_id,
            model_module,
            constrained_optimizer,
            primal_lr_schedule,
            run_checkpoint_dir,
        )

        filename = os.path.join(run_checkpoint_dir, "final_model_state.pt")
        torch.save(model_module.state_dict(), filename)
        wandb.save(filename, base_path=run_checkpoint_dir, policy="end")

    return model, cmp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Constrained L0 Training")

    parser.add_argument("--name", default="", type=str, help="name of experiment")
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument(
        "--epochs", default=200, type=int, help="number of total epochs to run"
    )

    # Dataset
    parser.add_argument(
        "-dn",
        "--dataset_name",
        default="mnist",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "tiny_imagenet", "imagenet"],
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=128,
        type=int,
        help="mini-batch size (default: 128)",
    )
    parser.add_argument(
        "-db",
        "--debug_batches",
        default=None,
        type=int,
        help="number of batches to use in training for debugging",
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="whether to use standard data augmentation (default: True)",
    )

    parser.add_argument("-nub", "--use_bias", action="store_false")

    # Used for training on ImageNet. Default set to 0 to not interfere with other datasets.
    parser.add_argument(
        "--label_smoothing",
        default=0.0,
        type=float,
        help="label smoothing (default: 0.0)",
    )

    parser.add_argument(
        "-wd",
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay (default: 0.0)",
    )
    parser.add_argument(
        "-no_detach_gates", dest="l2_detach_gates", action="store_false"
    )
    parser.add_argument("--temp", type=float, default=2.0 / 3.0)
    parser.add_argument("--droprate_init", type=float, default=0.5)
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        default="MLP",
        choices=["MLP", "LR", "LeNet", "ResNet-28-10", "ResNet-16-8", "ResNet50"],
    )
    parser.add_argument(
        "--act_fn", type=str, default="ReLU", choices=["ReLU", "Tanh", "LeakyReLU"]
    )
    parser.add_argument(
        "--bn_type", type=str, default="identity", choices=["identity", "regular", "L0"]
    )

    # Primal optimizer
    parser.add_argument("--primal_optim", default="Adam", type=str)
    parser.add_argument(
        "-lr", "--primal_lr", default=1e-3, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--gates_lr", default=None, type=float, help="initial learning rate 4 gates"
    )

    # Learning rate scheduler
    parser.add_argument("--use_lr_schedule", action="store_true")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.2)
    parser.add_argument("--sch_epoch_drop", nargs="*", type=int, default=(60, 120, 160))
    parser.add_argument("--use_gates_schedule", action="store_true")

    # Dual optimizer
    parser.add_argument("--dual_optim", default="SGD", type=str)
    parser.add_argument("-ndr", "--no_dual_restart", action="store_true")
    parser.add_argument(
        "--dual_lr", default=7e-4, type=float, help="dual learning rate"
    )

    # Task
    parser.add_argument(
        "--task_type",
        default="gated",
        type=str,
        choices=["gated", "baseline", "magnitude_pruning"],
    )
    parser.add_argument(
        "--sparsity_type",
        default="structured",
        type=str,
        choices=["structured", "unstructured"],
    )
    parser.add_argument(
        "--pretrain_type", default=None, type=str, choices=["torch", "ours"]
    )

    # Constraints and penalties
    parser.add_argument(
        "-tdst", "--target_density", nargs="*", type=float, default=None
    )
    parser.add_argument("--lmbdas", nargs="*", type=float, default=None)

    # Behaviour of the network at validation time
    parser.add_argument("--log_unpurged", action="store_true")
    # If -1, validate only the last iterate
    parser.add_argument("--val_freq", type=int, default=1)
    # Overridden when val_freq = -1
    parser.add_argument("--val_prop", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=-1)
    parser.add_argument("-no_save_model", dest="save_final_model", action="store_false")

    # W&B
    # By default run WandB (online)
    parser.add_argument("-wboff", dest="use_wandb", action="store_false")
    parser.add_argument("-wb_offline", dest="use_wandb_offline", action="store_true")
    parser.add_argument("--wandb_dir", default=None, type=str)

    parser.add_argument("--run_group", default="", type=str)

    parser.add_argument("-yaml", "--yaml_file", default="", type=str)

    return parser.parse_args()


if __name__ == "__main__":

    try_args = parse_arguments()
    try_args.verbose = True

    if try_args.yaml_file != "":
        opt = yaml.load(open(try_args.yaml_file), Loader=yaml.FullLoader)
        try_args.__dict__.update(opt)

    # Configure WandB settings
    if not try_args.use_wandb:
        wandb.setup(
            wandb.Settings(
                mode="disabled",
                program=__name__,
                program_relpath=__name__,
                disable_code=True,
            )
        )
    else:
        if try_args.use_wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

    main(try_args)
