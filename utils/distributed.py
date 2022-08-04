import os

import torch


def init_distributed_mode():

    dist_dict = {
        "dist_url": "env://",
        "rank": None,
        "world_size": None,
        "gpu": None,
        "distributed": None,
        "dist_backend": None,
    }

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist_dict["rank"] = int(os.environ["RANK"])
        dist_dict["world_size"] = int(os.environ["WORLD_SIZE"])
        dist_dict["gpu"] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        dist_dict["world_size"] = 1
        dist_dict["rank"] = int(os.environ["SLURM_PROCID"])
        dist_dict["gpu"] = dist_dict["rank"] % torch.cuda.device_count()
    # elif hasattr(args, "rank"):
    #     pass
    else:
        print("Not using distributed mode")
        dist_dict["distributed"] = False
        return dist_dict

    dist_dict["distributed"] = True

    torch.cuda.set_device(dist_dict["gpu"])
    dist_dict["dist_backend"] = "nccl"
    rank, dist_url = dist_dict["rank"], dist_dict["dist_url"]
    print(f"| distributed init (rank {rank}): {dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=dist_dict["dist_backend"],
        init_method=dist_dict["dist_url"],
        world_size=dist_dict["world_size"],
        rank=dist_dict["rank"],
    )
    torch.distributed.barrier()
    setup_for_distributed(dist_dict["rank"] == 0)

    return dist_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def distributed_wrapper(model, dist_dict):
    """Wraps model in DistributedDataParallel if distributed is True. Also gathers
    weights and gates parameters."""

    if dist_dict["distributed"]:
        assert (
            torch.cuda.is_available()
        ), "Distributed mode requested but no GPUs available"
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_dict["gpu"]]
        )
        model_module = model.module
    elif torch.cuda.is_available():
        model = model.cuda()
        model_module = model

    return model, model_module


def is_main_process():
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()

    if not is_dist:
        rank = 0
    else:
        rank = torch.distributed.get_rank()

    return rank == 0
