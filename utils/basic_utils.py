from typing import Dict, Type, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn

prng = np.random.default_rng(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def change_random_seed(seed):
    global prng

    # Seeds
    prng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## torch + cudnn behaviour. Otherwise convolutions are not reproducible
    # cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    cudnn.benchmark = True


def to_cuda(obj: Union[Dict, Type[torch.tensor]]):
    if isinstance(obj, dict):
        return {key: to_cuda(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(_) for _ in obj]
    else:
        return obj.cuda()


def prefix_keys(dict_, prefix):
    """Add a prefix to all keys in a dictionary. Used for WandB sections."""
    return {prefix + key: val for key, val in dict_.items()}


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def get_final_features(
    in_size: torch.Size, module: torch.nn.Module, verbose: bool = False
) -> torch.Size:
    """
    Computes the size of the output of a module given an input size.

    Args:
        in_size: Size of the input to the module.
        module: Module whose output size we are computing.
        verbose: Defaults to False.

    Returns:
        Size of the output of the module.
    """
    module.eval()  # prevent dummy_input from modifying BN stats
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = module(dummy_input)
    if verbose:
        print("module_out_size: {}".format(f.size()))

    return f.size()[1:]
