"""This module implements a simple modification of Pytorch's MultiStepLR scheduler.
Here, the learning rate is only decayed for the specified parameter groups. Other
groups are kept at their initial learning rate.

Motivation: decaying the gates learning rate is dangerous as it can lead to gates
freezing. Therefore, a constraint may never be satisfied if the decay happens too
soon.

Moreover, constructing a separete optimizer for the weights and gates would currently
not be compatible with Cooper.
"""
from bisect import bisect_right
from collections import Counter
import warnings
from torch.optim import lr_scheduler


class GroupMultiStepLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of *specified* parameter group by gamma once the
    number of epoch reaches one of the milestones. Other groups are kept at their
    initial learning rate.

    Notice that such decay can happen simultaneously with other changes to the
    learning rate from outside this scheduler. When last_epoch=-1, sets initial
    lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        groups (list): Binary mask of parameter groups to consider.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        milestones,
        groups,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.groups = groups
        super(GroupMultiStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch not in self.milestones:
            # No milestone was reached yet. Simply return the current LR.
            return [group["lr"] for group in self.optimizer.param_groups]

        new_lrs = []
        for i, group in enumerate(self.optimizer.param_groups):
            if self.groups[i]:
                # Decay the learning rate for this group
                new_lrs.append(
                    group["lr"] * self.gamma ** self.milestones[self.last_epoch]
                )
            if not self.groups[i]:
                # Keep the learning rate for this group unchanged
                new_lrs.append(group["lr"])

        return new_lrs

    def _get_closed_form_lr(self):
        milestones = list(sorted(self.milestones.elements()))

        new_lrs = []
        for i, (group, base_lr) in enumerate(
            zip(self.optimizer.param_groups, self.base_lrs)
        ):
            if self.groups[i]:
                # Decay the learning rate for this group
                new_lrs.append(
                    base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                )
            if not self.groups[i]:
                # Keep the learning rate for this group unchanged
                new_lrs.append(base_lr)

        return new_lrs
