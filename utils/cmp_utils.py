import dataclasses

import cooper
import torch
from torch import nn

from .constraints import ConstraintScheduler


@dataclasses.dataclass
class ModelRegStats:
    """
    A class to store the regularization statistics for a given model.
    """

    l0_layer: torch.Tensor
    l0_model: torch.Tensor
    l0_full: torch.Tensor
    l2_model: torch.Tensor


def construct_loss(loss_module: torch.nn.Module, weight_decay: float = None):
    if torch.cuda.is_available():
        loss_module = loss_module.cuda()

    def loss_func(model, pred, target_var, distributed=False):
        loss = loss_module(pred, target_var)

        model_module = model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model_module = model.module

        if hasattr(model_module, "regularization") and callable(
            model_module.regularization
        ):
            # Effectively selecting BaseL0Models

            reg = model_module.regularization()

            # Add contribution of L2 penalty to objective.
            # Multiplication by weight_decay happens inside model.regularization()
            # If weight_decay is 0, this will be a no-op
            loss += reg.l2_model

            return loss, reg
        else:
            l2_contrib = 0
            if weight_decay != 0:
                for m in model_module.modules():
                    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                        # Not applying weight decay to batchnorm layers
                        l2_contrib += torch.sum(m.weight**2)
                        if hasattr(m, "bias") and m.bias is not None:
                            l2_contrib += torch.sum(m.bias**2)
            return loss + weight_decay * l2_contrib

    return loss_func


class BaselineProblem(cooper.ConstrainedMinimizationProblem):
    """Baseline minimization problem for training a model"""

    def __init__(self, loss_module: torch.nn.Module, weight_decay: float = None):
        self.loss_func = construct_loss(loss_module, weight_decay)
        super().__init__(is_constrained=False)

    def closure(self, inputs, targets, model, reg_config):
        assert reg_config is None

        outputs = model.forward(inputs)
        loss = self.loss_func(model, outputs, targets)
        return cooper.CMPState(loss=loss, misc={"outputs": outputs})


class ConstrainedL0Problem(cooper.ConstrainedMinimizationProblem):
    """
    Class for L0-constrained minimization problems.
    """

    def __init__(self, loss_module: nn.Module, out_transform: callable = None):

        self.out_transform = out_transform
        self.loss_func = construct_loss(loss_module)
        super(ConstrainedL0Problem, self).__init__(is_constrained=True)

    def compute_constraints(
        self,
        reg_config: ConstraintScheduler,
        reg_stats: ModelRegStats,
    ):

        # Make sure that we actually have constraint levels
        target_density = reg_config.current_level
        assert target_density is not None, "No constraint levels were provided."

        if len(target_density) == len(reg_stats.l0_layer):
            # Layer-wise constraints
            ineq_defects = reg_stats.l0_layer - target_density
        elif len(target_density) == 1:
            # Model-wise constraints
            ineq_defects = reg_stats.l0_model - target_density
        else:
            raise ValueError(
                "target_density must be a list of length 1 or len(model.l0_layers)"
            )

        return ineq_defects

    def closure(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        reg_config: ConstraintScheduler,
    ) -> cooper.CMPState:
        """
        Computes CMP state for the given model and inputs.
        """

        outputs = model.forward(inputs)
        if self.out_transform is not None:
            outputs = self.out_transform(outputs)

        # Compute loss and regularization statistics
        loss, reg_stats = self.loss_func(model, outputs, targets)
        misc = {"outputs": outputs.data, "reg_stats": reg_stats}

        if reg_stats is not None:
            ineq_defects = self.compute_constraints(reg_config, reg_stats)
        else:
            ineq_defects = None

        # Store model output and other 'miscellaneous' objects in misc dict
        state = cooper.CMPState(loss=loss, ineq_defect=ineq_defects, misc=misc)

        return state


class PenalizedL0Problem(cooper.ConstrainedMinimizationProblem):
    """
    Class for L0-penalized minimization problems.
    """

    def __init__(self, loss_module: nn.Module, out_transform: callable = None):
        self.out_transform = out_transform
        self.loss_func = construct_loss(loss_module)
        super(ConstrainedL0Problem, self).__init__(is_constrained=False)

    def compute_penalty(
        self, reg_config: ConstraintScheduler, reg_stats: ModelRegStats
    ):
        lmbdas = reg_config.lmbdas
        assert lmbdas is not None, "No regularization hyper-params were provided."

        if len(lmbdas) == len(reg_stats.l0_layer):
            # Layer-wise penalties
            defects = reg_stats.l0_layer
            _lmbdas_t = torch.tensor(lmbdas, device=defects.device)
            violation = torch.dot(_lmbdas_t, defects)
        elif len(lmbdas) == 1:
            # Model-wise penalty
            violation = lmbdas[0] * reg_stats.l0_model
        else:
            raise ValueError(
                "lmbdas must be a list of length 1 or len(model.l0_layers)"
            )

        return violation

    def closure(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        reg_config: ConstraintScheduler,
    ) -> cooper.CMPState:
        """
        Computes CMP state for the given model and inputs.
        """

        outputs = model.forward(inputs)
        if self.out_transform is not None:
            outputs = self.out_transform(outputs)

        # Compute loss and regularization statistics
        loss, reg_stats = self.loss_func(model, outputs, targets)
        misc = {"outputs": outputs.data, "reg_stats": reg_stats}

        if reg_stats is not None:
            # We need to manually add the L0 penalty to the loss. There are no
            # constraints in this case, so Cooper can't perform the addition for us.
            loss += self.compute_penalty(reg_config, reg_stats)

        # Store model output and other 'miscellaneous' objects in misc dict
        state = cooper.CMPState(loss=loss, misc=misc)

        return state
