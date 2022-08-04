import logging

import torch


class ConstraintScheduler:
    def __init__(
        self,
        target_density: list = None,
        lmbdas: list = None,
        const_update: str = "fix",
        shrink_factor: float = 0.6,
        nbins_linear: int = None,
    ):

        # Exactly one of these two variables must be None
        assert (lmbdas is None) ^ (target_density is None)
        if lmbdas is not None:
            const_update = ""

        self.lmbdas = lmbdas

        self.current_level = None
        self.shrink_factor = shrink_factor
        self.const_update = const_update
        self.nbins_linear = nbins_linear

        if target_density is not None:
            tdst = (
                target_density if isinstance(target_density, list) else [target_density]
            )
            self.target_density = torch.tensor(tdst)
            self.current_level = torch.ones_like(self.target_density)

            if self.const_update == "linear_schedule":
                assert self.nbins_linear is not None
                self.linear_update_size = (
                    self.current_level - self.target_density
                ) / self.nbins_linear

            if torch.cuda.is_available():
                self.target_density = self.target_density.to("cuda")
                self.current_level = self.current_level.to("cuda")
                if self.const_update == "linear_schedule":
                    self.linear_update_size = self.linear_update_size.to("cuda")

            self.update_current_level(None)
            logging.info(
                "ConstraintScheduler initialized at {}".format(self.current_level)
            )

    def update_current_level(self, multipliers):
        if self.const_update == "fix":
            self.current_level = self.target_density
        elif self.const_update == "linear_schedule":
            self.current_level = torch.max(
                self.target_density, self.current_level - self.linear_update_size
            )
        elif self.const_update == "schedule" or multipliers is None:
            # Apply this initialization also when multipliers have not been initialized
            self.current_level = (
                self.current_level * (1 - self.shrink_factor)
                + self.target_density * self.shrink_factor
            )
        elif self.const_update == "dynamic":
            for cl, td, mul in zip(
                self.current_level, self.target_density, multipliers
            ):
                if mul == 0:
                    # Update current level in place
                    cl *= 1 - self.shrink_factor
                    cl += td * self.shrink_factor
