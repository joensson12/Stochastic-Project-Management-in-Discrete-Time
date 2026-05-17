"""Work packages, the atomic activities in the project network."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spm.distributions import DurationDistribution


@dataclass
class WorkPackage:
    """Activity i in the work-package set W, with sampled duration D_i.

    The human-facing name of the work package is deliberately absent in this
    prototype. The mathematical layer works only with positive integer IDs,
    matching the set W = {1, ..., N} in the report.
    """

    work_package_id: int  # Activity index i in W.
    duration_model: DurationDistribution  # Probability law of D_i.
    duration_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of D_i.

    def __post_init__(self) -> None:
        if not isinstance(self.work_package_id, int):  # W is indexed by integers.
            raise TypeError("work_package_id must be an integer.")
        if self.work_package_id <= 0:  # The report uses W = {1, ..., N}.
            raise ValueError("work_package_id must be positive.")

    def simulate_duration(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        """Generate and store the Monte Carlo vector for D_i."""
        samples = self.duration_model.sample(sample_count, rng)  # Draw the Monte Carlo vector D_i.
        if samples.dtype != np.uint32:  # Normalize custom distributions to the storage type.
            samples = samples.astype(np.uint32, copy=False)  # Keep storage compact.
        self.duration_samples = samples  # Store D_i on the activity itself.
        return samples  # Return D_i for immediate use if needed.
