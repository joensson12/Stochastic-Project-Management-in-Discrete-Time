"""Duration distributions for the stochastic duration variables D_i."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Protocol

import numpy as np
from numpy.typing import NDArray


UINT32_MAX = np.iinfo(np.uint32).max  # Largest storable sampled duration value.


class DurationDistribution(Protocol):
    """Interface for the law of a nonnegative integer duration D_i.

    A work package only needs to know how to draw samples from its duration
    distribution. This is the point where later models can replace shifted
    Poisson without changing the project-network algorithm.
    """

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        """Generate Monte Carlo samples of D_i as nonnegative integers."""


@dataclass(frozen=True)
class ShiftedPoissonDuration:
    """Shifted Poisson model D_i = a_i + X_i, X_i ~ Poisson(lambda_i)."""

    lambda_: float  # Poisson intensity lambda_i.
    a: int  # Deterministic minimum duration a_i in N_0.

    def __post_init__(self) -> None:
        # The report assumes lambda_i > 0 and a_i in N_0.
        if self.lambda_ <= 0:  # The Poisson parameter must satisfy lambda_i > 0.
            raise ValueError("lambda_ must be positive.")
        if self.a < 0:  # The shift a_i is a nonnegative integer.
            raise ValueError("a must be nonnegative.")
        if self.a > UINT32_MAX:  # Even the deterministic lower bound must fit in storage.
            raise ValueError("a must fit in uint32.")

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        if sample_count <= 0:  # A Monte Carlo vector must contain at least one scenario.
            raise ValueError("sample_count must be positive.")

        # NumPy samples the stochastic part X_i; the deterministic minimum
        # duration a_i shifts every realization by the same amount.
        samples = rng.poisson(self.lambda_, size=sample_count) + self.a  # D_i = X_i + a_i.
        if np.any(samples < 0):  # Defensive check for the nonnegative support of D_i.
            raise ValueError("duration samples must be nonnegative.")
        if np.any(samples > UINT32_MAX):  # The mathematical law is unbounded; storage is not.
            raise OverflowError("duration samples exceed uint32 capacity.")

        return samples.astype(np.uint32, copy=False)  # Store D_i compactly as nonnegative time data.

    @property
    def expected_value(self) -> float:
        """Return E[D_i] = a_i + lambda_i for this prototype law."""
        return self.a + self.lambda_  # Mean of a_i + Poi(lambda_i).

    @property
    def modes(self) -> tuple[int, ...]:
        """Return the possible modal duration values of D_i.

        This implements the report's formula
        a_i + floor(lambda_i) and a_i + ceil(lambda_i) - 1.
        For non-integer lambda_i these coincide, so only one value is returned.
        """
        possible_modes = {  # The set removes the duplicate in the non-integer case.
            self.a + floor(self.lambda_),  # First possible mode.
            self.a + ceil(self.lambda_) - 1,  # Second possible mode when lambda_i is integer.
        }
        return tuple(sorted(possible_modes))  # Sorted output is deterministic.

    @classmethod
    def from_independent_sum(
        cls,
        durations: Iterable["ShiftedPoissonDuration"],
    ) -> "ShiftedPoissonDuration":
        """Use closure: sum_i D_i is ShiftedPoi(sum_i lambda_i, sum_i a_i).

        This is intentionally specific to the shifted-Poisson prototype. It is
        not a generic distribution-composition interface.
        """
        duration_list = list(durations)  # Materialize once so lambda_i and a_i use the same collection.
        if not duration_list:
            raise ValueError("at least one duration is required.")
        return cls(
            lambda_=sum(duration.lambda_ for duration in duration_list),  # Sum of Poisson intensities.
            a=sum(duration.a for duration in duration_list),  # Sum of deterministic shifts.
        )
