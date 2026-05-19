"""Duration distributions for the stochastic duration variables D_i."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from numbers import Integral
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


class ProbabilityDistribution(Protocol):
    """Interface for a sampled occurrence-probability vector P_i^j."""

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Generate Monte Carlo samples as probabilities in [0, 1]."""


@dataclass(frozen=True)
class DeterministicProbability:
    """Deterministic occurrence-probability model P_i^j = p."""

    probability: float  # Fixed occurrence probability p.

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be between 0 and 1.")

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        return np.full(sample_count, self.probability, dtype=np.float64)  # P_i^j is fixed in every scenario.


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


@dataclass(frozen=True)
class ShiftedBinomialDuration:
    """Bounded shifted binomial model D_i = minimum + X_i.

    Here X_i ~ Binomial(maximum - minimum, p_i). The supplied most_likely
    duration determines the interval of p_i values for which that duration is
    a binomial mode, and this model uses the midpoint of that interval.
    """

    minimum: int  # Smallest possible duration.
    maximum: int  # Largest possible duration.
    most_likely: int  # Requested unique modal duration.

    def __post_init__(self) -> None:
        for name in ("minimum", "maximum", "most_likely"):
            value = getattr(self, name)
            if not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            object.__setattr__(self, name, int(value))

        if self.minimum < 0:
            raise ValueError("minimum must be nonnegative.")
        if self.maximum < self.minimum:
            raise ValueError("maximum must be at least minimum.")
        if self.maximum > UINT32_MAX:
            raise ValueError("maximum must fit in uint32.")
        if not self.minimum <= self.most_likely <= self.maximum:
            raise ValueError("most_likely must be between minimum and maximum.")

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")

        samples = rng.binomial(self.trial_count, self.p, size=sample_count) + self.minimum
        if np.any(samples > UINT32_MAX):
            raise OverflowError("duration samples exceed uint32 capacity.")
        return samples.astype(np.uint32, copy=False)

    @property
    def trial_count(self) -> int:
        """Return the binomial number of trials n = maximum - minimum."""
        return self.maximum - self.minimum

    @property
    def p(self) -> float:
        """Return the midpoint p that makes most_likely the modal duration."""
        n = self.trial_count
        if n == 0:
            return 0.0

        mode_offset = self.most_likely - self.minimum
        lower = mode_offset / (n + 1)
        upper = (mode_offset + 1) / (n + 1)
        return (lower + upper) / 2

    @property
    def expected_value(self) -> float:
        """Return E[D_i] = minimum + n p_i."""
        return self.minimum + self.trial_count * self.p

    @property
    def modes(self) -> tuple[int, ...]:
        """Return the modal duration values of D_i."""
        return (self.most_likely,)
