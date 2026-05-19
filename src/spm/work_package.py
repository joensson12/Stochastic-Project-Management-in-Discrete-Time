"""Work packages, the atomic activities in the project network."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spm.distributions import DurationDistribution, ProbabilityDistribution, UINT32_MAX


@dataclass(frozen=True)
class ScheduleRiskSimulationResult:
    """Monte Carlo output for one schedule-risk effect R_i^{t,j}."""

    probabilities: NDArray[np.float64]  # Sampled occurrence probabilities P_i^j.
    severities: NDArray[np.uint32]  # Sampled discrete severities U_i^j.
    occurrences: NDArray[np.bool_]  # Sampled Bernoulli indicators xi_i^j.
    effects: NDArray[np.uint32]  # Realized risk effects R_i^{t,j} = xi_i^j U_i^j.


@dataclass(frozen=True)
class ScheduleRisk:
    """Time-risk model attached to a work package."""

    probability_model: ProbabilityDistribution  # Law of P_i^j.
    severity_model: DurationDistribution  # Law of the discrete severity U_i^j.

    def simulate(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> ScheduleRiskSimulationResult:
        """Simulate one discrete schedule-risk effect vector R_i^{t,j}."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")

        probabilities = np.asarray(  # Draw P_i^j as one probability per scenario.
            self.probability_model.sample(sample_count, rng),
            dtype=np.float64,
        )
        if probabilities.shape != (sample_count,):
            raise ValueError("probability samples must be a vector of length sample_count.")
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probability samples must be between 0 and 1.")

        severities = np.asarray(self.severity_model.sample(sample_count, rng))  # Draw U_i^j as discrete time impact.
        if severities.shape != (sample_count,):
            raise ValueError("severity samples must be a vector of length sample_count.")
        if np.any(severities < 0) or np.any(severities > UINT32_MAX):
            raise ValueError("severity samples must be nonnegative and fit in uint32.")
        if severities.dtype != np.uint32:
            severities = severities.astype(np.uint32, copy=False)  # Store U_i^j as nonnegative integer time.

        occurrences = rng.random(sample_count) < probabilities  # xi_i^j | P_i^j ~ Bernoulli(P_i^j).
        effects = (occurrences.astype(np.uint32) * severities).astype(np.uint32, copy=False)  # R_i^j = xi_i^j U_i^j.
        return ScheduleRiskSimulationResult(  # Keep every vector needed for diagnostics and later cost extension.
            probabilities=probabilities,  # P_i^j samples.
            severities=severities,  # U_i^j samples.
            occurrences=occurrences,  # xi_i^j samples.
            effects=effects,  # R_i^{t,j} samples.
        )


@dataclass
class WorkPackage:
    """Activity i in the work-package set W, with sampled duration D_i.

    The human-facing name of the work package is deliberately absent in this
    prototype. The mathematical layer works only with positive integer IDs,
    matching the set W = {1, ..., N} in the report.
    """

    work_package_id: int  # Activity index i in W.
    duration_model: DurationDistribution  # Probability law of D_i.
    schedule_risks: list[ScheduleRisk] = field(default_factory=list)  # Time risks assigned to activity i.
    baseline_duration_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of Z_i.
    duration_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of D_i.
    start_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of S_i.
    finish_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of T_i.
    lag_samples_by_predecessor: dict[int, NDArray[np.uint32]] = field(default_factory=dict, init=False)  # Incoming L_{j,i}.
    schedule_risk_results: list[ScheduleRiskSimulationResult] = field(default_factory=list, init=False)  # Each R_i^{t,j}.
    schedule_risk_total_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sum_j R_i^{t,j}.

    def __post_init__(self) -> None:
        if not isinstance(self.work_package_id, int):  # W is indexed by integers.
            raise TypeError("work_package_id must be an integer.")
        if self.work_package_id <= 0:  # The report uses W = {1, ..., N}.
            raise ValueError("work_package_id must be positive.")

    def add_schedule_risk(self, schedule_risk: ScheduleRisk) -> None:
        """Attach one time-risk model R_i^{t,j} to this work package."""
        if not isinstance(schedule_risk, ScheduleRisk):
            raise TypeError("schedule_risk must be a ScheduleRisk.")
        self.schedule_risks.append(schedule_risk)  # Append risk j to the activity's time-risk list.

    def simulate_schedule_risk(
        self,
        risk_index: int,
        sample_count: int,
        rng: np.random.Generator,
    ) -> ScheduleRiskSimulationResult:
        """Simulate one schedule risk R_i^{t,j} for this activity."""
        try:
            schedule_risk = self.schedule_risks[risk_index]  # Select risk j assigned to work package i.
        except IndexError as exc:
            raise IndexError(f"unknown schedule risk index {risk_index}.") from exc
        return schedule_risk.simulate(sample_count, rng)  # Execute Algorithm: P_i^j, U_i^j, xi_i^j, R_i^j.

    def simulate_schedule_risks(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        """Simulate all schedule risks and store their summed time effect."""
        self.schedule_risk_results = []  # Replace stale risk vectors when n or RNG state changes.
        total_effects = np.zeros(sample_count, dtype=np.uint64)  # Accumulator sum_j R_i^{t,j}.

        for risk_index in range(len(self.schedule_risks)):
            result = self.simulate_schedule_risk(risk_index, sample_count, rng)  # Draw one R_i^{t,j}.
            self.schedule_risk_results.append(result)  # Store detailed vectors for diagnostics.
            total_effects += result.effects.astype(np.uint64)  # Add realized time risk to activity i.
            if np.any(total_effects > UINT32_MAX):
                raise OverflowError("schedule risk effects exceed uint32 capacity.")

        self.schedule_risk_total_samples = total_effects.astype(np.uint32, copy=False)  # Store sum_j R_i^{t,j}.
        return self.schedule_risk_total_samples  # Return summed schedule-risk time impact.

    def simulate_duration(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        """Generate and store the Monte Carlo vector for total duration D_i."""
        baseline_samples = self.duration_model.sample(sample_count, rng)  # Draw baseline duration Z_i.
        if baseline_samples.dtype != np.uint32:  # Normalize custom distributions to the storage type.
            baseline_samples = baseline_samples.astype(np.uint32, copy=False)  # Keep storage compact.
        self.baseline_duration_samples = baseline_samples  # Store Z_i separately from risk-loaded D_i.

        risk_effects = self.simulate_schedule_risks(sample_count, rng)  # Compute sum_j R_i^{t,j}.
        return self.assemble_duration_from_schedule_risk_sum(risk_effects)  # Save D_i = Z_i + sum_j R_i^{t,j}.

    def assemble_duration_from_schedule_risk_sum(
        self,
        risk_effects: NDArray[np.uint32],
    ) -> NDArray[np.uint32]:
        """Store D_i after adding a precomputed schedule-risk sum to Z_i."""
        if self.baseline_duration_samples is None:
            raise RuntimeError("baseline duration samples are missing.")
        if risk_effects.shape != self.baseline_duration_samples.shape:
            raise ValueError("risk effect samples must match baseline duration samples.")

        duration_samples = (  # Assemble D_i = Z_i + sum_j R_i^{t,j}.
            self.baseline_duration_samples.astype(np.uint64)  # Z_i, widened before risk addition.
            + risk_effects.astype(np.uint64)  # Sum of realized schedule-risk effects.
        )
        if np.any(duration_samples > UINT32_MAX):
            raise OverflowError("duration samples exceed uint32 capacity.")

        self.duration_samples = duration_samples.astype(np.uint32, copy=False)  # Store total duration D_i.
        return self.duration_samples  # Return D_i for immediate use if needed.

    def set_timing_samples(
        self,
        start_samples: NDArray[np.uint32],
        finish_samples: NDArray[np.uint32],
    ) -> None:
        """Store sampled start and finish vectors for this work package."""
        if start_samples.shape != finish_samples.shape:
            raise ValueError("start and finish samples must have the same shape.")
        self.start_samples = start_samples.astype(np.uint32, copy=False)  # Store S_i on the activity.
        self.finish_samples = finish_samples.astype(np.uint32, copy=False)  # Store T_i on the activity.

    def set_lag_samples(
        self,
        predecessor: int,
        lag_samples: NDArray[np.uint32],
    ) -> None:
        """Store sampled incoming lag L_{predecessor,i} for this work package."""
        self.lag_samples_by_predecessor[int(predecessor)] = lag_samples.astype(np.uint32, copy=False)  # Store L_{j,i}.

    def get_start_samples(self) -> NDArray[np.uint32]:
        """Return sampled earliest starts S_i after project-time simulation."""
        if self.start_samples is None:
            raise RuntimeError("start samples are missing; run a project-time simulation first.")
        return self.start_samples

    def get_finish_samples(self) -> NDArray[np.uint32]:
        """Return sampled earliest finishes T_i after project-time simulation."""
        if self.finish_samples is None:
            raise RuntimeError("finish samples are missing; run a project-time simulation first.")
        return self.finish_samples

    def get_lag_samples(self, predecessor: int) -> NDArray[np.uint32]:
        """Return sampled incoming lag L_{predecessor,i} after project-time simulation."""
        try:
            return self.lag_samples_by_predecessor[int(predecessor)]
        except KeyError as exc:
            raise KeyError(f"lag samples from predecessor {predecessor} are missing.") from exc
