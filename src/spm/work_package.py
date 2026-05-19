"""Work packages, the atomic activities in the project network."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spm.distributions import (
    ContinuousDistribution,
    DeterministicContinuousDistribution,
    DurationDistribution,
    ProbabilityDistribution,
    UINT32_MAX,
)


@dataclass(frozen=True)
class ScheduleRiskSimulationResult:
    """Monte Carlo output for one schedule-risk effect R_i^{t,j}."""

    probabilities: NDArray[np.float64]  # Sampled occurrence probabilities P_i^j.
    severities: NDArray[np.uint32]  # Sampled discrete severities U_i^j.
    occurrences: NDArray[np.bool_]  # Sampled Bernoulli indicators xi_i^j.
    effects: NDArray[np.uint32]  # Realized risk effects R_i^{t,j} = xi_i^j U_i^j.
    expected_effect: float  # E[R_i^{t,j}], using independence of occurrence probability and severity.


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
        expected_effect = _expected_value(self.probability_model, probabilities) * _expected_value(
            self.severity_model,
            severities,
        )  # E[xi U] = E[P]E[U] when P and U are sampled independently.
        return ScheduleRiskSimulationResult(  # Keep every vector needed for diagnostics and later cost extension.
            probabilities=probabilities,  # P_i^j samples.
            severities=severities,  # U_i^j samples.
            occurrences=occurrences,  # xi_i^j samples.
            effects=effects,  # R_i^{t,j} samples.
            expected_effect=expected_effect,  # Analytic mean when available, otherwise sample mean fallback.
        )


@dataclass(frozen=True)
class CostRiskSimulationResult:
    """Monte Carlo output for one cost-risk effect R_i^{c,j}."""

    probabilities: NDArray[np.float64]  # Sampled occurrence probabilities P_i^j.
    severities: NDArray[np.float64]  # Sampled continuous cost severities U_i^j.
    occurrences: NDArray[np.bool_]  # Sampled Bernoulli indicators xi_i^j.
    effects: NDArray[np.float64]  # Realized risk effects R_i^{c,j} = xi_i^j U_i^j.
    expected_effect: float  # E[R_i^{c,j}], using independence of occurrence probability and severity.


@dataclass(frozen=True)
class CostRisk:
    """Cost-risk model attached to a work package."""

    probability_model: ProbabilityDistribution  # Law of P_i^j.
    severity_model: ContinuousDistribution  # Law of the continuous cost severity U_i^j.

    def simulate(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> CostRiskSimulationResult:
        """Simulate one continuous cost-risk effect vector R_i^{c,j}."""
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

        severities = np.asarray(self.severity_model.sample(sample_count, rng), dtype=np.float64)  # Draw U_i^j.
        if severities.shape != (sample_count,):
            raise ValueError("severity samples must be a vector of length sample_count.")
        if not np.all(np.isfinite(severities)) or np.any(severities < 0):
            raise ValueError("severity samples must be nonnegative finite values.")

        occurrences = rng.random(sample_count) < probabilities  # xi_i^j | P_i^j ~ Bernoulli(P_i^j).
        effects = occurrences.astype(np.float64) * severities  # R_i^j = xi_i^j U_i^j.
        expected_effect = _expected_value(self.probability_model, probabilities) * _expected_value(
            self.severity_model,
            severities,
        )  # E[xi U] = E[P]E[U] when P and U are sampled independently.
        return CostRiskSimulationResult(
            probabilities=probabilities,  # P_i^j samples.
            severities=severities,  # U_i^j samples.
            occurrences=occurrences,  # xi_i^j samples.
            effects=effects,  # R_i^{c,j} samples.
            expected_effect=expected_effect,  # Analytic mean when available, otherwise sample mean fallback.
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
    daily_cost_model: ContinuousDistribution | None = None  # Law of K_i, the daily cost.
    fixed_costs: list[ContinuousDistribution | float] = field(default_factory=list)  # Fixed-cost models H_i^j.
    cost_risks: list[CostRisk] = field(default_factory=list)  # Cost risks assigned to activity i.
    baseline_duration_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of Z_i.
    duration_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of D_i.
    daily_cost_samples: NDArray[np.float64] | None = field(default=None, init=False)  # Sample vector of K_i.
    fixed_cost_samples: list[NDArray[np.float64]] = field(default_factory=list, init=False)  # Samples of H_i^j.
    baseline_cost_samples: NDArray[np.float64] | None = field(default=None, init=False)  # Sample vector of A_i.
    cost_samples: NDArray[np.float64] | None = field(default=None, init=False)  # Sample vector of total cost C_i.
    start_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of S_i.
    finish_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sample vector of T_i.
    lag_samples_by_predecessor: dict[int, NDArray[np.uint32]] = field(default_factory=dict, init=False)  # Incoming L_{j,i}.
    schedule_risk_results: list[ScheduleRiskSimulationResult] = field(default_factory=list, init=False)  # Each R_i^{t,j}.
    schedule_risk_total_samples: NDArray[np.uint32] | None = field(default=None, init=False)  # Sum_j R_i^{t,j}.
    cost_risk_results: list[CostRiskSimulationResult] = field(default_factory=list, init=False)  # Each R_i^{c,j}.
    cost_risk_total_samples: NDArray[np.float64] | None = field(default=None, init=False)  # Sum_j R_i^{c,j}.
    expected_baseline_duration: float | None = field(default=None, init=False)  # E[Z_i].
    expected_schedule_risk_total: float | None = field(default=None, init=False)  # Sum_j E[R_i^{t,j}].
    expected_duration: float | None = field(default=None, init=False)  # E[D_i].
    expected_baseline_cost: float | None = field(default=None, init=False)  # E[A_i].
    expected_cost_risk_total: float | None = field(default=None, init=False)  # Sum_j E[R_i^{c,j}].
    expected_total_cost: float | None = field(default=None, init=False)  # E[C_i].

    def __post_init__(self) -> None:
        if not isinstance(self.work_package_id, int):  # W is indexed by integers.
            raise TypeError("work_package_id must be an integer.")
        if self.work_package_id <= 0:  # The report uses W = {1, ..., N}.
            raise ValueError("work_package_id must be positive.")
        self.fixed_costs = [_coerce_continuous_distribution(cost) for cost in self.fixed_costs]

    def add_schedule_risk(self, schedule_risk: ScheduleRisk) -> None:
        """Attach one time-risk model R_i^{t,j} to this work package."""
        if not isinstance(schedule_risk, ScheduleRisk):
            raise TypeError("schedule_risk must be a ScheduleRisk.")
        self.schedule_risks.append(schedule_risk)  # Append risk j to the activity's time-risk list.

    def set_cost_model(
        self,
        daily_cost_model: ContinuousDistribution,
        fixed_costs: list[ContinuousDistribution | float] | None = None,
    ) -> None:
        """Set the daily cost model K_i and optional fixed costs H_i^j."""
        self.daily_cost_model = daily_cost_model  # Store the law used to sample K_i.
        self.fixed_costs = [
            _coerce_continuous_distribution(cost)
            for cost in (fixed_costs or [])
        ]  # Store continuous H_i^j models.

    def add_fixed_cost(self, fixed_cost: ContinuousDistribution | float) -> None:
        """Add one fixed-cost model H_i^j."""
        self.fixed_costs.append(_coerce_continuous_distribution(fixed_cost))  # Store H_i^j as a continuous model.

    def add_cost_risk(self, cost_risk: CostRisk) -> None:
        """Attach one cost-risk model R_i^{c,j} to this work package."""
        if not isinstance(cost_risk, CostRisk):
            raise TypeError("cost_risk must be a CostRisk.")
        self.cost_risks.append(cost_risk)  # Append risk j to the activity's cost-risk list.

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
        expected_total = 0.0  # Accumulator sum_j E[R_i^{t,j}].

        for risk_index in range(len(self.schedule_risks)):
            result = self.simulate_schedule_risk(risk_index, sample_count, rng)  # Draw one R_i^{t,j}.
            self.schedule_risk_results.append(result)  # Store detailed vectors for diagnostics.
            total_effects += result.effects.astype(np.uint64)  # Add realized time risk to activity i.
            expected_total += result.expected_effect  # Add independent expected schedule-risk effect.
            if np.any(total_effects > UINT32_MAX):
                raise OverflowError("schedule risk effects exceed uint32 capacity.")

        self.schedule_risk_total_samples = total_effects.astype(np.uint32, copy=False)  # Store sum_j R_i^{t,j}.
        self.expected_schedule_risk_total = expected_total  # Store sum_j E[R_i^{t,j}].
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
        self.expected_baseline_duration = _expected_value(self.duration_model, baseline_samples)  # Store E[Z_i].

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
        if self.expected_baseline_duration is None:
            self.expected_baseline_duration = _expected_value(self.duration_model, self.baseline_duration_samples)
        self.expected_duration = self.expected_baseline_duration + (self.expected_schedule_risk_total or 0.0)  # E[D_i].
        return self.duration_samples  # Return D_i for immediate use if needed.

    def simulate_baseline_cost(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Simulate accumulated baseline cost A_i = K_i D_i + sum_j H_i^j."""
        if self.duration_samples is None:
            raise RuntimeError("duration samples are missing; simulate duration before cost.")
        if len(self.duration_samples) != sample_count:
            raise ValueError("duration samples must have length sample_count.")
        if self.daily_cost_model is None:
            raise RuntimeError("daily cost model is missing.")

        daily_cost_samples = np.asarray(  # Generate K_i as a vector of length n.
            self.daily_cost_model.sample(sample_count, rng),
            dtype=np.float64,
        )
        if daily_cost_samples.shape != (sample_count,):
            raise ValueError("daily cost samples must be a vector of length sample_count.")
        if not np.all(np.isfinite(daily_cost_samples)) or np.any(daily_cost_samples < 0):
            raise ValueError("daily cost samples must be nonnegative finite values.")

        baseline_costs = daily_cost_samples * self.duration_samples.astype(np.float64)  # A_i <- K_i o D_i.
        self.fixed_cost_samples = []  # Replace stale H_i^j sample vectors.
        expected_fixed_cost_total = 0.0  # Accumulator sum_j E[H_i^j].
        for fixed_cost_model in self.fixed_costs:
            fixed_cost_samples = np.asarray(fixed_cost_model.sample(sample_count, rng), dtype=np.float64)  # Draw H_i^j.
            if fixed_cost_samples.shape != (sample_count,):
                raise ValueError("fixed cost samples must be a vector of length sample_count.")
            if not np.all(np.isfinite(fixed_cost_samples)) or np.any(fixed_cost_samples < 0):
                raise ValueError("fixed cost samples must be nonnegative finite values.")
            baseline_costs = baseline_costs + fixed_cost_samples  # Add sampled fixed cost H_i^j.
            self.fixed_cost_samples.append(fixed_cost_samples)  # Store H_i^j samples.
            expected_fixed_cost_total += _expected_value(fixed_cost_model, fixed_cost_samples)  # Add E[H_i^j].

        self.daily_cost_samples = daily_cost_samples  # Store sampled K_i.
        self.baseline_cost_samples = baseline_costs.astype(np.float64, copy=False)  # Store A_i.
        if self.expected_duration is None:
            raise RuntimeError("expected duration is missing; simulate duration before cost.")
        # K_i is sampled independently of D_i, so E[K_i D_i] = E[K_i]E[D_i].
        self.expected_baseline_cost = (
            _expected_value(self.daily_cost_model, daily_cost_samples)
            * self.expected_duration
            + expected_fixed_cost_total
        )
        return self.baseline_cost_samples

    def simulate_cost_risk(
        self,
        risk_index: int,
        sample_count: int,
        rng: np.random.Generator,
    ) -> CostRiskSimulationResult:
        """Simulate one cost risk R_i^{c,j} for this activity."""
        try:
            cost_risk = self.cost_risks[risk_index]  # Select risk j assigned to work package i.
        except IndexError as exc:
            raise IndexError(f"unknown cost risk index {risk_index}.") from exc
        return cost_risk.simulate(sample_count, rng)  # Execute Algorithm: P_i^j, U_i^j, xi_i^j, R_i^j.

    def simulate_cost_risks(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Simulate all cost risks and store their summed cost effect."""
        self.cost_risk_results = []  # Replace stale risk vectors when n or RNG state changes.
        total_effects = np.zeros(sample_count, dtype=np.float64)  # Accumulator sum_j R_i^{c,j}.
        expected_total = 0.0  # Accumulator sum_j E[R_i^{c,j}].

        for risk_index in range(len(self.cost_risks)):
            result = self.simulate_cost_risk(risk_index, sample_count, rng)  # Draw one R_i^{c,j}.
            self.cost_risk_results.append(result)  # Store detailed vectors for diagnostics.
            total_effects = total_effects + result.effects  # Add realized cost risk to activity i.
            expected_total += result.expected_effect  # Add independent expected cost-risk effect.

        self.cost_risk_total_samples = total_effects.astype(np.float64, copy=False)  # Store sum_j R_i^{c,j}.
        self.expected_cost_risk_total = expected_total  # Store sum_j E[R_i^{c,j}].
        return self.cost_risk_total_samples

    def simulate_cost(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Simulate total cost C_i = A_i + sum_j R_i^{c,j}."""
        baseline_costs = self.simulate_baseline_cost(sample_count, rng)  # Generate A_i from K_i, D_i, and H_i^j.
        risk_effects = self.simulate_cost_risks(sample_count, rng)  # Compute sum_j R_i^{c,j}.
        self.cost_samples = baseline_costs + risk_effects  # C_i <- A_i + sum_j R_i^{c,j}.
        self.expected_total_cost = (self.expected_baseline_cost or 0.0) + (self.expected_cost_risk_total or 0.0)
        return self.cost_samples

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

    def get_cost_samples(self) -> NDArray[np.float64]:
        """Return sampled total costs C_i after cost simulation."""
        if self.cost_samples is None:
            raise RuntimeError("cost samples are missing; run a cost simulation first.")
        return self.cost_samples

    def get_expected_total_cost(self) -> float:
        """Return E[C_i] after cost simulation."""
        if self.expected_total_cost is None:
            raise RuntimeError("expected total cost is missing; run a cost simulation first.")
        return self.expected_total_cost


def _expected_value(model: object, samples: NDArray[np.float64]) -> float:
    """Return analytic expected value when provided, otherwise use sample mean."""
    expected = getattr(model, "expected_value", None)
    if expected is not None:
        return float(expected)  # Distribution classes expose expected_value as a property.
    return float(np.mean(samples))  # Fallback for user-provided models without an analytic mean.


def _coerce_continuous_distribution(value: ContinuousDistribution | float) -> ContinuousDistribution:
    """Return a continuous distribution, converting plain numbers to degenerate laws."""
    if hasattr(value, "sample"):
        return value  # User-provided continuous random variable.
    return DeterministicContinuousDistribution(float(value))  # Numeric shorthand for H_i^j = value.
