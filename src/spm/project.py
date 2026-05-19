"""Project-network simulation for earliest starts, finishes, and critical paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from numbers import Integral
from typing import Callable, Protocol, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from spm.distributions import ContinuousDistribution, DurationDistribution, ShiftedPoissonDuration, UINT32_MAX
from spm.work_package import CostRisk, ScheduleRisk, WorkPackage


BYTES_PER_UINT32 = np.dtype(np.uint32).itemsize  # One D_i sample is stored as one uint32.
BYTES_PER_UINT64 = np.dtype(np.uint64).itemsize  # One sampled predecessor ID is stored as one uint64.
BYTES_PER_SIM_ACTIVITY_SAMPLE = (  # Conservative storage estimate for full project-time simulation.
    5 * BYTES_PER_UINT32  # D_i, S_i dict, T_i dict, public starts matrix, public finishes matrix.
    + BYTES_PER_UINT64  # p_i, the sampled predecessor used for critical path reconstruction.
)
GIB = 1024**3  # Memory budgets are expressed in binary gigabytes.
PROBABILITY_DTYPE = np.longdouble  # Use the widest floating type NumPy exposes on this platform.

# The predecessor vectors store work-package IDs, which are positive. This
# largest uint64 value therefore represents the mathematical empty predecessor
# p_i = empty without mixing signed and unsigned integer arrays.
NO_PREDECESSOR = np.iinfo(np.uint64).max  # Sentinel for p_i = empty.


@dataclass(frozen=True)
class ProjectTimeSimulationResult:
    """Monte Carlo realization of the project-network timing variables.

    Rows in starts and finishes follow topological_order. Columns are Monte
    Carlo samples. Thus starts[k, s] is S_i for the kth topological activity in
    sample s, and finishes[k, s] is the corresponding T_i.
    """

    starts: NDArray[np.uint32]  # Sample matrix of S_i values.
    finishes: NDArray[np.uint32]  # Sample matrix of T_i values.
    completion_times: NDArray[np.uint32]  # Sample vector of T_project = max_i T_i.
    critical_paths: list[list[int]]  # One reconstructed critical path C per sample.
    topological_order: list[int]  # The activity order T used in the recursion.
    lag_samples_by_edge: dict[tuple[int, int], NDArray[np.uint32]] = field(default_factory=dict)  # Sampled L_{j,i}.


class LagModel(Protocol):
    """Interface for a dependency lag L_{j,i}."""

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        """Generate Monte Carlo samples of L_{j,i} as nonnegative integers."""


@dataclass(frozen=True)
class DeterministicLag:
    """Deterministic dependency lag model L_{j,i} = lag."""

    lag: int  # Fixed nonnegative lag on edge j -> i.

    def __post_init__(self) -> None:
        if not isinstance(self.lag, Integral):
            raise TypeError("lag must be an integer.")
        object.__setattr__(self, "lag", int(self.lag))  # Normalize NumPy integer scalars.
        if self.lag < 0:
            raise ValueError("lag must be nonnegative.")
        if self.lag > UINT32_MAX:
            raise ValueError("lag must fit in uint32.")

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> NDArray[np.uint32]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        return np.full(sample_count, self.lag, dtype=np.uint32)  # Deterministic L_{j,i} for every scenario.


class Project:
    """Top-level project object containing W, Pred(i), and duration laws.

    The graph is the fixed schedule S from the report. Its nodes are integer
    work packages and its directed edges j -> i mean j is in Pred(i).
    """

    def __init__(
        self,
        sample_count: int | None = None,
        rng_seed: int | None = None,
        max_memory_bytes: int | None = None,
        memory_provider: Callable[[], int] | None = None,
    ) -> None:
        if sample_count is not None and (
            not isinstance(sample_count, Integral) or sample_count <= 0
        ):
            raise ValueError("sample_count must be positive when provided.")
        if max_memory_bytes is not None and max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive when provided.")

        self.sample_count = int(sample_count) if sample_count is not None else None  # Common sample dimension n.
        self.max_memory_bytes = max_memory_bytes  # Optional cap on the automatic memory budget.
        self._memory_provider = memory_provider or _system_memory_bytes  # Source for total RAM.
        self.rng = np.random.default_rng(rng_seed)  # Random generator used for all sampled variables.
        self.work_packages: dict[int, WorkPackage] = {}  # The set W, keyed by activity ID i.
        self.graph = nx.DiGraph()  # The deterministic schedule graph S.
        self.time_simulation_result: ProjectTimeSimulationResult | None = None  # Last project-time sample.
        self.completion_time_samples: NDArray[np.uint32] | None = None  # Last sampled T_project vector.
        self.critical_activity_samples: NDArray[np.bool_] | None = None  # Last all-critical-activity matrix C.
        self.critical_activity_topological_order: list[int] | None = None  # Row labels for C.

    def add_work_package(
        self,
        work_package_id: int,
        duration_model: DurationDistribution,
    ) -> None:
        """Add activity i with its duration distribution D_i."""
        if not isinstance(work_package_id, Integral):
            raise TypeError("work_package_id must be an integer.")
        work_package_id = int(work_package_id)  # Normalize NumPy integer scalars to Python int.
        if work_package_id <= 0:
            raise ValueError("work_package_id must be positive.")
        if work_package_id in self.work_packages:
            raise ValueError(f"work package {work_package_id} already exists.")

        self.work_packages[work_package_id] = WorkPackage(  # Register activity i in W.
            work_package_id=work_package_id,  # The integer label i.
            duration_model=duration_model,  # The probability law of D_i.
        )
        self.graph.add_node(work_package_id)  # Add i as a node in the schedule graph S.

    def add_schedule_risk(
        self,
        work_package_id: int,
        schedule_risk: ScheduleRisk,
    ) -> None:
        """Attach one schedule-risk model R_i^{t,j} to activity i."""
        if not isinstance(work_package_id, Integral):
            raise TypeError("work_package_id must be an integer.")
        work_package_id = int(work_package_id)  # Normalize NumPy integer scalar labels.
        try:
            work_package = self.work_packages[work_package_id]  # Select the activity i.
        except KeyError as exc:
            raise KeyError(f"unknown work package {work_package_id}.") from exc
        work_package.add_schedule_risk(schedule_risk)  # Assign risk j to work package i.

    def set_work_package_cost_model(
        self,
        work_package_id: int,
        daily_cost_model: ContinuousDistribution,
        fixed_costs: list[ContinuousDistribution | float] | None = None,
    ) -> None:
        """Set the daily cost model K_i and fixed costs H_i^j for activity i."""
        work_package = self._work_package_by_id(work_package_id)  # Select the activity i.
        work_package.set_cost_model(daily_cost_model, fixed_costs)  # Store K_i and H_i^j.

    def add_fixed_cost(
        self,
        work_package_id: int,
        fixed_cost: ContinuousDistribution | float,
    ) -> None:
        """Attach one fixed-cost model H_i^j to activity i."""
        work_package = self._work_package_by_id(work_package_id)  # Select the activity i.
        work_package.add_fixed_cost(fixed_cost)  # Store H_i^j.

    def add_cost_risk(
        self,
        work_package_id: int,
        cost_risk: CostRisk,
    ) -> None:
        """Attach one cost-risk model R_i^{c,j} to activity i."""
        work_package = self._work_package_by_id(work_package_id)  # Select the activity i.
        work_package.add_cost_risk(cost_risk)  # Assign cost risk j to work package i.

    def add_dependency(
        self,
        predecessor: int,
        successor: int,
        lag: int | LagModel = 0,
    ) -> None:
        """Add predecessor -> successor, meaning predecessor in Pred(successor).

        Lags are stored on dependency edges as L_{predecessor,successor}. The
        current concrete implementation is deterministic, but sampled lag
        models can later satisfy the same interface.
        """
        if not isinstance(predecessor, Integral) or not isinstance(successor, Integral):
            raise TypeError("dependencies must use integer work package IDs.")
        predecessor = int(predecessor)  # Normalize NumPy integer scalar labels.
        successor = int(successor)  # Normalize NumPy integer scalar labels.
        if predecessor <= 0 or successor <= 0:
            raise ValueError("dependencies must use positive work package IDs.")
        self.graph.add_edge(  # Encode predecessor in Pred(successor), with separate lag model.
            predecessor,
            successor,
            lag_model=_coerce_lag_model(lag),  # Store L_{predecessor,successor} on the edge.
        )

    def calculate_default_sample_count(self) -> int:
        """Choose a common Monte Carlo sample size for every D_i.

        The full project-time simulation stores more than D_i. It also stores
        S_i, T_i, output starts/finishes matrices, and p_i. The automatic
        sample count therefore uses a conservative per-activity simulation
        estimate rather than only the duration-vector storage.
        """
        if self.sample_count is not None:
            return self.sample_count  # Use the user-specified Monte Carlo dimension n.
        if not self.work_packages:
            raise ValueError("cannot calculate sample count without work packages.")

        system_ram = self._memory_provider()  # Total physical memory available for sizing n.
        if system_ram <= 2 * GIB:
            raise ValueError("system RAM must exceed 2 GiB for automatic sample sizing.")

        memory_budget = min(10 * GIB, system_ram - 2 * GIB)  # min(10 GiB, RAM - 2 GiB).
        if self.max_memory_bytes is not None:
            memory_budget = min(memory_budget, self.max_memory_bytes)  # Apply optional stricter cap.

        sample_count = memory_budget // (
            BYTES_PER_SIM_ACTIVITY_SAMPLE * len(self.work_packages)  # Approx bytes per activity sample.
        )
        if sample_count <= 0:
            raise ValueError("memory budget is too small for one sample per work package.")
        return int(sample_count)  # Equal n for every work package in this prototype.

    def topological_order(self) -> list[int]:
        """Return a topological order T of the acyclic schedule graph."""
        missing_nodes = sorted(set(self.graph.nodes) - set(self.work_packages))  # Nodes without a D_i law.
        if missing_nodes:
            raise ValueError(
                "graph contains nodes without work packages: "
                f"{missing_nodes}."
            )

        try:
            return list(nx.topological_sort(self.graph))  # The order T used by the dynamic program.
        except nx.NetworkXUnfeasible as exc:
            raise ValueError("project dependency graph must be a DAG.") from exc

    def complete_paths(self) -> list[list[int]]:
        """Return all complete source-to-sink paths p_1, ..., p_q in the DAG."""
        self.topological_order()  # Validate that S is a DAG and every node has a duration law.
        sources = [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]  # Activities with Pred(i)=empty.
        sinks = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]  # Terminal activities.
        paths: list[list[int]] = []  # This will become P = {p_1, ..., p_q}.

        for source in sources:
            for sink in sinks:
                if source == sink:  # Isolated activity: the complete path is just [i].
                    paths.append([source])
                else:
                    paths.extend(
                        list(nx.all_simple_paths(self.graph, source, sink))  # DAG paths from source to sink.
                    )

        return paths  # Every path is represented as a list of activity IDs.

    def project_completion_cdf_shifted_poisson(
        self,
        t: int,
        paths: Sequence[Sequence[int]] | None = None,
    ) -> float:
        """Exact prototype CDF P(T_project <= t) for shifted-Poisson durations.

        This is a deliberately narrow prototyping alternative. It only applies
        when every duration model is ShiftedPoissonDuration and the activity
        durations are interpreted as independent. It is exact, but the product
        over shared path-incidence groups can grow exponentially.
        """
        if not isinstance(t, Integral):  # The report works in discrete time.
            raise TypeError("t must be an integer threshold.")
        t = int(t)  # Normalize NumPy integer scalar thresholds.
        if t < 0:
            return 0.0  # Nonnegative durations imply P(T_project <= t)=0 for t<0.

        complete_paths = self._normalize_complete_paths(paths)  # P = {p_1, ..., p_q}.
        parameters = self._shifted_poisson_parameters_by_node()  # Maps i to (lambda_i, a_i).
        path_count = len(complete_paths)  # q, the number of complete paths.

        group_lambdas: dict[tuple[int, ...], float] = {}  # Lambda_S for each incidence set S.
        group_shifts: dict[tuple[int, ...], int] = {}  # A_S for each incidence set S.
        path_sets = [set(path) for path in complete_paths]  # Speeds up tests of i in p_theta.

        for node, duration in parameters.items():
            incidence = tuple(  # S_i = {theta: i in p_theta}, using zero-based theta internally.
                theta
                for theta, path_set in enumerate(path_sets)
                if node in path_set
            )
            if not incidence:
                raise ValueError(f"activity {node} does not appear in any complete path.")
            group_lambdas[incidence] = group_lambdas.get(incidence, 0.0) + duration.lambda_  # Lambda_S.
            group_shifts[incidence] = group_shifts.get(incidence, 0) + duration.a  # A_S.

        path_minimums = [  # A_theta = sum_{S: theta in S} A_S.
            sum(
                shift_sum  # A_S contribution to path theta.
                for incidence, shift_sum in group_shifts.items()
                if theta in incidence
            )
            for theta in range(path_count)
        ]
        deterministic_project_minimum = max(path_minimums)  # min possible max_theta D(p_theta).
        if t < deterministic_project_minimum:
            return 0.0  # T_project cannot be below the largest deterministic path minimum.

        remaining_slacks = [t - path_minimum for path_minimum in path_minimums]  # b_theta(t)=t-A_theta.

        shared_groups = [  # G = {S: |S| >= 2 and Lambda_S > 0}.
            incidence
            for incidence, lambda_sum in group_lambdas.items()
            if len(incidence) >= 2 and lambda_sum > 0
        ]
        shared_groups.sort()  # Deterministic enumeration of the product space.
        upper_bounds = [  # u_S(t)=min_{theta in S} b_theta(t).
            min(remaining_slacks[theta] for theta in incidence)
            for incidence in shared_groups
        ]
        singleton_lambdas = [  # Lambda_{theta}: stochastic mass unique to path theta.
            group_lambdas.get((theta,), 0.0)
            for theta in range(path_count)
        ]

        probability = PROBABILITY_DTYPE(0)  # Accumulator P in the algorithm.
        summation_ranges = [range(upper_bound + 1) for upper_bound in upper_bounds]  # Product domain for z_S.
        for shared_values in product(*summation_ranges):  # Enumerate (z_S)_{S in G}.
            shared_by_group = dict(zip(shared_groups, shared_values, strict=True))  # Attach each z_S to S.
            term_probability = PROBABILITY_DTYPE(1)  # P_z before adding it into P.

            for incidence, z_value in shared_by_group.items():
                term_probability *= _poisson_pmf(group_lambdas[incidence], z_value)  # P(Z_S=z_S).

            for theta in range(path_count):
                shared_sum = sum(  # Sum of z_S over shared groups that affect path theta.
                    z_value
                    for incidence, z_value in shared_by_group.items()
                    if theta in incidence
                )
                residual_slack = remaining_slacks[theta] - shared_sum  # r_theta in the algorithm.
                if residual_slack < 0:
                    term_probability = PROBABILITY_DTYPE(0)  # This z choice makes path theta exceed t.
                    break
                term_probability *= _poisson_cdf(
                    singleton_lambdas[theta],
                    residual_slack,
                )  # F_{Poi(Lambda_{theta})}(r_theta).

            probability += term_probability  # P <- P + P_z.

        return min(max(probability, PROBABILITY_DTYPE(0)), PROBABILITY_DTYPE(1))  # Clamp roundoff outside [0,1].

    def simulate_work_package(self, work_package_id: int) -> NDArray[np.uint32]:
        """Simulate and store the duration vector D_i for one activity."""
        if not isinstance(work_package_id, Integral):
            raise TypeError("work_package_id must be an integer.")
        work_package_id = int(work_package_id)  # Normalize NumPy integer scalar labels.
        try:
            work_package = self.work_packages[work_package_id]  # Select the activity i.
        except KeyError as exc:
            raise KeyError(f"unknown work package {work_package_id}.") from exc

        return work_package.simulate_duration(
            self.calculate_default_sample_count(),
            self.rng,
        )

    def simulate_all_work_packages(self) -> None:
        """Simulate and store D_i for every activity i in W."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        for work_package in self.work_packages.values():
            work_package.simulate_duration(sample_count, self.rng)  # Draw the vector D_i.

    def simulate_work_package_schedule_risks(self, work_package_id: int) -> NDArray[np.uint32]:
        """Simulate and store the summed schedule-risk vector for one activity."""
        if not isinstance(work_package_id, Integral):
            raise TypeError("work_package_id must be an integer.")
        work_package_id = int(work_package_id)  # Normalize NumPy integer scalar labels.
        try:
            work_package = self.work_packages[work_package_id]  # Select the activity i.
        except KeyError as exc:
            raise KeyError(f"unknown work package {work_package_id}.") from exc

        return work_package.simulate_schedule_risks(  # Draw and sum R_i^{t,1}, ..., R_i^{t,r_i^t}.
            self.calculate_default_sample_count(),
            self.rng,
        )

    def simulate_all_schedule_risks(self) -> None:
        """Simulate and store summed schedule-risk vectors for all activities."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        for work_package in self.work_packages.values():
            work_package.simulate_schedule_risks(sample_count, self.rng)  # Store sum_j R_i^{t,j}.

    def simulate_work_package_baseline_cost(self, work_package_id: int) -> NDArray[np.float64]:
        """Simulate and store baseline cost A_i for one activity."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        work_package = self._work_package_by_id(work_package_id)  # Select the activity i.
        self._ensure_work_package_duration_samples(work_package, sample_count)  # Ensure D_i exists before A_i.
        return work_package.simulate_baseline_cost(sample_count, self.rng)  # Draw K_i and assemble A_i.

    def simulate_all_baseline_costs(self) -> None:
        """Simulate and store baseline cost vectors A_i for all activities."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        self._ensure_duration_samples(sample_count)  # Ensure every D_i exists before K_i o D_i.
        for work_package in self.work_packages.values():
            work_package.simulate_baseline_cost(sample_count, self.rng)  # Store A_i.

    def simulate_work_package_cost_risks(self, work_package_id: int) -> NDArray[np.float64]:
        """Simulate and store the summed cost-risk vector for one activity."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        work_package = self._work_package_by_id(work_package_id)  # Select the activity i.
        return work_package.simulate_cost_risks(sample_count, self.rng)  # Draw and sum R_i^{c,j}.

    def simulate_all_cost_risks(self) -> None:
        """Simulate and store summed cost-risk vectors for all activities."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        for work_package in self.work_packages.values():
            work_package.simulate_cost_risks(sample_count, self.rng)  # Store sum_j R_i^{c,j}.

    def simulate_work_package_cost(self, work_package_id: int) -> NDArray[np.float64]:
        """Simulate and store total cost C_i for one activity."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        work_package = self._work_package_by_id(work_package_id)  # Select the activity i.
        self._ensure_work_package_duration_samples(work_package, sample_count)  # Ensure D_i exists before cost.
        return work_package.simulate_cost(sample_count, self.rng)  # Store C_i = A_i + sum_j R_i^{c,j}.

    def simulate_all_costs(self) -> None:
        """Simulate and store total cost vectors C_i for all activities."""
        sample_count = self.calculate_default_sample_count()  # Common sample dimension n.
        self._ensure_duration_samples(sample_count)  # Ensure every D_i exists before cost simulation.
        for work_package in self.work_packages.values():
            work_package.simulate_cost(sample_count, self.rng)  # Store C_i and E[C_i].

    def expected_project_cost(self) -> float:
        """Return sum_i E[C_i] after all work-package costs have been simulated."""
        total = 0.0  # Accumulator for the project-level expected cost.
        for work_package in self.work_packages.values():
            total += work_package.get_expected_total_cost()  # Linearity gives E[sum_i C_i] = sum_i E[C_i].
        return total

    def simulate_project_time(self) -> ProjectTimeSimulationResult:
        """Apply the longest-path recursion to all Monte Carlo samples.

        This mirrors the report algorithm:
        S_i = 0 and T_i = D_i when Pred(i) is empty.
        Otherwise S_i = max_{j in Pred(i)} (T_j + L_{j,i}) and T_i = S_i + D_i.

        The max is vectorized over samples. For each activity, only the
        predecessor finish vectors needed for that local argmax are stacked
        temporarily; the work-package duration samples remain stored on their
        own WorkPackage objects.
        """
        order = self.topological_order()  # T, a topological order of the DAG.
        if not order:
            raise ValueError("cannot simulate an empty project.")

        sample_count = self.calculate_default_sample_count()  # Number of Monte Carlo scenarios.
        self._ensure_duration_samples(sample_count)  # Ensure every D_i vector exists.
        lag_samples_by_edge = self._simulate_dependency_lags(sample_count)  # Sample all L_{j,i} vectors.

        # These dictionaries are the sampled versions of the mathematical
        # vectors S_i, T_i, and p_i. Each value has one entry per Monte Carlo
        # realization.
        predecessors_by_node: dict[int, NDArray[np.uint64]] = {}  # Stores sampled p_i vectors.

        for node in order:
            work_package = self.work_packages[node]  # Activity i, where S_i and T_i will be stored.
            duration_samples = work_package.duration_samples  # The sampled D_i vector.
            if duration_samples is None:
                raise RuntimeError(f"duration samples are missing for activity {node}.")
            predecessor_nodes = list(self.graph.predecessors(node))  # The set Pred(i).
            predecessors_by_node[node] = np.full(  # Initialize p_i = empty for all samples.
                sample_count,  # One predecessor choice per Monte Carlo sample.
                NO_PREDECESSOR,  # Empty predecessor value.
                dtype=np.uint64,  # Unsigned integer storage for activity IDs.
            )

            if not predecessor_nodes:
                # Pred(i) is empty, so activity i can start at project time 0.
                start_samples = np.zeros(sample_count, dtype=np.uint32)  # S_i = 0.
            else:
                # Temporarily form the matrix (T_j + L_{j,i})_{j in Pred(i)} only for the
                # local argmax that defines S_i and p_i.
                predecessor_finish_lags = np.stack(  # Matrix with rows T_j + L_{j,i}.
                    [
                        self.work_packages[predecessor].get_finish_samples().astype(np.uint64)  # Previously computed T_j.
                        + lag_samples_by_edge[(predecessor, node)].astype(np.uint64)  # Edge lag L_{j,i}.
                        for predecessor in predecessor_nodes
                    ],
                    axis=0,  # Rows index predecessors; columns index samples.
                )
                if np.any(predecessor_finish_lags > UINT32_MAX):
                    raise OverflowError("start times exceed uint32 capacity.")
                selected_pred_offsets = np.argmax(predecessor_finish_lags, axis=0)  # argmax_j (T_j + L_{j,i}).
                start_samples = np.take_along_axis(  # S_i = max_{j in Pred(i)} (T_j + L_{j,i}).
                    predecessor_finish_lags,  # Candidate predecessor finish-plus-lag times.
                    selected_pred_offsets[np.newaxis, :],  # Winning predecessor row per sample.
                    axis=0,  # Max was taken over predecessor rows.
                )[0].astype(np.uint32, copy=False)  # Collapse the single selected row into a vector.
                predecessors_by_node[node] = np.asarray(  # p_i = selected predecessor m.
                    predecessor_nodes,  # Candidate predecessor activity IDs.
                    dtype=np.uint64,  # Store IDs as unsigned integers.
                )[
                    selected_pred_offsets  # Convert argmax offsets into actual activity IDs.
                ]

            # Compute T_i = S_i + D_i in uint64 first. This catches overflow
            # before storing back into the compact uint32 representation.
            row_finishes = (
                start_samples.astype(np.uint64)  # S_i, widened before addition.
                + duration_samples.astype(np.uint64)  # D_i, widened before addition.
            )
            if np.any(row_finishes > UINT32_MAX):
                raise OverflowError("project finish times exceed uint32 capacity.")
            work_package.set_timing_samples(  # Store sampled timing output on the activity itself.
                start_samples,
                row_finishes.astype(np.uint32, copy=False),  # T_i = S_i + D_i.
            )

        # The project completion time is the maximum terminal finish time. We
        # restrict the final argmax to sink activities so zero-duration ties do
        # not stop the reconstructed critical path at an internal activity.
        sink_nodes = [node for node in order if self.graph.out_degree(node) == 0]  # Terminal activities.
        sink_finishes = np.stack([self.work_packages[node].get_finish_samples() for node in sink_nodes], axis=0)  # Terminal T_i.
        completion_row_indices = np.argmax(sink_finishes, axis=0)  # r = argmax over sinks per sample.
        completion_nodes = np.asarray(sink_nodes, dtype=np.uint64)[completion_row_indices]  # Terminal r.
        completion_times = np.take_along_axis(  # T_project = T_r.
            sink_finishes,  # Candidate terminal finish times T_i.
            completion_row_indices[np.newaxis, :],  # Winning activity row r per sample.
            axis=0,  # Select over activities.
        )[0].astype(np.uint32, copy=False)  # Store the project duration vector.
        critical_paths = _build_critical_paths(  # Reconstruct C by following p_i backward.
            completion_nodes,  # Final activity r for each sample.
            predecessors_by_node,  # The sampled predecessor maps p_i.
        )

        # The public result is returned as matrices ordered by the topological
        # order, which makes rows easy to interpret next to that list.
        starts = np.stack([self.work_packages[node].get_start_samples() for node in order], axis=0)  # Matrix of all S_i.
        finishes = np.stack([self.work_packages[node].get_finish_samples() for node in order], axis=0)  # Matrix of all T_i.

        self.time_simulation_result = ProjectTimeSimulationResult(  # Package the sampled variables.
            starts=starts,  # S_i samples.
            finishes=finishes,  # T_i samples.
            completion_times=completion_times,  # T_project samples.
            critical_paths=critical_paths,  # Critical path samples C.
            topological_order=order,  # Row labels for starts and finishes.
            lag_samples_by_edge=lag_samples_by_edge,  # L_{j,i} samples kept outside duration D_i.
        )
        self.completion_time_samples = completion_times  # Store T_project directly on the project.
        return self.time_simulation_result

    def simulate_project_time_with_critical_activities(
        self,
    ) -> dict[str, NDArray[np.uint32] | NDArray[np.bool_] | list[int]]:
        """Calculate project length and all activities on at least one critical path.

        This slower variant keeps the full tied-predecessor information. For
        every Monte Carlo scenario, critical_activities[k, s] is true when the
        kth activity in topological_order lies on at least one longest path.
        """
        order = self.topological_order()  # T, a topological order of the DAG.
        if not order:
            raise ValueError("cannot simulate an empty project.")

        sample_count = self.calculate_default_sample_count()  # Number of Monte Carlo scenarios.
        self.simulate_all_work_packages()  # Draw Z_i, all schedule risks, and total D_i before timing.
        lag_samples_by_edge = self._simulate_dependency_lags(sample_count)  # Draw all L_{j,i} before timing.

        node_offsets = {node: offset for offset, node in enumerate(order)}  # Convert activity IDs to matrix rows.
        activity_count = len(order)  # N, the number of activities.
        starts = np.zeros((activity_count, sample_count), dtype=np.uint32)  # S_i samples.
        finishes = np.zeros((activity_count, sample_count), dtype=np.uint32)  # T_i samples.
        critical_by_node = np.zeros(  # B_i vectors for every i and scenario.
            (activity_count, activity_count, sample_count),
            dtype=np.bool_,
        )

        for node in order:
            node_offset = node_offsets[node]  # Row for activity i.
            duration_samples = self.work_packages[node].duration_samples  # Total sampled D_i.
            if duration_samples is None:
                raise RuntimeError(f"duration samples are missing for activity {node}.")
            predecessor_nodes = list(self.graph.predecessors(node))  # Pred(i).

            if not predecessor_nodes:
                starts[node_offset] = np.zeros(sample_count, dtype=np.uint32)  # S_i = 0.
                critical_by_node[node_offset, node_offset, :] = True  # B_i(i) = 1.
            else:
                predecessor_finish_lags = np.stack(  # Q_i = (T_j + L_{j,i})_{j in Pred(i)}.
                    [
                        finishes[node_offsets[predecessor]].astype(np.uint64)  # T_j.
                        + lag_samples_by_edge[(predecessor, node)].astype(np.uint64)  # L_{j,i}.
                        for predecessor in predecessor_nodes
                    ],
                    axis=0,  # Rows index predecessors; columns index samples.
                )
                if np.any(predecessor_finish_lags > UINT32_MAX):
                    raise OverflowError("start times exceed uint32 capacity.")

                starts[node_offset] = np.max(predecessor_finish_lags, axis=0).astype(np.uint32, copy=False)  # S_i=max Q_i.
                tied_predecessors = predecessor_finish_lags == starts[node_offset][np.newaxis, :]  # M_i per scenario.
                critical_by_node[node_offset, node_offset, :] = True  # e_i contribution to B_i.
                for predecessor_offset, predecessor in enumerate(predecessor_nodes):
                    sample_mask = tied_predecessors[predecessor_offset]  # Samples where predecessor is in M_i.
                    if np.any(sample_mask):
                        critical_by_node[node_offset, :, sample_mask] |= critical_by_node[
                            node_offsets[predecessor],
                            :,
                            sample_mask,
                        ]  # B_i <- B_i OR B_j for tied predecessor j.

            row_finishes = (  # T_i = S_i + D_i.
                starts[node_offset].astype(np.uint64)  # S_i, widened before addition.
                + duration_samples.astype(np.uint64)  # D_i, widened before addition.
            )
            if np.any(row_finishes > UINT32_MAX):
                raise OverflowError("project finish times exceed uint32 capacity.")
            finishes[node_offset] = row_finishes.astype(np.uint32, copy=False)  # Store sampled T_i.
            self.work_packages[node].set_timing_samples(  # Store sampled timing output on the activity itself.
                starts[node_offset],
                finishes[node_offset],
            )

        completion_times = np.max(finishes, axis=0)  # T_project = max_i T_i.
        terminal_ties = finishes == completion_times[np.newaxis, :]  # R = {i: T_i = T_project}.
        critical_activities = np.zeros((activity_count, sample_count), dtype=np.bool_)  # C per scenario.
        for node_offset in range(activity_count):
            sample_mask = terminal_ties[node_offset]  # Samples where i is in R.
            if np.any(sample_mask):
                critical_activities[:, sample_mask] |= critical_by_node[
                    node_offset,
                    :,
                    sample_mask,
                ].T  # C <- C OR B_i for tied project finish node i.

        self.completion_time_samples = completion_times.astype(np.uint32, copy=False)  # Store T_project on Project.
        self.critical_activity_samples = critical_activities  # Store C on Project for later notebook access.
        self.critical_activity_topological_order = order  # Store row labels for C.
        return {
            "completion_times": completion_times.astype(np.uint32, copy=False),  # T_project samples.
            "critical_activities": critical_activities,  # Boolean C vectors for all scenarios.
            "topological_order": order,  # Row labels for starts, finishes, and critical_activities.
        }

    def _ensure_duration_samples(self, sample_count: int) -> None:
        for work_package in self.work_packages.values():
            self._ensure_work_package_duration_samples(work_package, sample_count)  # Draw D_i when missing/stale.

    def _ensure_work_package_duration_samples(
        self,
        work_package: WorkPackage,
        sample_count: int,
    ) -> None:
        """Ensure one activity has a duration vector D_i with the common sample size."""
        samples = work_package.duration_samples  # Existing sample vector D_i, if any.
        if samples is None or len(samples) != sample_count:
            work_package.simulate_duration(sample_count, self.rng)  # Draw D_i when missing/stale.

    def _work_package_by_id(self, work_package_id: int) -> WorkPackage:
        """Return a validated work package by activity ID."""
        if not isinstance(work_package_id, Integral):
            raise TypeError("work_package_id must be an integer.")
        work_package_id = int(work_package_id)  # Normalize NumPy integer scalar labels.
        try:
            return self.work_packages[work_package_id]  # Select the activity i.
        except KeyError as exc:
            raise KeyError(f"unknown work package {work_package_id}.") from exc

    def _simulate_dependency_lags(
        self,
        sample_count: int,
    ) -> dict[tuple[int, int], NDArray[np.uint32]]:
        """Sample every dependency lag L_{j,i} without adding it to D_i."""
        lag_samples_by_edge: dict[tuple[int, int], NDArray[np.uint32]] = {}  # Maps edge (j, i) to L_{j,i}.
        for predecessor, successor, edge_data in self.graph.edges(data=True):
            lag_model = edge_data.get("lag_model", DeterministicLag(0))  # Old edges default to zero lag.
            samples = np.asarray(lag_model.sample(sample_count, self.rng))  # Draw the lag vector for this dependency.
            if samples.shape != (sample_count,):
                raise ValueError("lag samples must be a vector of length sample_count.")
            if np.any(samples < 0) or np.any(samples > UINT32_MAX):
                raise ValueError("lag samples must be nonnegative and fit in uint32.")
            if samples.dtype != np.uint32:
                samples = samples.astype(np.uint32, copy=False)  # Store L_{j,i} as nonnegative integer time.
            lag_samples_by_edge[(predecessor, successor)] = samples  # Keep lag separate from activity duration.
            self.work_packages[successor].set_lag_samples(predecessor, samples)  # Store incoming L_{j,i} on activity i.
        return lag_samples_by_edge

    def _normalize_complete_paths(
        self,
        paths: Sequence[Sequence[int]] | None,
    ) -> list[list[int]]:
        """Validate or construct the complete path set P."""
        self.topological_order()  # The exact path formula assumes a valid DAG schedule S.
        if paths is None:
            normalized_paths = self.complete_paths()  # Use the project DAG to construct P.
        else:
            normalized_paths = [list(path) for path in paths]  # Copy caller paths into mutable lists.

        if not normalized_paths:
            raise ValueError("at least one complete path is required.")

        known_nodes = set(self.work_packages)  # Activities with defined D_i laws.
        covered_nodes: set[int] = set()  # Activities represented in the supplied path set P.
        for path in normalized_paths:
            if not path:
                raise ValueError("complete paths cannot be empty.")
            unknown_nodes = sorted(set(path) - known_nodes)  # Path activities without D_i.
            if unknown_nodes:
                raise ValueError(f"path contains unknown activities: {unknown_nodes}.")
            if self.graph.in_degree(path[0]) != 0:
                raise ValueError("each complete path must start at a source activity.")
            if self.graph.out_degree(path[-1]) != 0:
                raise ValueError("each complete path must end at a terminal activity.")
            for predecessor, successor in zip(path, path[1:]):
                if not self.graph.has_edge(predecessor, successor):
                    raise ValueError("each complete path must follow dependency edges.")
            covered_nodes.update(path)

        missing_nodes = sorted(known_nodes - covered_nodes)  # Activities omitted from P.
        if missing_nodes:
            raise ValueError(f"activities are missing from the complete paths: {missing_nodes}.")

        return normalized_paths  # P = {p_1, ..., p_q}.

    def _shifted_poisson_parameters_by_node(self) -> dict[int, ShiftedPoissonDuration]:
        """Return (lambda_i, a_i) for every activity in this prototype model."""
        parameters: dict[int, ShiftedPoissonDuration] = {}  # Map i -> ShiftedPoi(lambda_i, a_i).
        for node, work_package in self.work_packages.items():
            duration_model = work_package.duration_model  # Candidate law for D_i.
            if not isinstance(duration_model, ShiftedPoissonDuration):
                raise TypeError(
                    "project_completion_cdf_shifted_poisson only supports "
                    "ShiftedPoissonDuration models."
                )
            parameters[node] = duration_model  # Store the validated shifted-Poisson law.
        return parameters


def _build_critical_paths(
    completion_nodes: NDArray[np.uint64],
    predecessors_by_node: dict[int, NDArray[np.uint64]],
) -> list[list[int]]:
    """Trace p_i backward from the final activity r for every sample."""
    critical_paths: list[list[int]] = []  # Collection of sampled critical paths C.
    for sample_index, node in enumerate(completion_nodes):
        path: list[int] = []  # Current path, built backward from r.
        current = int(node)  # Start with r = argmax_i T_i.
        while current != NO_PREDECESSOR:
            path.append(current)  # Add current activity to C.
            current = int(predecessors_by_node[current][sample_index])  # Move to p_current.
        path.reverse()  # Convert backward trace into chronological order.
        critical_paths.append(path)  # Store this sample's critical path.
    return critical_paths


def _coerce_lag_model(lag: int | LagModel) -> LagModel:
    """Return a lag model, converting plain integers to deterministic lag."""
    if isinstance(lag, Integral):
        return DeterministicLag(int(lag))  # User-facing shorthand for fixed L_{j,i}.
    if not hasattr(lag, "sample"):
        raise TypeError("lag must be an integer or lag model.")
    return lag  # Future random lag models can satisfy LagModel without project changes.


def _poisson_pmf(lambda_: float, k: int) -> np.longdouble:
    """Return P(Poi(lambda_) = k) using NumPy long-double arithmetic."""
    if k < 0:
        return PROBABILITY_DTYPE(0)  # A Poisson random variable has support N_0.
    if lambda_ == 0:
        return PROBABILITY_DTYPE(1) if k == 0 else PROBABILITY_DTYPE(0)  # Poi(0) is degenerate at zero.
    return _poisson_probability_vector(lambda_, k)[k]  # Select P(X=k) from the support vector 0, ..., k.


def _poisson_cdf(lambda_: float, x: int | float) -> np.longdouble:
    """Return P(Poi(lambda_) <= x), with value 0 for impossible x < 0."""
    upper = int(np.floor(x))  # Discrete CDF uses the largest integer <= x.
    if upper < 0:
        return PROBABILITY_DTYPE(0)  # Negative slack is outside the support N_0.
    if lambda_ == 0:
        return PROBABILITY_DTYPE(1)  # Poi(0) <= x for every x >= 0.

    probability = np.sum(  # F_X(upper)=sum_{k=0}^{upper} P(X=k).
        _poisson_probability_vector(lambda_, upper),
        dtype=PROBABILITY_DTYPE,
    )
    return min(max(probability, PROBABILITY_DTYPE(0)), PROBABILITY_DTYPE(1))  # Clamp floating-point roundoff.


def _poisson_probability_vector(lambda_: float, upper: int) -> NDArray[np.longdouble]:
    """Return P(Poi(lambda_)=k) for k=0, ..., upper.

    The recurrence p_k = p_{k-1} lambda / k makes the integer support explicit:
    no probability is allocated below k=0, and the caller decides the largest
    feasible k from the shifted-Poisson slack calculation.
    """
    if upper < 0:
        return np.asarray([], dtype=PROBABILITY_DTYPE)  # Empty support below zero.

    lambda_value = PROBABILITY_DTYPE(lambda_)  # Work in NumPy's widest floating type.
    probabilities = np.empty(upper + 1, dtype=PROBABILITY_DTYPE)  # Stores p_0, ..., p_upper.
    probabilities[0] = np.exp(-lambda_value, dtype=PROBABILITY_DTYPE)  # p_0 = exp(-lambda).
    if upper == 0:
        return probabilities  # Only k=0 is requested.

    for k in range(1, upper + 1):
        probabilities[k] = probabilities[k - 1] * lambda_value / PROBABILITY_DTYPE(k)  # p_k recurrence.
    return probabilities


def _system_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return _system_memory_bytes_without_psutil()


def _system_memory_bytes_without_psutil() -> int:
    try:
        import os

        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count)
    except (AttributeError, OSError, ValueError):
        # Conservative fallback for platforms without sysconf/psutil.
        return 8 * GIB
