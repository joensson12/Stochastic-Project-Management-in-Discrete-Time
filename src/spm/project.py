"""Project-network simulation for earliest starts, finishes, and critical paths."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, floor, lgamma, log
from typing import Callable, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from spm.distributions import DurationDistribution, ShiftedPoissonDuration, UINT32_MAX
from spm.work_package import WorkPackage


BYTES_PER_UINT32 = np.dtype(np.uint32).itemsize  # One D_i sample is stored as one uint32.
GIB = 1024**3  # Memory budgets are expressed in binary gigabytes.

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
        if sample_count is not None and sample_count <= 0:
            raise ValueError("sample_count must be positive when provided.")
        if max_memory_bytes is not None and max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive when provided.")

        self.sample_count = sample_count  # Common Monte Carlo dimension n for every D_i.
        self.max_memory_bytes = max_memory_bytes  # Optional cap on the automatic memory budget.
        self._memory_provider = memory_provider or _system_memory_bytes  # Source for total RAM.
        self.rng = np.random.default_rng(rng_seed)  # Random generator used for all sampled variables.
        self.work_packages: dict[int, WorkPackage] = {}  # The set W, keyed by activity ID i.
        self.graph = nx.DiGraph()  # The deterministic schedule graph S.
        self.time_simulation_result: ProjectTimeSimulationResult | None = None  # Last project-time sample.

    def add_work_package(
        self,
        work_package_id: int,
        duration_model: DurationDistribution,
    ) -> None:
        """Add activity i with its duration distribution D_i."""
        if not isinstance(work_package_id, int):
            raise TypeError("work_package_id must be an integer.")
        if work_package_id <= 0:
            raise ValueError("work_package_id must be positive.")
        if work_package_id in self.work_packages:
            raise ValueError(f"work package {work_package_id} already exists.")

        self.work_packages[work_package_id] = WorkPackage(  # Register activity i in W.
            work_package_id=work_package_id,  # The integer label i.
            duration_model=duration_model,  # The probability law of D_i.
        )
        self.graph.add_node(work_package_id)  # Add i as a node in the schedule graph S.

    def add_dependency(self, predecessor: int, successor: int) -> None:
        """Add predecessor -> successor, meaning predecessor in Pred(successor).

        This first prototype implements the zero-lag case L_{i,j} = 0.
        """
        if not isinstance(predecessor, int) or not isinstance(successor, int):
            raise TypeError("dependencies must use integer work package IDs.")
        if predecessor <= 0 or successor <= 0:
            raise ValueError("dependencies must use positive work package IDs.")
        self.graph.add_edge(predecessor, successor)  # Encode predecessor in Pred(successor).

    def calculate_default_sample_count(self) -> int:
        """Choose a common Monte Carlo sample size for every D_i.

        Each work package stores one uint32 vector of duration samples. The
        automatic sample count is therefore the memory budget divided equally
        across the work packages.
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
            BYTES_PER_UINT32 * len(self.work_packages)  # Four bytes for each activity sample D_i.
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
        durations are interpreted as independent.
        """
        if not isinstance(t, int):  # The report works in discrete time.
            raise TypeError("t must be an integer threshold.")
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
        remaining_slacks = [t - path_minimum for path_minimum in path_minimums]  # b_theta(t)=t-A_theta.
        if any(slack < 0 for slack in remaining_slacks):
            return 0.0  # At least one path exceeds t even at its deterministic minimum.

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

        probability = 0.0  # Accumulator P in the algorithm.
        summation_ranges = [range(upper_bound + 1) for upper_bound in upper_bounds]  # Product domain for z_S.
        for shared_values in product(*summation_ranges):  # Enumerate (z_S)_{S in G}.
            shared_by_group = dict(zip(shared_groups, shared_values, strict=True))  # Attach each z_S to S.
            term_probability = 1.0  # P_z before adding it into P.

            for incidence, z_value in shared_by_group.items():
                term_probability *= _poisson_pmf(group_lambdas[incidence], z_value)  # P(Z_S=z_S).

            for theta in range(path_count):
                shared_sum = sum(  # Sum of z_S over shared groups that affect path theta.
                    z_value
                    for incidence, z_value in shared_by_group.items()
                    if theta in incidence
                )
                residual_slack = remaining_slacks[theta] - shared_sum  # r_theta in the algorithm.
                term_probability *= _poisson_cdf(
                    singleton_lambdas[theta],
                    residual_slack,
                )  # F_{Poi(Lambda_{theta})}(r_theta).

            probability += term_probability  # P <- P + P_z.

        return min(max(probability, 0.0), 1.0)  # Clamp tiny floating-point roundoff outside [0,1].

    def simulate_work_package(self, work_package_id: int) -> NDArray[np.uint32]:
        """Simulate and store the duration vector D_i for one activity."""
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

    def simulate_project_time(self) -> ProjectTimeSimulationResult:
        """Apply the longest-path recursion to all Monte Carlo samples.

        This mirrors the report algorithm:
        S_i = 0 and T_i = D_i when Pred(i) is empty.
        Otherwise S_i = max_{j in Pred(i)} T_j and T_i = S_i + D_i.

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

        # These dictionaries are the sampled versions of the mathematical
        # vectors S_i, T_i, and p_i. Each value has one entry per Monte Carlo
        # realization.
        starts_by_node: dict[int, NDArray[np.uint32]] = {}  # Stores sampled S_i vectors.
        finishes_by_node: dict[int, NDArray[np.uint32]] = {}  # Stores sampled T_i vectors.
        predecessors_by_node: dict[int, NDArray[np.uint64]] = {}  # Stores sampled p_i vectors.

        for node in order:
            duration_samples = self.work_packages[node].duration_samples  # The sampled D_i vector.
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
                starts_by_node[node] = np.zeros(sample_count, dtype=np.uint32)  # S_i = 0.
            else:
                # Temporarily form the matrix (T_j)_{j in Pred(i)} only for the
                # local argmax that defines S_i and p_i.
                pred_finishes = np.stack(  # Matrix with rows T_j for j in Pred(i).
                    [
                        finishes_by_node[predecessor]  # Previously computed T_j.
                        for predecessor in predecessor_nodes
                    ],
                    axis=0,  # Rows index predecessors; columns index samples.
                )
                selected_pred_offsets = np.argmax(pred_finishes, axis=0)  # argmax_j T_j per sample.
                starts_by_node[node] = np.take_along_axis(  # S_i = max_{j in Pred(i)} T_j.
                    pred_finishes,  # Candidate predecessor finish times.
                    selected_pred_offsets[np.newaxis, :],  # Winning predecessor row per sample.
                    axis=0,  # Max was taken over predecessor rows.
                )[0]  # Collapse the single selected row into a vector.
                predecessors_by_node[node] = np.asarray(  # p_i = selected predecessor m.
                    predecessor_nodes,  # Candidate predecessor activity IDs.
                    dtype=np.uint64,  # Store IDs as unsigned integers.
                )[
                    selected_pred_offsets  # Convert argmax offsets into actual activity IDs.
                ]

            # Compute T_i = S_i + D_i in uint64 first. This catches overflow
            # before storing back into the compact uint32 representation.
            row_finishes = (
                starts_by_node[node].astype(np.uint64)  # S_i, widened before addition.
                + duration_samples.astype(np.uint64)  # D_i, widened before addition.
            )
            if np.any(row_finishes > UINT32_MAX):
                raise OverflowError("project finish times exceed uint32 capacity.")
            finishes_by_node[node] = row_finishes.astype(np.uint32, copy=False)  # T_i = S_i + D_i.

        # The project completion time is max_i T_i. We stack all T_i vectors
        # only here, because this is the single global argmax in the algorithm.
        finishes = np.stack([finishes_by_node[node] for node in order], axis=0)  # Matrix of all T_i.
        completion_row_indices = np.argmax(finishes, axis=0)  # r = argmax_i T_i per sample.
        completion_nodes = np.asarray(order, dtype=np.uint64)[completion_row_indices]  # Activity r.
        completion_times = np.take_along_axis(  # T_project = T_r.
            finishes,  # Candidate finish times T_i.
            completion_row_indices[np.newaxis, :],  # Winning activity row r per sample.
            axis=0,  # Select over activities.
        )[0].astype(np.uint32, copy=False)  # Store the project duration vector.
        critical_paths = _build_critical_paths(  # Reconstruct C by following p_i backward.
            completion_nodes,  # Final activity r for each sample.
            predecessors_by_node,  # The sampled predecessor maps p_i.
        )

        # The public result is returned as matrices ordered by the topological
        # order, which makes rows easy to interpret next to that list.
        starts = np.stack([starts_by_node[node] for node in order], axis=0)  # Matrix of all S_i.

        self.time_simulation_result = ProjectTimeSimulationResult(  # Package the sampled variables.
            starts=starts,  # S_i samples.
            finishes=finishes,  # T_i samples.
            completion_times=completion_times,  # T_project samples.
            critical_paths=critical_paths,  # Critical path samples C.
            topological_order=order,  # Row labels for starts and finishes.
        )
        return self.time_simulation_result

    def _ensure_duration_samples(self, sample_count: int) -> None:
        for work_package in self.work_packages.values():
            samples = work_package.duration_samples  # Existing sample vector D_i, if any.
            if samples is None or len(samples) != sample_count:
                work_package.simulate_duration(sample_count, self.rng)  # Draw D_i when missing/stale.

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


def _poisson_pmf(lambda_: float, k: int) -> float:
    """Return P(Poi(lambda_) = k) for the prototype exact CDF calculation."""
    if k < 0:
        return 0.0  # A Poisson random variable has support N_0.
    if lambda_ == 0:
        return 1.0 if k == 0 else 0.0  # Poi(0) is degenerate at zero.
    return exp(-lambda_ + k * log(lambda_) - lgamma(k + 1))  # Stable log-PMF formula.


def _poisson_cdf(lambda_: float, x: int | float) -> float:
    """Return P(Poi(lambda_) <= x), with value 0 for x < 0."""
    upper = floor(x)  # Discrete CDF uses the largest integer <= x.
    if upper < 0:
        return 0.0  # Convention from the theorem.
    if lambda_ == 0:
        return 1.0  # Poi(0) <= x for every x >= 0.

    log_terms = [  # Log probabilities for k=0, ..., upper.
        -lambda_ + k * log(lambda_) - lgamma(k + 1)
        for k in range(upper + 1)
    ]
    max_log_term = max(log_terms)  # Centering term for log-sum-exp.
    probability = exp(max_log_term) * sum(  # Sum probabilities without avoidable underflow.
        exp(log_term - max_log_term)
        for log_term in log_terms
    )
    return min(max(probability, 0.0), 1.0)  # Clamp floating-point roundoff.


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
