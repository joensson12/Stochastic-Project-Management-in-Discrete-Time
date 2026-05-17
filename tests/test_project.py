from math import exp, lgamma, log

import numpy as np
import pytest

from spm import Project, ShiftedPoissonDuration, WorkPackage


class FixedDuration:
    def __init__(self, samples: list[int]) -> None:
        self.samples = np.asarray(samples, dtype=np.uint32)

    def sample(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        assert sample_count == len(self.samples)
        return self.samples


def poisson_cdf(lambda_: float, x: int) -> float:
    if x < 0:
        return 0.0
    if lambda_ == 0:
        return 1.0
    return sum(
        exp(-lambda_ + k * log(lambda_) - lgamma(k + 1))
        for k in range(x + 1)
    )


def test_explicit_sample_count_is_used_directly() -> None:
    project = Project(sample_count=7, memory_provider=lambda: 16 * 1024**3)

    assert project.calculate_default_sample_count() == 7


def test_default_sample_count_uses_memory_budget_per_work_package() -> None:
    project = Project(memory_provider=lambda: 16 * 1024**3)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1, a=0))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=1, a=0))

    assert project.calculate_default_sample_count() == (10 * 1024**3) // (4 * 2)


def test_default_sample_count_uses_system_ram_minus_two_gib_when_smaller() -> None:
    project = Project(memory_provider=lambda: 6 * 1024**3)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1, a=0))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=1, a=0))

    assert project.calculate_default_sample_count() == (4 * 1024**3) // (4 * 2)


def test_serial_network_completion_and_critical_path() -> None:
    project = Project(sample_count=3, rng_seed=123)
    project.add_work_package(1, FixedDuration([1, 2, 3]))
    project.add_work_package(2, FixedDuration([4, 5, 6]))
    project.add_work_package(3, FixedDuration([7, 8, 9]))
    project.add_dependency(1, 2)
    project.add_dependency(2, 3)

    result = project.simulate_project_time()

    np.testing.assert_array_equal(
        result.starts,
        np.asarray(
            [
                [0, 0, 0],
                [1, 2, 3],
                [5, 7, 9],
            ],
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(
        result.finishes,
        np.asarray(
            [
                [1, 2, 3],
                [5, 7, 9],
                [12, 15, 18],
            ],
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(
        result.completion_times,
        np.asarray([12, 15, 18], dtype=np.uint32),
    )
    assert result.critical_paths == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]


def test_parallel_network_uses_latest_predecessor_per_sample() -> None:
    project = Project(sample_count=3, rng_seed=123)
    project.add_work_package(1, FixedDuration([10, 1, 5]))
    project.add_work_package(2, FixedDuration([2, 9, 5]))
    project.add_work_package(3, FixedDuration([1, 1, 1]))
    project.add_dependency(1, 3)
    project.add_dependency(2, 3)

    result = project.simulate_project_time()
    row_for_3 = result.topological_order.index(3)

    np.testing.assert_array_equal(
        result.starts[row_for_3],
        np.asarray([10, 9, 5], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        result.completion_times,
        np.asarray([11, 10, 6], dtype=np.uint32),
    )
    assert result.critical_paths == [[1, 3], [2, 3], [1, 3]]


def test_complete_paths_returns_source_to_sink_paths() -> None:
    project = Project(sample_count=1)
    for node in [1, 2, 3, 4]:
        project.add_work_package(node, ShiftedPoissonDuration(lambda_=1, a=0))
    project.add_dependency(1, 2)
    project.add_dependency(1, 3)
    project.add_dependency(2, 4)
    project.add_dependency(3, 4)

    assert sorted(project.complete_paths()) == [[1, 2, 4], [1, 3, 4]]


def test_shifted_poisson_project_cdf_for_serial_path_uses_closure() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1.0, a=2))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=3.0, a=1))
    project.add_dependency(1, 2)

    probability = project.project_completion_cdf_shifted_poisson(t=5)

    assert probability == pytest.approx(poisson_cdf(4.0, 2))


def test_shifted_poisson_project_cdf_for_parallel_independent_paths() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1.0, a=0))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=2.0, a=1))

    probability = project.project_completion_cdf_shifted_poisson(t=2)

    assert probability == pytest.approx(
        poisson_cdf(1.0, 2) * poisson_cdf(2.0, 1)
    )


def test_shifted_poisson_project_cdf_is_zero_below_deterministic_path_minimum() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1.0, a=2))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=1.0, a=3))
    project.add_work_package(3, ShiftedPoissonDuration(lambda_=1.0, a=7))
    project.add_dependency(1, 2)

    assert project.project_completion_cdf_shifted_poisson(t=6) == 0.0
    assert project.project_completion_cdf_shifted_poisson(t=7) > 0.0


def test_shifted_poisson_project_cdf_groups_shared_path_incidence() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=0.5, a=0))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=1.0, a=0))
    project.add_work_package(3, ShiftedPoissonDuration(lambda_=2.0, a=0))
    project.add_work_package(4, ShiftedPoissonDuration(lambda_=0.7, a=0))
    project.add_dependency(1, 2)
    project.add_dependency(1, 3)
    project.add_dependency(2, 4)
    project.add_dependency(3, 4)

    probability = project.project_completion_cdf_shifted_poisson(t=3)
    expected = sum(
        exp(-1.2 + z * log(1.2) - lgamma(z + 1))
        * poisson_cdf(1.0, 3 - z)
        * poisson_cdf(2.0, 3 - z)
        for z in range(4)
    )

    assert probability == pytest.approx(expected)


def test_shifted_poisson_project_cdf_rejects_general_duration_models() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, FixedDuration([1]))

    with pytest.raises(TypeError, match="ShiftedPoissonDuration"):
        project.project_completion_cdf_shifted_poisson(t=1)


def test_shifted_poisson_project_cdf_rejects_invalid_manual_paths() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1.0, a=0))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=1.0, a=0))
    project.add_dependency(1, 2)

    with pytest.raises(ValueError, match="source"):
        project.project_completion_cdf_shifted_poisson(t=2, paths=[[2]])


def test_missing_work_package_node_raises_validation_error() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1, a=0))
    project.add_dependency(1, 2)

    with pytest.raises(ValueError, match="without work packages"):
        project.topological_order()


def test_cycle_raises_validation_error() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, ShiftedPoissonDuration(lambda_=1, a=0))
    project.add_work_package(2, ShiftedPoissonDuration(lambda_=1, a=0))
    project.add_dependency(1, 2)
    project.add_dependency(2, 1)

    with pytest.raises(ValueError, match="DAG"):
        project.topological_order()


def test_work_package_rejects_nonpositive_id() -> None:
    with pytest.raises(ValueError, match="positive"):
        WorkPackage(-1, ShiftedPoissonDuration(lambda_=1, a=0))

    with pytest.raises(ValueError, match="positive"):
        WorkPackage(0, ShiftedPoissonDuration(lambda_=1, a=0))


def test_project_time_overflow_raises_error() -> None:
    project = Project(sample_count=1)
    project.add_work_package(1, FixedDuration([np.iinfo(np.uint32).max]))
    project.add_work_package(2, FixedDuration([1]))
    project.add_dependency(1, 2)

    with pytest.raises(OverflowError, match="uint32"):
        project.simulate_project_time()
