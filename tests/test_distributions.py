import numpy as np
import pytest

from spm import ShiftedPoissonDuration


def test_shifted_poisson_samples_are_uint32_and_shifted() -> None:
    model = ShiftedPoissonDuration(lambda_=3.5, a=2)
    samples = model.sample(100, np.random.default_rng(123))

    assert samples.dtype == np.uint32
    assert len(samples) == 100
    assert np.all(samples >= 2)


def test_shifted_poisson_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="lambda_"):
        ShiftedPoissonDuration(lambda_=0, a=0)

    with pytest.raises(ValueError, match="nonnegative"):
        ShiftedPoissonDuration(lambda_=1, a=-1)


def test_shifted_poisson_mean_modes_and_independent_sum() -> None:
    non_integer_model = ShiftedPoissonDuration(lambda_=3.5, a=2)
    integer_model = ShiftedPoissonDuration(lambda_=3.0, a=2)

    assert non_integer_model.expected_value == 5.5
    assert non_integer_model.modes == (5,)
    assert integer_model.modes == (4, 5)

    summed = ShiftedPoissonDuration.from_independent_sum(
        [non_integer_model, integer_model]
    )

    assert summed.lambda_ == 6.5
    assert summed.a == 4
