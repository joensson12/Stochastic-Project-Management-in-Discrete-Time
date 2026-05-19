import numpy as np
import pytest

from spm import ShiftedBinomialDuration, ShiftedPoissonDuration


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


def test_shifted_binomial_samples_are_uint32_and_bounded() -> None:
    model = ShiftedBinomialDuration(minimum=2, maximum=8, most_likely=5)
    samples = model.sample(1_000, np.random.default_rng(123))

    assert samples.dtype == np.uint32
    assert len(samples) == 1_000
    assert np.all(samples >= 2)
    assert np.all(samples <= 8)


def test_shifted_binomial_uses_midpoint_of_modal_p_interval() -> None:
    model = ShiftedBinomialDuration(minimum=1, maximum=6, most_likely=4)

    assert model.trial_count == 5
    assert model.p == pytest.approx(7 / 12)
    assert model.expected_value == pytest.approx(1 + 5 * 7 / 12)
    assert model.modes == (4,)


def test_shifted_binomial_handles_deterministic_interval() -> None:
    model = ShiftedBinomialDuration(minimum=4, maximum=4, most_likely=4)
    samples = model.sample(10, np.random.default_rng(123))

    assert model.trial_count == 0
    assert model.p == 0.0
    assert model.expected_value == 4.0
    assert model.modes == (4,)
    np.testing.assert_array_equal(samples, np.full(10, 4, dtype=np.uint32))


def test_shifted_binomial_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="minimum"):
        ShiftedBinomialDuration(minimum=-1, maximum=4, most_likely=2)

    with pytest.raises(ValueError, match="maximum"):
        ShiftedBinomialDuration(minimum=5, maximum=4, most_likely=5)

    with pytest.raises(ValueError, match="most_likely"):
        ShiftedBinomialDuration(minimum=2, maximum=8, most_likely=9)
