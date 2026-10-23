
import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from src.features import extract_signal_features


def test_basic_signal_statistics():
    signal = np.array([1.0, 2.0, 3.0])

    result = extract_signal_features(signal)

    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(np.std(signal))
    assert result["rms"] == pytest.approx(
        np.sqrt(np.mean(signal ** 2))
    )
    assert result["minimum"] == pytest.approx(1.0)
    assert result["maximum"] == pytest.approx(3.0)
    assert result["peak_to_peak"] == pytest.approx(2.0)


def test_constant_signal():
    result = extract_signal_features([5.0, 5.0, 5.0])

    assert result["mean"] == pytest.approx(5.0)
    assert result["std"] == pytest.approx(0.0)
    assert result["rms"] == pytest.approx(5.0)
    assert result["peak_to_peak"] == pytest.approx(0.0)


def test_empty_signal_is_rejected():
    with pytest.raises(ValueError):
        extract_signal_features([])


def test_non_finite_signal_is_rejected():
    with pytest.raises(ValueError):
        extract_signal_features([1.0, np.nan, 3.0])


def test_signal_with_infinity_is_rejected():
    with pytest.raises(ValueError):
        extract_signal_features([1.0, np.inf, 3.0])


def test_signal_is_flattened():
    result = extract_signal_features(
        np.array([[1.0, 2.0], [3.0, 4.0]])
    )

    assert result["mean"] == pytest.approx(2.5)
    assert result["minimum"] == pytest.approx(1.0)
    assert result["maximum"] == pytest.approx(4.0)
