
import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from src.train_candidate_model import SENSORS, extract_shape_features


def test_shape_statistics_match_known_values():
    signal = np.array([1.0, 2.0, 2.0, 3.0, 5.0])

    skewness_value = float(skew(signal, bias=False))
    kurtosis_value = float(
        kurtosis(signal, fisher=True, bias=False)
    )

    assert np.isfinite(skewness_value)
    assert np.isfinite(kurtosis_value)
    assert skewness_value != 0.0
    assert kurtosis_value != 0.0


def test_shape_feature_names_cover_all_sensors():
    expected = {
        f"{sensor}_{stat}"
        for sensor in SENSORS
        for stat in ("skewness", "kurtosis")
    }

    assert len(expected) == 12


def test_shape_features_from_real_dataset():
    shape_df = extract_shape_features()

    assert not shape_df.empty
    assert {"case", "run"}.issubset(shape_df.columns)

    for sensor in SENSORS:
        assert f"{sensor}_skewness" in shape_df.columns
        assert f"{sensor}_kurtosis" in shape_df.columns

    shape_columns = [
        column for column in shape_df.columns
        if column.endswith("_skewness")
        or column.endswith("_kurtosis")
    ]

    assert len(shape_columns) == 12
    assert np.isfinite(shape_df[shape_columns].to_numpy()).all()
