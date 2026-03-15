"""Test load dynamic reservoir model."""

from collections.abc import Sequence
import pathlib
import pickle
from typing import cast
import numpy as np
import pandas as pd

from georead._data_directory import DataType, ValueType
from georead._load_utils import load


def test_load():
    """Test loading Egg reservoir model."""
    egg_model_path = (
        pathlib.Path(__file__).parent / 'data' / 'egg' / 'Egg_Model_ECL.DATA'
    )
    data = load(egg_model_path)
    with open(
        pathlib.Path(__file__).parent / 'data' / 'egg_loaded_data.pickle', 'rb'
    ) as f:
        data_expected = pickle.load(f)  # pyright: ignore[reportAny]
        data_expected = cast(DataType, data_expected)
    validate_data(data, data_expected)


def validate_data(data: DataType, data_expected: DataType):
    for section in data_expected:
        for r, e in zip(data[section], data_expected[section], strict=True):
            assert r[0] == e[0]
            if not isinstance(e[1], tuple | list):
                expected_res = [e[1]]
                res = [r[1]]
            else:
                expected_res = e[1]
                res = r[1]
            expected_res = cast(Sequence[ValueType], expected_res)
            res = cast(Sequence[ValueType], res)
            for r, e in zip(res, expected_res):
                if isinstance(e, np.ndarray):
                    np.testing.assert_equal(r, e)
                elif isinstance(e, pd.DataFrame):
                    if not isinstance(r, pd.DataFrame):
                        raise ValueError('`r` should be of type `pandas.DataFrame`.')
                    pd.testing.assert_frame_equal(r, e)
                else:
                    assert res == expected_res
