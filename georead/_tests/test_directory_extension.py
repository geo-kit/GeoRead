"""Test loading model with updating keyword directory."""

import pathlib
import pickle
from typing import cast
from georead._data_directory import (
    SECTIONS,
    DataTypes,
    KeywordSpecification,
    StatementSpecification,
)
from georead import DataType, load
from .test_load_model import validate_data


def test_update_keyword():
    """Load Egg model with renaming columns for DIMENS keyword."""
    keyword_update: dict[str, KeywordSpecification | None] = {
        'DIMENS': KeywordSpecification(
            'DIMENS',
            DataTypes.SINGLE_STATEMENT,
            StatementSpecification(
                ('NX_TEST', 'NY_TEST', 'NZ_TEST'), ('int', 'int', 'int')
            ),
            (SECTIONS.RUNSPEC,),
        )
    }
    egg_model_path = (
        pathlib.Path(__file__).parent / 'data' / 'egg' / 'Egg_Model_ECL.DATA'
    )
    data = load(egg_model_path, directory_extension=keyword_update)
    with open(
        pathlib.Path(__file__).parent
        / 'data'
        / 'egg_loaded_data_altered_dimens.pickle',
        'rb',
    ) as f:
        data_expected = pickle.load(f)  # pyright: ignore[reportAny]
        data_expected = cast(DataType, data_expected)
    validate_data(data, data_expected)


def test_new_keyword():
    """Load Egg model with DIMENS keyword renamed."""
    keyword_update: dict[str, KeywordSpecification | None] = {
        'DIMENS_TEST': KeywordSpecification(
            'DIMENS',
            DataTypes.SINGLE_STATEMENT,
            StatementSpecification(('NX', 'NY', 'NZ'), ('int', 'int', 'int')),
            (SECTIONS.RUNSPEC,),
        )
    }
    egg_model_path = (
        pathlib.Path(__file__).parent
        / 'data'
        / 'egg'
        / '_test_altered_keyword_Egg_Model_ECL.DATA'
    )
    data = load(egg_model_path, directory_extension=keyword_update)
    with open(
        pathlib.Path(__file__).parent
        / 'data'
        / 'egg_loaded_data_altered_keyword.pickle',
        'rb',
    ) as f:
        data_expected = pickle.load(f)  # pyright: ignore[reportAny]
        data_expected = cast(DataType, data_expected)
    validate_data(data, data_expected)
