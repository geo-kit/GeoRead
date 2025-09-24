"""Test dynamic specfication generation."""

from collections.abc import Sequence
from typing import cast
import pandas as pd
import pytest
from resdp._data_directory import (
    SECTIONS,
    DTypeString,
    DataType,
    DataTypes,
    StatementSpecification,
    get_dynamic_keyword_specification,
    KeywordSpecification,
    TableSpecification,
    _get_gptable_number,  # pyright: ignore[reportPrivateUsage]
    _get_eql_regions_number,  # pyright: ignore[reportPrivateUsage]
)

TEST_DATA = (
    (
        (
            'ZMFVD',
            {
                'RUNSPEC': (('COMPS', pd.DataFrame([[5]], columns=['N'])),),
            },
        ),
        KeywordSpecification(
            'ZMFVD',
            DataTypes.TABLE_SET,
            TableSpecification(
                ['DEPTH'] + [f'C{i}' for i in range(1, 6)],
                [0],
                cast(Sequence[DTypeString], ['float'] * 6),
                number=_get_eql_regions_number,
            ),
            (SECTIONS.PROPS,),
        ),
    ),
    (
        (
            'ZMFVD',
            {
                'RUNSPEC': (),
            },
        ),
        ValueError(),
    ),
    (
        ('COMPVD', {'PROPS': (('NCOMPS', pd.DataFrame([[5]], columns=['N'])),)}),
        KeywordSpecification(
            'COMPVD',
            DataTypes.TABLE_SET,
            TableSpecification(
                ['DEPTH'] + [f'Z{i}' for i in range(1, 6)] + ['LIQUID_FLAG', 'P_SAT'],
                [0],
                cast(Sequence[DTypeString], ['float'] * (6) + ['int', 'float']),
                number=_get_eql_regions_number,
            ),
            (SECTIONS.PROPS,),
        ),
    ),
    (
        ('GPTABLEN', {'PROPS': (('NCOMPS', pd.DataFrame([[8]], columns=['N'])),)}),
        KeywordSpecification(
            'GPTABLEN',
            DataTypes.TABLE_SET,
            TableSpecification(
                ['C_HEAVY']
                + [f'OIL_RECOVERY_FRACTION{i}' for i in range(1, 9)]
                + [f'NGL_RECOVERY_FRACTION{i}' for i in range(1, 9)],
                domain=[0],
                header=StatementSpecification(
                    ['GPTABLE_NUM', 'HEAVY_C1', 'HEAVY_CLAST'],
                    cast(Sequence[DTypeString], ['int'] * 3),
                ),
                number=_get_gptable_number,
            ),
            (SECTIONS.SOLUTION, SECTIONS.SCHEDULE),
        ),
    ),
)


@pytest.mark.parametrize(
    'kw, data, expected', [(kw, data, expected) for ((kw, data), expected) in TEST_DATA]
)
def test_dynamic_specification(kw: str, data: DataType, expected: KeywordSpecification):
    """Test dynamic specfication generation."""
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            res = get_dynamic_keyword_specification(kw, data)
    else:
        res = get_dynamic_keyword_specification(kw, data)
        assert res == expected
