"""Classes for keyword specifications and directory itself."""

from collections.abc import Sequence, Callable
from enum import Enum, auto
import itertools
from typing import Final, Literal, NamedTuple, cast

import pandas as pd
import numpy.typing as npt
import numpy as np


INT_NAN = -99999999

FLUID_KEYWORDS = ('OIL', 'GAS', 'WATER', 'DISGAS')
ORTHOGONAL_GRID_KEYWORDS = ('DX', 'DY', 'DZ', 'TOPS', 'ACTNUM', 'DXV', 'DYV', 'DZV')
ROCK_GRID_KEYWORDS = ('PORO', 'PERMX', 'PERMY', 'PERMZ', 'NTG')
TABLES_KEYWORDS = ('DENSITY', 'PVCDO', 'PVTW', 'ROCK', 'SWOF')
FIELD_SUMMARY_KEYWORDS = (
    'FOPR',
    'FWPR',
    'FWIR',
    'FHPV',
    'FMWPR',
    'FMWPT',
    'FMWPA',
    'FGIP',
    'FGOR',
    'FGORH',
    'FGIP',
    'FGPR',
    'FGPRH',
    'FGPT',
    'FGPTH',
    'FLPR',
    'FLPRH',
    'FLPT',
    'FLPTH',
    'FOIP',
    'FOPRH',
    'FOPTH',
    'FPR',
    'FWCT',
    'FWCTH',
    'FWIRH',
    'FWITH',
    'FWPRH',
    'FWPTH',
    'FVPR',
    'FVPT',
    'FTPRAQW',
    'FTPRFW',
    'FTPRMW',
    'FTPRFO',
    'FTPRMO',
    'FOSRC',
    'FHLR',
    'FHLT',
    'FSTPR',
    'FSTPT',
)
WELL_SUMMARY_KEYWORDS = (
    'WOPR',
    'WWPR',
    'WWIR',
    'WLPR',
    'WBHP',
    'WBP9',
    'WBP',
    'WGIR',
    'WGIRH',
    'WGIT',
    'WGITH',
    'WGOR',
    'WGORH',
    'WGPR',
    'WGPRH',
    'WGPT',
    'WGPTH',
    'WLPRH',
    'WLPT',
    'WLPTH',
    'WOPRH',
    'WOPTH',
    'WWCT',
    'WWCTH',
    'WWIRH',
    'WWIT',
    'WWITH',
    'WWPRH',
    'WWPT',
    'WWPTH',
    'WTPRAQW',
    'WTRFW',
    'WTPRMV',
    'WPIW',
    'WPIO',
    'WPIG',
    'WWPP',
    'WOPP',
    'WGPP',
    'WOPT',
    'WTPRFW',
    'WTPRMW',
    'WSTPR',
    'WSTPT',
    'WTEMP',
)
REGION_SUMMARY_KEYWORDS = ('RGIPL', 'ROIP', 'RWIP', 'RGIP', 'RPR', 'ROE')
TOTAL_SUMMARY_KEYWORDS = ('FOPT', 'FWPT', 'FWIT')
SCHEDULE_KEYWORDS = ('WELSPECS', 'COMPDAT', 'WCONPROD', 'WCONINJE')
DIMS_KEYWORDS = (
    'TABDIMS',
    'EQLDIMS',
    'REGDIMS',
    'WELLDIMS',
    'VFPPDIMS',
    'VFPIDIMS',
    'AQUDIMS',
)
MODEL_SUMMARY_KEYWORDS = (
    'TIMESTEP',
    'ELAPSED',
    'TCPU',
    'MLINEARS',
    'MSUMLINS',
    'MSUMNEWT',
    'NEWTON',
    'MLINEARS',
    'MSUMLINS',
    'MSUMNEWT',
    'NEWTON',
    'NLINEARS',
    'STEPTYPE',
)
GROUP_SUMMARY_KEYWORDS = (
    'GTPRAQW',
    'GTPRFW',
    'GTPRMW',
    'GGOR',
    'GGPR',
    'GGPRH',
    'GGPT',
    'GGPTH',
    'GLPR',
    'GLPRH',
    'GLPT',
    'GLPTH',
    'GOPR',
    'GOPRH',
    'GOPT',
    'GOPTH',
    'GWCT',
    'GWCTH',
    'GWPR',
    'GWPRH',
    'GWPT',
    'GWPTH',
    'GGORH',
)
REGIONS_SUMMARY_KEYWORDS = ()
_ATM_TO_PSI = 14.69

NpFloatingArray = (
    npt.NDArray[np.float16]
    | npt.NDArray[np.float32]
    | npt.NDArray[np.float64]
    | npt.NDArray[np.float128]
)
NpIntegerArray = (
    npt.NDArray[np.int8]
    | npt.NDArray[np.int16]
    | npt.NDArray[np.int32]
    | npt.NDArray[np.int64]
)
NpArray = NpIntegerArray | NpFloatingArray | npt.NDArray[np.bool_]


class ArrayWithUnits(NamedTuple):
    """
    Array with units.

    Attributes
    ----------
    units : str
        Data units.
    data : npt.NDArray[np.integer | np.floating | np.bool_]
        Array data.

    """

    units: str
    data: NpArray


RecordValueType = list[pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | NpArray]
ValueType = (
    Sequence[pd.DataFrame]
    | Sequence[tuple[pd.DataFrame, pd.DataFrame]]
    | tuple[str, pd.DataFrame]
    | pd.DataFrame
    | str
    | NpArray
    | None
    | dict[str, str | None]
    | ArrayWithUnits
    | Sequence[str]
    | Sequence[pd.Timestamp]
    | pd.Timestamp
    | RecordValueType
    | dict[str, pd.DataFrame]
)
DataType = dict[str, list[tuple[str, ValueType]]]
IntType = int | np.signedinteger
DTypeString = Literal['int', 'float', 'text']


class DataTypes(Enum):
    """
    Types of data corresponding to specific keywords.

    Attributes
    ----------
    STRING : Single string or date.
    SINGLE_STATEMENT : Single statement.
    STATEMENT_LIST : List of statements.
    ARRAY : Array.
    TABLE_SET : Set of tables.
    PARAMETERS : Parameters.
    OBJECT_LIST : List of strings or date corresponding to specific objects.
    RECORDS : Set of records of heterogeneous types.
    ARRAY_WITH_UNITS : ArrayWithUnits.

    """

    STRING = auto()
    SINGLE_STATEMENT = auto()
    STATEMENT_LIST = auto()
    ARRAY = auto()
    TABLE_SET = auto()
    PARAMETERS = auto()
    OBJECT_LIST = auto()
    RECORDS = auto()
    ARRAY_WITH_UNITS = auto()
    NAMED_TABLE = auto()  # e.g WELLTRACK


class SECTIONS(Enum):
    """
    Model data sections.

    Attributes
    ----------
    RUNSPEC : str
        RUNSPEC section.
    NONE : str
        No section.
    GRID : str
        GRID section.
    SCHEDULE : str
        SCHEDULE section.
    EDIT : str
        EDIT section.
    PROPS : str
        PROPS section.
    SOLUTION : str
        SOLUTION section.
    SUMMARY : str
        SUMMARY section.
    REGIONS : str
        REGIONS section

    """

    RUNSPEC = 'RUNSPEC'
    NONE = ''
    GRID = 'GRID'
    SCHEDULE = 'SCHEDULE'
    EDIT = 'EDIT'
    PROPS = 'PROPS'
    SOLUTION = 'SOLUTION'
    SUMMARY = 'SUMMARY'
    REGIONS = 'REGIONS'


TABLE_COLUMNS = {
    'RUNCTRL': ('Parameter', 'Value'),
    'TNAVCTRL': ('Parameter', 'Value'),
    'DENSITY': ('DENSO', 'DENSW', 'DENSG'),
}

DTYPES = {
    'ACTNUM': bool,
    'TSTEP': int,
}


class StringSpecification(NamedTuple):
    """
    Specification for string data type.

    Attributes
    ----------
    date : bool, default False
        Does string represents date.

    """

    date: bool = False


class ParametersSpecification(NamedTuple):
    """
    Specification for parameter data type.

    Attributes
    ----------
    tabulated : bool, default False
        Are values tabulated.

    """

    tabulated: bool = False


class StatementSpecification(NamedTuple):
    """
    Specification for statement data type.

    Attributes
    ----------
    columns : Sequence[str]
        Statement columns.
    dtypes : Sequence[DTypeString]
        Column data types.
    terminated : bool, default False
        Is statement teminated by slash.

    """

    columns: Sequence[str]
    dtypes: Sequence[DTypeString]
    terminated: bool = True


class TableSpecification(NamedTuple):
    """
    Specifiction for table data type.

    Attributes
    ----------
    columns : Sequence[str]
        Table columns.
    domain : Sequence[int] | None
    dtypes : Sequence[DTypeString]
        Column data types.
    header : StatementSpecification | None
        Header specification.
    number : int | Callable[[DataType | None], IntType]
        Number of tables.

    """

    columns: Sequence[str] | Callable[[pd.DataFrame], Sequence[str]]
    domain: Sequence[int] | None
    dtypes: Sequence[DTypeString] | DTypeString = 'float'
    header: StatementSpecification | None = None
    number: (
        int
        | Callable[
            [DataType | None],
            IntType,
        ]
    ) = 1


class ArraySpecification(NamedTuple):
    """
    Specification for array data type.

    Attributes
    ----------
    dtype : type
        Data type.

    """

    dtype: type


class RecordsSpecification(NamedTuple):
    """
    Specification for records data type.

    Attributes
    ----------
    specifications : Sequence[StatementSpecification | ArraySpecification] | None
    dynamic : bool, default False
    get_next_specification : Callable[
            [
                Sequence[
                    pd.DataFrame
                    | npt.NDArray[np.integer | np.floating | np.bool_]
                    | tuple[pd.DataFrame, pd.DataFrame]
                ]
            ],
            StatementSpecification | ArraySpecification,
        ]
        | None
        Callable to get new specification.

    """

    specifications: Sequence[StatementSpecification | ArraySpecification] | None
    dynamic: bool = False
    get_next_specification: (
        Callable[
            [
                Sequence[
                    pd.DataFrame
                    | npt.NDArray[np.integer | np.floating | np.bool_]
                    | tuple[pd.DataFrame, pd.DataFrame]
                ]
            ],
            StatementSpecification | ArraySpecification,
        ]
        | None
    ) = None


class ObjectSpecification(NamedTuple):
    """
    Specification for object data type.

    Attributes
    ----------
    terminated : bool, default False
        Should objects be terminated by slashes.
    date : bool, default False
        Does each object represent date.

    """

    terminated: bool = False
    date: bool = False


class NoDataSpecification(NamedTuple):
    """
    Specification for entries with no data.

    Attributes
    ----------
    terminated : bool, default False
        Is keyword terminated with slash.

    """

    terminated: bool = False


SpecificationType = (
    None
    | StringSpecification
    | ObjectSpecification
    | ArraySpecification
    | StatementSpecification
    | TableSpecification
    | RecordsSpecification
    | NoDataSpecification
    | ParametersSpecification
)


class KeywordSpecification(NamedTuple):
    """
    Keyword specification.

    Attributes
    ----------
    keyword : str
        Keyword.
    type : DataTypes | None
        Data type.
    specification : SpecificationType
        Specification for corresponding data type.
    sections : Sequence[SECTIONS]
        Sections in which keyword may present.

    """

    keyword: str
    type: DataTypes | None
    specification: SpecificationType
    sections: Sequence[SECTIONS]


def _get_vfpprod_specification(
    data: Sequence[
        pd.DataFrame
        | tuple[pd.DataFrame, pd.DataFrame]
        | npt.NDArray[np.floating | np.integer | np.bool_]
    ],
):
    default_specs = (
        StatementSpecification(
            [
                'TABLE_NUM',
                'BH_DATUM_DEPTH',
                'FLO',
                'WFR',
                'GFR',
                'THP',
                'ALQ',
                'UNITS',
                'QUANTITY',
            ],
            cast(
                Sequence[DTypeString],
                [
                    'int',
                    'float',
                    'text',
                    'text',
                    'text',
                    'text',
                    'text',
                    'text',
                    'text',
                ],
            ),
        ),
        ArraySpecification(float),
        ArraySpecification(float),
        ArraySpecification(float),
        ArraySpecification(float),
        ArraySpecification(float),
    )
    if len(data) < 6:
        return default_specs[len(data)]
    n_max = 6 + len(data[2]) * len(data[3]) * len(data[4]) * len(data[5])
    if len(data) >= n_max:
        raise ValueError('Max number of records is exceeded.')
    spec = StatementSpecification(
        ['NT', 'NW', 'NG', 'NA'] + [f'BHP_THT{i + 1}' for i in range(len(data[1]))],
        cast(Sequence[DTypeString], ['int'] * 4 + ['float'] * len(data[1])),
    )
    return spec


STATEMENT_LIST_INFO = {
    'RUNCTRL': {'columns': ['PARAMETER', 'VALUE'], 'dtypes': ['text', 'text']},
    'WELSPECS': {
        'columns': [
            'WELL',
            'GROUP',
            'IW',
            'JW',
            'REF_DEPTH',
            'PHASE',
            'DRAINAGE_RADIUS',
            'INFLOW_EQUATION_FLAG',
            'SHUT_OR_STOP',
            'CROSFLOW_ABILITY_FLAG',
            'PRESSURE_TABLE_NUMBER',
            'DENSITY_CALCULATION_TYPE',
            'PRESSURE_RETRIEVING',
        ],
        'dtypes': [
            'text',
            'text',
            'int',
            'int',
            'float',
            'text',
            'float',
            'text',
            'text',
            'text',
            'int',
            'text',
            'text',
        ],
    },
    'COMPDAT': {
        'columns': [
            'WELL',
            'IW',
            'JW',
            'K1',
            'K2',
            'STATUS',
            'SAT_TABLE_NUM',
            'CF',
            'DIAMETER',
            'KH_EFF',
            'SKIN',
            'D_FACTOR',
            'DIRECTION',
            'RADIUS_EFF',
        ],
        'dtypes': [
            'text',
            'int',
            'int',
            'int',
            'int',
            'text',
            'int',
            'float',
            'float',
            'float',
            'float',
            'float',
            'text',
            'float',
        ],
    },
    'WCONINJE': {
        'columns': [
            'WELL',
            'FLUID',
            'MODE',
            'CONTROL',
            'SURFACE_RATE',
            'RESERVOIR_RATE',
            'BHP',
            'THP',
            'VFP_TABLE_NUM',
            'OIL_GAS_CONCETRATION',
            '',
            'SURFACE_OIL_PROPORTION',
            'SURFACE_WATER_PROPORTION',
            'SURFACE_GAS_CONCETRATION',
        ],
        'dtypes': [
            'text',
            'text',
            'text',
            'text',
            'float',
            'float',
            'float',
            'float',
            'int',
            'float',
            'text',
            'float',
            'float',
            'float',
        ],
    },
    'WCONPROD': {
        'columns': [
            'WELL',
            'MODE',
            'CONTROL',
            'OIL_RATE',
            'WATER_RATE',
            'GAS_RATE',
            'SURFACE_LIQUID_RATE',
            'RESERVOIR_LIQUID_RATE',
            'BHP',
            'THP',
            'VFP_TABLE_NUM',
            'ALQ',
            'WET_GAS_PRODUCTION_RATE',
            'TOTAL_MOLAR_RATE',
            'STEAM_PRODUCTION',
            'PRESSURE_OFFSET',
            'TEMPERATURE_OFFSET',
            'CALORIFIC_RATE',
            'LINEARLY_COMBINED_RATE_TARGET',
            'NGL_RATE',
        ],
        'dtypes': ['text'] * 3 + ['float'] * 7 + ['int'] + ['float'] * 9,
    },
    'DIMENS': {
        'columns': ['NX', 'NY', 'NZ'],
        'dtypes': ['int', 'int', 'int'],
    },
    'NUMRES': {'columns': ['NRES'], 'dtypes': ['int']},
    'NSTACK': {'columns': ['NSTACK'], 'dtypes': ['int']},
    'MAPAXES': {
        'columns': ['X1', 'Y1', 'X0', 'Y0', 'X2', 'Y2'],
        'dtypes': ['float'] * 6,
    },
    'TABDIMS': {
        'columns': [
            'SAT_REGIONS_NUM',
            'PVT_REGIONS_NUM',
            'SAT_NODES_NUM',
            'PVT_NODES_NUM',
            'FIP_REGIONS_NUM',
            'OIL_VAP_NODES_NUM',
            'OGR_NODES_NUM',
            'SAT_END_POINT_NUM',
            'EOS_REGIONS_NUM',
            'EOS_SURFACE_REGIONS_NUM',
            'FLUX_REGIONS_NUM',
            'THERMAL_REGIONS_NUM',
            'ROCK_TABLES_NUM',
            'PRESSURE_MAINTAINACE_REGIONS_NUM',
            'TEMPERATURE_NODES_NUM',
            'TRANSPORT_COEFFICIENTS_NUM',
        ],
        'dtypes': ['int'] * 16,
    },
    'EQLDIMS': {
        'columns': [
            'EQL_NUM',
            'EQL_NODE_NUM',
            'DEPTH_NODE_MAX_NUM',
            'INIT_TRAC_CONC_NUM',
            'INIT_TRAC_CONC_NODE_NUM',
        ],
        'dtypes': ['int'] * 5,
    },
    'REGDIMS': {
        'columns': [
            'FIP_REGIONS_NUM',
            'FIP_FAMILIES_NUM',
            'RESERVOIR_REGIONS_NUM',
            'FLUX_REGIONS_NUM',
            'TRACK_REGIONS_NUM',
            'COAL_REG_NUM',
            'OPER_REGIONS_NUM',
            'WORK_NUM',
            'IWORK_NUM',
            'PLMIX_REGIONS_NUM',
        ],
        'dtypes': ['int'] * 10,
    },
    'WELLDIMS': {
        'columns': [
            'WELL_NUM',
            'CONN_NUM',
            'GROUP_NUM',
            'WELL_IN_GROUP_NUM',
            'SEP_STAGES_NUM',
            'WELL_STREAM_NUM',
            'MIXTURE_NUM',
            'SEP_NUM',
            'MIXTURE_ITEMS_NUM',
            'CON_GROUP_NUM',
            'WELL_LIST_NUM',
            'DYN_WELL_LIST_NUM',
        ],
        'dtypes': ['int'] * 12,
    },
    'VFPPDIMS': {
        'columns': [
            'FLOW_VAL_NUM',
            'TUB_HEAD_PRES_VAL_NUM',
            'WFR_NUM',
            'GFR_NUM',
            'ALQ_NUM',
            'VFP_TAB_NUM',
        ],
        'dtypes': ['int'] * 6,
    },
    'VFPIDIMS': {
        'columns': ['FLOW_VAL_NUM', 'THP_VAL_NUN', 'VFP_TAB_NUM'],
        'dtypes': ['int'] * 3,
    },
    'AQUDIMS': {
        'columns': [
            'AQUNUM_LINES_NUM',
            'AQUCON_LINES_NUM',
            'AQUTAB_LINES_NUM',
            'AQUCT_INF_TABLE_ROW_NUM',
            'AQU_ANALYTIC_NUM',
            'AQU_AN_GRID_BLOCKS_NUM',
            'AQU_LIST_NUM',
            'AQU_IN_LIST_NUM',
        ],
        'dtypes': ['int'] * 8,
    },
    'SPECGRID': {
        'columns': ['NX', 'NY', 'NZ', 'NRES', 'COORDINATE_SYSTEM'],
        'dtypes': ['int'] * 4 + ['text'],
    },
    'COPY': {
        'columns': ['SOURCE', 'DEST', 'IMIN', 'IMAX', 'JMIN', 'JMAX', 'KMIN', 'KMAX'],
        'dtypes': ['text'] * 2 + ['int'] * 6,
    },
    'MULTIPLY': {
        'columns': [
            'ARR',
            'MULTIPLYER',
            'IMIN',
            'IMAX',
            'JMIN',
            'JMAX',
            'KMIN',
            'KMAX',
        ],
        'dtypes': ['text', 'float'] + ['int'] * 6,
    },
}

RECORDS_INFO = {
    'TUNING': [
        {
            'columns': [
                'TSINIT',
                'TSMAXZ',
                'TSMINZ',
                'TSMCHP',
                'TSFMAX',
                'TSFMIN',
                'TSFCNV',
                'TFDIFF',
                'TFRUPT',
                'TMAXWC',
            ],
            'dtypes': ['float'] * 10,
        },
        {
            'columns': [
                'TRGTTE',
                'TRGCNV',
                'TRGMBE',
                'TRGLCV',
                'XXXTTE',
                'XXXCNV',
                'XXXMBE',
                'XXXLCV',
                'XXXWFL',
                'TRGFIP',
                'TRGSFT',
                'THIONX',
                'TRWGHT',
            ],
            'dtypes': ['float'] * 12 + ['int'],
        },
        {
            'columns': [
                'NEWTMX',
                'NEWTMN',
                'LITMAX',
                'LITMIN',
                'MXWSIT',
                'MXWPIT',
                'DDPLIM',
                'DDSLIM',
                'TRGDPR',
                'XXXDPR',
            ],
            'dtypes': ['int'] * 6 + ['float'] * 4,
        },
    ]
}


def _get_generic_region_number(
    data: DataType | None, kw: str, column: str, section: str = 'RUNSPEC'
) -> IntType:
    if data is None:
        return 1
    if section not in data:
        return 1
    for d in data[section]:
        if d[0] == kw:
            if not isinstance(d[1], pd.DataFrame):
                raise ValueError(
                    f'Value for keyword {kw} should be of type pandas.DataFrame.'
                )
            if d[1][column].values[0] > 0:
                val = d[1][column].values[0]  # pyright: ignore[reportAny]
                if not isinstance(val, IntType):
                    raise ValueError('Value in column {column} should be integer.')
                return val  # pyright: ignore[reportUnknownVariableType]
    return 1


def _get_pvt_regions_number(data: DataType | None = None):
    return _get_generic_region_number(data, 'TABDIMS', 'PVT_REGIONS_NUM')


def _get_eql_regions_number(data: DataType | None = None):
    return _get_generic_region_number(data, 'EQLDIMS', 'EQL_NUM')


def _get_sat_fun_regions_number(data: DataType | None = None):
    return _get_generic_region_number(data, 'TABDIMS', 'SAT_REGIONS_NUM')


def _get_rock_tables_number(data: DataType | None = None):  # pyright: ignore[reportUnusedFunction]
    rock_num = _get_generic_region_number(data, 'TABDIMS', 'ROCK_TABLES_NUM')
    if rock_num == INT_NAN:
        rock_num = _get_generic_region_number(data, 'TABDIMS', 'PVT_REGIONS_NUM')
    return rock_num


def _get_rock_regions_number(data: DataType | None = None):
    return _get_generic_region_number(data, 'ROCKCOMP', 'ROCK_REGIONS_NUM')


def _get_eos_regions_number(data: DataType | None = None):
    return _get_generic_region_number(data, 'TABDIMS', 'EOS_REGIONS_NUM')


def _get_gptable_number(data: DataType | None = None):
    return _get_generic_region_number(data, 'FIELDSEP', 'GPTABLE_NUM')


def _get_reaction_number(data: DataType | None = None):
    return _get_generic_region_number(data, 'REACTION', 'REACTIONS_NUM')


DATA_DIRECTORY: Final[dict[str, KeywordSpecification | None]] = {
    'TITLE': KeywordSpecification('TITLE', DataTypes.STRING, None, (SECTIONS.RUNSPEC,)),
    **{
        kw: KeywordSpecification(kw, None, None, (SECTIONS.RUNSPEC,))
        for kw in ['MULTOUT', 'MULTOUTS', 'UNIFOUT', 'METRIC']
    },
    'START': KeywordSpecification(
        'START', DataTypes.STRING, StringSpecification(True), (SECTIONS.RUNSPEC,)
    ),
    **{
        kw: KeywordSpecification(kw, None, None, (SECTIONS.RUNSPEC,))
        for kw in FLUID_KEYWORDS
    },
    **{
        kw: KeywordSpecification(
            kw,
            DataTypes.SINGLE_STATEMENT,
            StatementSpecification(
                STATEMENT_LIST_INFO[kw]['columns'],
                cast(Sequence[DTypeString], STATEMENT_LIST_INFO[kw]['dtypes']),
            ),
            (SECTIONS.RUNSPEC,),
        )
        for kw in ['DIMENS', 'NUMRES', 'NSTACK'] + list(DIMS_KEYWORDS)
    },
    **{
        kw: KeywordSpecification(
            kw,
            DataTypes.PARAMETERS,
            ParametersSpecification(tabulated=True),
            (SECTIONS.RUNSPEC,),
        )
        for kw in ('RUNCTRL',)
    },
    'TUNING': KeywordSpecification(
        'TUNING',
        DataTypes.RECORDS,
        RecordsSpecification(
            [
                StatementSpecification(
                    r['columns'], cast(Sequence[DTypeString], r['dtypes'])
                )
                for r in RECORDS_INFO['TUNING']
            ]
        ),
        (SECTIONS.RUNSPEC, SECTIONS.SCHEDULE),
    ),
    'SPECGRID': KeywordSpecification(
        'SPECGRID',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            STATEMENT_LIST_INFO['SPECGRID']['columns'],
            cast(Sequence[DTypeString], STATEMENT_LIST_INFO['SPECGRID']['dtypes']),
        ),
        (SECTIONS.GRID,),
    ),
    'INIT': KeywordSpecification('INIT', None, None, (SECTIONS.GRID,)),
    'MAPAXES': KeywordSpecification(
        'MAPAXES',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            STATEMENT_LIST_INFO['MAPAXES']['columns'],
            cast(Sequence[DTypeString], STATEMENT_LIST_INFO['MAPAXES']['dtypes']),
        ),
        (SECTIONS.GRID,),
    ),
    **{
        kw: KeywordSpecification(
            kw,
            DataTypes.ARRAY,
            ArraySpecification(DTYPES[kw] if kw in DTYPES else float),
            (SECTIONS.GRID,),
        )
        for kw in list(ORTHOGONAL_GRID_KEYWORDS) + list(ROCK_GRID_KEYWORDS)
    },
    'MULTIPLY': KeywordSpecification(
        'MULTIPLY',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            STATEMENT_LIST_INFO['MULTIPLY']['columns'],
            cast(Sequence[DTypeString], STATEMENT_LIST_INFO['MULTIPLY']['dtypes']),
        ),
        (SECTIONS.GRID,),
    ),
    'COPY': KeywordSpecification(
        'COPY',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            STATEMENT_LIST_INFO['COPY']['columns'],
            cast(Sequence[DTypeString], STATEMENT_LIST_INFO['COPY']['dtypes']),
        ),
        (SECTIONS.GRID,),
    ),
    'PVTO': KeywordSpecification(
        'PVTO',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['RS', 'PRESSURE', 'FVF', 'VISC'],
            domain=[0, 1],
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'PVTG': KeywordSpecification(
        'PVTG',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['PRESSURE', 'RV', 'FVF', 'VISC'],
            domain=[0, 1],
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'PVDG': KeywordSpecification(
        'PVDG',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['PRESSURE', 'FVF', 'VISC'], domain=[0], number=_get_pvt_regions_number
        ),
        (SECTIONS.PROPS,),
    ),
    'PVDO': KeywordSpecification(
        'PVDO',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['PRESSURE', 'FVF', 'VISC'], domain=[0], number=_get_pvt_regions_number
        ),
        (SECTIONS.PROPS,),
    ),
    'PVTW': KeywordSpecification(
        'PVTW',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['PRESSURE', 'FVF', 'COMPR', 'VISC', 'VISCOSIBILITY'],
            domain=[0],
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'PVCDO': KeywordSpecification(
        'PVCDO',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['PRESSURE', 'FVF', 'COMPR', 'VISC', 'VISCOSIBILITY'],
            domain=[0],
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'SWOF': KeywordSpecification(
        'SWOF',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['SW', 'KRWO', 'KROW', 'POW'],
            domain=[0],
            number=_get_sat_fun_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'SGOF': KeywordSpecification(
        'SGOF',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['SW', 'KRGO', 'KROG', 'POG'],
            domain=[0],
            number=_get_sat_fun_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'RSVD': KeywordSpecification(
        'RSVD',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['DEPTH', 'RS'],
            domain=[0],
            number=_get_eql_regions_number,
        ),
        (SECTIONS.SOLUTION,),
    ),
    'ROCK': KeywordSpecification(
        'ROCK',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['PRESSURE', 'COMPR'],
            domain=[0],
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'DENSITY': KeywordSpecification(
        'DENSITY',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['DENSO', 'DENSW', 'DENSG'],
            domain=None,
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'EQUIL': KeywordSpecification(
        'EQUIL',
        DataTypes.TABLE_SET,
        TableSpecification(
            [
                'DEPTH',
                'PRES',
                'WOC_DEPTH',
                'WOC_PC',
                'GOC_DEPTH',
                'GOC_PC',
                'RSVD_PBVD_TABLE',
                'RVVD_PDVD_TABLE',
                'ACCURACY',
                'INITIALIZTION_TYPE',
                'FLAG',
            ],
            None,
            cast(Sequence[DTypeString], ['float'] * 6 + ['int'] * 5),
            header=None,
            number=_get_eql_regions_number,
        ),
        (SECTIONS.SOLUTION,),
    ),
    'RPTSOL': KeywordSpecification(
        'RPTSOL', DataTypes.PARAMETERS, ParametersSpecification(), (SECTIONS.SOLUTION,)
    ),
    **{
        kw: KeywordSpecification(kw, None, None, (SECTIONS.SUMMARY,))
        for kw in FIELD_SUMMARY_KEYWORDS
    },
    **{
        kw: KeywordSpecification(kw, DataTypes.OBJECT_LIST, None, (SECTIONS.SUMMARY,))
        for kw in WELL_SUMMARY_KEYWORDS
    },
    **{
        kw: KeywordSpecification(kw, None, None, (SECTIONS.SUMMARY,))
        for kw in TOTAL_SUMMARY_KEYWORDS
    },
    'EXCEL': KeywordSpecification('EXCEL', None, None, (SECTIONS.SUMMARY,)),
    'RPTONLY': KeywordSpecification('RPTONLY', None, None, (SECTIONS.SUMMARY,)),
    'RPTSCHED': KeywordSpecification(
        'RPTSCHED',
        DataTypes.PARAMETERS,
        ParametersSpecification(),
        (SECTIONS.SCHEDULE,),
    ),
    'RPTRST': KeywordSpecification(
        'RPTRST',
        DataTypes.PARAMETERS,
        ParametersSpecification(),
        (SECTIONS.SCHEDULE, SECTIONS.SOLUTION),
    ),
    'WELSPECS': KeywordSpecification(
        'WELSPECS', DataTypes.STATEMENT_LIST, None, (SECTIONS.SCHEDULE,)
    ),
    'TSTEP': KeywordSpecification(
        'TSTEP', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.SCHEDULE,)
    ),
    **{
        kw: KeywordSpecification(
            kw,
            DataTypes.STATEMENT_LIST,
            StatementSpecification(
                STATEMENT_LIST_INFO[kw]['columns'],
                cast(Sequence[DTypeString], STATEMENT_LIST_INFO[kw]['dtypes']),
            ),
            (SECTIONS.SCHEDULE,),
        )
        for kw in SCHEDULE_KEYWORDS
    },
    **{
        kw: KeywordSpecification(kw, None, None, [val for val in SECTIONS])
        for kw in ('NOECHO', 'ECHO', 'END')
    },
    'INCLUDE': KeywordSpecification(
        'INCLUDE', DataTypes.STRING, None, [val for val in SECTIONS]
    ),
    **{
        kw: KeywordSpecification(
            kw,
            DataTypes.PARAMETERS,
            ParametersSpecification(
                tabulated=True,
            ),
            [val for val in SECTIONS],
        )
        for kw in ('REPORTSCREEN', 'REPORTFILE')
    },
    'ENDSCALE': KeywordSpecification(
        'ENDSCALE',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            [
                'DIRECTIONAL_SWITCH',
                'IRREVERSIBLE_SWITCH',
                'SAT_POINTS_NUM',
                'NODE_NUM',
                'COMBININNG',
                'EQUILIBRATION',
            ],
            cast(Sequence[DTypeString], ['text'] * 2 + ['int'] * 3 + ['text']),
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'FAULTDIM': KeywordSpecification(
        'FAULTDIM',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(['N'], ['int']),
        (SECTIONS.RUNSPEC,),
    ),
    'EQLOPTS': KeywordSpecification(
        'EQLOPTS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(['OPTION'], cast(Sequence[DTypeString], ['text'])),
        (SECTIONS.RUNSPEC,),
    ),
    'MESSAGES': KeywordSpecification(
        'MESSAGES',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            [f'PRINT_LIM_SEV{i}' for i in range(1, 7)]
            + [f'STOP_LIM_SEV{i}' for i in range(1, 7)],
            cast(Sequence[DTypeString], ['int'] * 12),
        ),
        [val for val in SECTIONS],
    ),
    'GRIDFILE': KeywordSpecification(
        'GRIDFILE',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['_', 'DUMP_INIT_EGRID'], cast(Sequence[DTypeString], ['int'] * 2)
        ),
        (SECTIONS.GRID,),
    ),
    'MINPV': KeywordSpecification(
        'MINPV',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(['VAL'], ['float']),
        (SECTIONS.GRID,),
    ),
    'PINCH': KeywordSpecification(
        'PINCH',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            [
                'THRESHOLD_THICKNESS',
                'MINPV_NN_CONTROL',
                'MAX_GAP',
                'NN_TRANSMISSIBILITY_METHOD',
                'VERT_TRANSMISSIBILITY_METHOD',
            ],
            cast(Sequence[DTypeString], ['float', 'text', 'float', 'text', 'text']),
        ),
        (SECTIONS.GRID,),
    ),
    'FAULTS': KeywordSpecification(
        'FAULTS',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            ['NAME', 'I1', 'I2', 'J1', 'J2', 'K1', 'K2', 'FACE'],
            cast(Sequence[DTypeString], ['text'] + ['int'] * 6 + ['text']),
        ),
        (SECTIONS.GRID,),
    ),
    'JFUNC': KeywordSpecification(
        'JFUNC',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['PHASE', 'STW', 'STG', 'ALPHA', 'BETA', 'PERM_DIR'],
            cast(Sequence[DTypeString], ['text'] + ['float'] * 4 + ['text']),
        ),
        (SECTIONS.GRID,),
    ),
    'COORD': KeywordSpecification(
        'COORD', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'ZCORN': KeywordSpecification(
        'ZCORN', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'EDITNNC': KeywordSpecification(
        'EDITNNC',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'I1',
                'J1',
                'K1',
                'I2',
                'J2',
                'K2',
                'TRANSMISSIBILITY_MULT',
                'SAT_REG1',
                'SAT_REG2',
                'PVT_REG1',
                'PVT_REG2',
                'DIR1',
                'DIR2',
                'DIFFUSIVITY_MULT',
            ],
            cast(
                Sequence[DTypeString],
                ['int'] * 6 + ['float'] + ['int'] * 4 + ['text'] * 2 + ['float'],
            ),
        ),
        (SECTIONS.EDIT,),
    ),
    'STONE1': KeywordSpecification('STONE1', None, None, (SECTIONS.PROPS,)),
    'STONE2': KeywordSpecification('STONE2', None, None, (SECTIONS.PROPS,)),
    'SCALECRS': KeywordSpecification(
        'SCALECRS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(['VALUE'], cast(Sequence[DTypeString], ['text'])),
        (SECTIONS.PROPS,),
    ),
    'PVTNUM': KeywordSpecification(
        'PVTNUM',
        DataTypes.ARRAY,
        ArraySpecification(int),
        (SECTIONS.REGIONS, SECTIONS.SCHEDULE),
    ),
    'EQLNUM': KeywordSpecification(
        'EQLNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'FIPFAULT': KeywordSpecification(
        'FIPFAULT', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'SATNUM': KeywordSpecification(
        'SATNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'FIPNUM': KeywordSpecification(
        'FIPNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'PBVD': KeywordSpecification(
        'PBVD',
        DataTypes.TABLE_SET,
        TableSpecification(['DEPTH', 'PB'], [0], number=_get_eql_regions_number),
        (SECTIONS.SOLUTION,),
    ),
    'THPRES': KeywordSpecification(
        'THPRES',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(['REG1', 'REG2', 'TH'], ['int', 'int', 'float']),
        (SECTIONS.SOLUTION,),
    ),
    'OUTSOL': KeywordSpecification(
        'OUTSOL',
        DataTypes.PARAMETERS,
        ParametersSpecification(),
        (SECTIONS.SOLUTION, SECTIONS.SCHEDULE),
    ),
    **{
        kw: KeywordSpecification(kw, None, None, [val for val in SECTIONS])
        for kw in ('SKIP', 'SKIPOFF', 'SKIPON', 'ENDSKIP')
    },
    **{
        kw: KeywordSpecification(kw, DataTypes.OBJECT_LIST, None, (SECTIONS.SUMMARY,))
        for kw in REGION_SUMMARY_KEYWORDS
    },
    **{
        kw: KeywordSpecification(kw, None, None, (SECTIONS.SUMMARY,))
        for kw in MODEL_SUMMARY_KEYWORDS
    },
    **{
        kw: KeywordSpecification(kw, DataTypes.OBJECT_LIST, None, (SECTIONS.SUMMARY,))
        for kw in GROUP_SUMMARY_KEYWORDS
    },
    'WCONHIST': KeywordSpecification(
        'WCONHIST',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'WELL',
                'MODE',
                'CONTROL',
                'OIL_RATE',
                'WATER_RATE',
                'GAS_RATE',
                'VFP_TABLE_NUM',
                'ALQ',
                'THP',
                'BHP',
                'WET_GAS_PRODUCTION_RATE',
                'NGL_RATE',
            ],
            cast(
                Sequence[DTypeString],
                ['text'] * 3 + ['float'] * 3 + ['int'] + ['float'] * 5,
            ),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'DATES': KeywordSpecification(
        'DATES',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True, date=True),
        (SECTIONS.SCHEDULE,),
    ),
    'WCONINJH': KeywordSpecification(
        'WCONINJH',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'WELL',
                'FLUID',
                'MODE',
                'SURFACE_RATE',
                'BHP',
                'THP',
                'VFP_TABLE_NUM',
                'OIL_GAS_CONCETRATION',
                'SURFACE_OIL_PROPORTION',
                'SURFACE_WATER_PROPORTION',
                'SURFACE_GAS_CONCETRATION',
                'CONTROL',
            ],
            cast(
                Sequence[DTypeString],
                [
                    'text',
                    'text',
                    'text',
                    'float',
                    'float',
                    'float',
                    'int',
                    'float',
                    'float',
                    'float',
                    'float',
                    'text',
                ],
            ),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'RPTRSTD': KeywordSpecification(
        'RPTRSTD',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(date=True),
        (SECTIONS.SOLUTION,),
    ),
    'COMPS': KeywordSpecification(
        'COMPS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(columns=['N'], dtypes=['int']),
        (SECTIONS.RUNSPEC,),
    ),
    'NCOMPS': KeywordSpecification(
        'NCOMPS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(columns=['N'], dtypes=['int']),
        (SECTIONS.PROPS,),
    ),
    'PARALLEL': KeywordSpecification(
        'PARALLEL',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['NDMAIN', 'MACHINE_TYPE'], cast(Sequence[DTypeString], ['int', 'text'])
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'NOMIX': KeywordSpecification('NOMIX', None, None, (SECTIONS.RUNSPEC,)),
    'MULTPV': KeywordSpecification(
        'MULTPV',
        DataTypes.ARRAY,
        ArraySpecification(float),
        (SECTIONS.EDIT, SECTIONS.GRID),
    ),
    'STCOND': KeywordSpecification(
        'STCOND',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(['TEMP', 'PRES'], ['float', 'float']),
        (SECTIONS.PROPS,),
    ),
    'RTEMP': KeywordSpecification(
        'RTEMP',
        DataTypes.TABLE_SET,
        TableSpecification(['TEMP'], None, number=_get_eos_regions_number),
        (SECTIONS.PROPS,),
    ),
    'EOS': KeywordSpecification(
        'EOS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(['EOS'], cast(Sequence[DTypeString], ['text'])),
        (
            SECTIONS.RUNSPEC,
            SECTIONS.PROPS,
        ),
    ),
    'CNAMES': KeywordSpecification(
        'CNAMES', DataTypes.PARAMETERS, ParametersSpecification(), (SECTIONS.PROPS,)
    ),
    'TCRIT': None,
    'PCRIT': None,
    'VCRIT': None,
    'ZCRIT': None,
    'VCRITVIS': None,
    'ZCRITVIS': None,
    'MW': None,
    'ACF': None,
    'BIC': KeywordSpecification(
        'BIC', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.PROPS,)
    ),
    'ZMFVD': None,
    'VFPPROD': KeywordSpecification(
        'VFPPROD',
        DataTypes.RECORDS,
        RecordsSpecification(None, True, _get_vfpprod_specification),
        (SECTIONS.SCHEDULE,),
    ),
    'GRUPTREE': KeywordSpecification(
        'GRUPTREE',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            ['CHILD', 'PARENT'], cast(Sequence[DTypeString], ['text', 'text'])
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'GCONPROD': KeywordSpecification(
        'CGONPROD',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'GROUP',
                'CONTROL',
                'OIL_RATE',
                'WATER_RATE',
                'GAS_RATE',
                'SURFACE_LIQUID_RATE',
                'WORKOVER_ACTION_OIL',
                'HIGHER_GROUP_CONTROL',
                'HIGHER_GROUP_CONTROL_RATIO',
                'HIGHER_GROUP_CONTROL_RATIO_PHASE',
                'WORKOVER_ACTION_WATER',
                'WORKOVER_ACTION_GAS',
                'WORKOVER_ACTION_LIQ',
                'RESERVOIR_LIQUID_RATE',
                'RESERVOIR_FRACTION_TARGET',
                'WET_GAS_PRODUCTION_RATE',
                'CALORIFIC_RATE',
                'SURFACE_GAS_FRACTION_TARGET',
                'SURFACE_WATER_FRACTION_TARGET',
            ],
            cast(
                Sequence[DTypeString],
                ['text'] * 2
                + ['foat'] * 4
                + ['text'] * 2
                + ['float']
                + ['text'] * 4
                + ['float'] * 6,
            ),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'IMPLICIT': KeywordSpecification('IMPLICIT', None, None, (SECTIONS.RUNSPEC,)),
    'FIPPATT': KeywordSpecification(
        'FIPATT', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'WELTARG': KeywordSpecification(
        'WELTARG',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            ['WELL', 'CONTROL', 'VAL'],
            cast(Sequence[DTypeString], ['text', 'text', 'float']),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'WECON': KeywordSpecification(
        'WECON',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'WELL',
                'OIL_RATE_LOWER_LIMIT',
                'GAS_RATE_UPPER_LIMIT',
                'WATER_CUT_UPPER_LIMIT',
                'GAS_OIL_RATIO_UPPER_LIMIT',
                'WATER_GAS_RATIO_UPPER_LIMIT',
                'VIOLATION_WORKOVER',
                'END_RUN_FLUG',
                'WELL_TO_OPEN',
                'PARAMETER_TO_APPLY_LIMIT',
                'SECONDARY_WATER_CUT_LIMIT',
                'SECONDARY_WATER_CUT_VIOLATION_WORKOVER',
                'GAS_LIQUID_RATIO_UPPER_LIMIT',
                'LIQUID_RATE_LOWER_LIMIT',
            ],
            cast(
                Sequence[DTypeString],
                ['text']
                + ['float'] * 5
                + ['text'] * 4
                + ['float']
                + ['text']
                + ['float'] * 2,
            ),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'FIELD': KeywordSpecification('FIELD', None, None, (SECTIONS.RUNSPEC,)),
    'GRIDOPTS': KeywordSpecification(
        'GRIDOPTS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['MULTIPLIERS_FLAG', 'N_MULTIPLIERS_REG'],
            cast(Sequence[DTypeString], ['text', 'int']),
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'RPTGRID': KeywordSpecification(
        'RPTGRID', DataTypes.OBJECT_LIST, ObjectSpecification(), (SECTIONS.GRID,)
    ),
    'MAPUNITS': KeywordSpecification(
        'MAPUNITS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(['UNITS'], cast(Sequence[DTypeString], ['text'])),
        (SECTIONS.GRID,),
    ),
    'GRIDUNIT': KeywordSpecification(
        'GRIDUNIT',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['UNITS', 'MAP_FLAG'], cast(Sequence[DTypeString], ['text', 'text'])
        ),
        (SECTIONS.GRID,),
    ),
    'NEWTRAN': KeywordSpecification('NEWTRAN', None, None, (SECTIONS.GRID,)),
    'COORDSYS': None,
    'COREYWO': KeywordSpecification(
        'COREYWO',
        DataTypes.TABLE_SET,
        TableSpecification(
            [
                'SWL',
                'SWU',
                'SWCR',
                'SOWCR',
                'KROLW',
                'KRORW',
                'KRWR',
                'KRWU',
                'PCOW',
                'NOW',
                'NW',
                'NP',
                'SPCO',
            ],
            number=_get_sat_fun_regions_number,
            domain=None,
        ),
        (SECTIONS.PROPS,),
    ),
    'PRESSURE': KeywordSpecification(
        'PRESSURE', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'SWAT': KeywordSpecification(
        'SWAT', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'SGAS': KeywordSpecification(
        'SGAS', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'PBUB': KeywordSpecification(
        'PBUB', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'MULTX': KeywordSpecification(
        'MULTX', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'MULTY': KeywordSpecification(
        'MULTY', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'MULTZ': KeywordSpecification(
        'MULTZ', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'RPTRSTT': KeywordSpecification(
        'RPTRSTT',
        DataTypes.ARRAY_WITH_UNITS,
        ArraySpecification(float),
        (SECTIONS.SOLUTION, SECTIONS.SCHEDULE),
    ),
    'FIPCCC': KeywordSpecification(
        'FIPCCC', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'FIPDDD': KeywordSpecification(
        'FIPDDD', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'FIPRPT': KeywordSpecification(
        'FIPRPT', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'FIPXXX': KeywordSpecification(
        'FIPXXX', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'RESTART': KeywordSpecification(
        'RESTART', DataTypes.STRING, None, (SECTIONS.SOLUTION,)
    ),
    'RPTRSTL': KeywordSpecification(
        'RPTRSTL', None, NoDataSpecification(True), (SECTIONS.SOLUTION,)
    ),
    'UNIFIN': KeywordSpecification(
        'UINIFIN', None, NoDataSpecification(False), (SECTIONS.RUNSPEC,)
    ),
    'SWFN': KeywordSpecification(
        'SWFN',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['SW', 'KRW', 'POW'], domain=[0], number=_get_sat_fun_regions_number
        ),
        (SECTIONS.PROPS,),
    ),
    'SGFN': KeywordSpecification(
        'SGFN',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['SG', 'KRG', 'POG'], domain=[0], number=_get_sat_fun_regions_number
        ),
        (SECTIONS.PROPS,),
    ),
    'SOF3': KeywordSpecification(
        'SOF3',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['SO', 'KRO_NO_GAS', 'KRO_CONNATE_WATER'],
            domain=[0],
            number=_get_sat_fun_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'GRAVITY': KeywordSpecification(
        'GRAVITY',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['OIL_API_GRAVITY', 'WATER_GRAVITY', 'GAS_GRAVITY'],
            domain=None,
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'PRCORR': KeywordSpecification(
        'PRCORR', None, NoDataSpecification(False), (SECTIONS.PROPS,)
    ),
    'OMEGAA': KeywordSpecification(
        'OMEGAA', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.PROPS,)
    ),
    'OMEGAB': KeywordSpecification(
        'OMEGAB', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.PROPS,)
    ),
    'SSHIFT': KeywordSpecification(
        'SSHIFT', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.PROPS,)
    ),
    'DNGL': KeywordSpecification(
        'DNGL', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.PROPS,)
    ),
    'PARACHOR': KeywordSpecification(
        'PARACHOR', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.PROPS,)
    ),
    'LBCCOEF': KeywordSpecification(
        'LBCCOEF',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['A1', 'A2', 'A3', 'A4', 'A5'], cast(Sequence[DTypeString], ['float'] * 5)
        ),
        (SECTIONS.PROPS,),
    ),
    'COMPVD': None,
    'FIELDSEP': KeywordSpecification(
        'FIELDSEP',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'STAGE_NUM',
                'TEMP',
                'PRES',
                'LIQ_DEST',
                'GAS_DEST',
                'K_VAL_TABLE_NUM',
                'GPTABLE_NUM',
                'SURFACE_EOS_NUM',
            ],
            cast(Sequence[DTypeString], ['int'] + ['float'] * 2 + ['int'] * 5),
        ),
        (SECTIONS.SOLUTION,),
    ),
    'GPTABLEN': None,
    'SOLID': KeywordSpecification('SOLID', None, None, (SECTIONS.RUNSPEC,)),
    'THERMAL': KeywordSpecification('THERMAL', None, None, (SECTIONS.RUNSPEC,)),
    'FLASHCTRL': KeywordSpecification(
        'FLASCHCTRL',
        DataTypes.PARAMETERS,
        ParametersSpecification(tabulated=True),
        (SECTIONS.RUNSPEC,),
    ),
    'REACTION': KeywordSpecification(
        'REACTION',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            [
                'REACTIONS_NUM',
                'EQUIL_CORRECTION_TERM_NUM',
                'EQUIL_CONST_NUM',
                'TRACER_DEPENDENT_RATE_TERM_NUM',
                'EQUIL_TABLES_NUM',
            ],
            ['int', 'int', 'int', 'int', 'int'],
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'ROCKDIMS': KeywordSpecification(
        'ROCKDIMS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['CAP_BASE_ROCK_TYPES_NUM', '_', 'BLOCK_ROCK_CON_NUM'],
            ['int', 'int', 'int'],
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'ALL': KeywordSpecification('ALL', None, None, (SECTIONS.SUMMARY,)),
    'FULLIMP': KeywordSpecification('FULLIMP', None, None, (SECTIONS.RUNSPEC,)),
    'HEATCR': KeywordSpecification(
        'HEATCR', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'THCONR': KeywordSpecification(
        'THCONR', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'WELLSTRE': None,
    'CVTYPE': None,
    'DREF': None,
    'PREF': None,
    'TREF': None,
    'CREF': None,
    'THERMEX1': None,
    'SPECHA': None,
    'SDREF': None,
    'SPECHB': None,
    'SPECHG': None,
    'SPECHS': None,
    'KVCR': None,
    'OILVISCC': None,
    'GASVISCF': None,
    'STOREAC': None,
    'STOPROD': None,
    'REACRATE': None,
    'REACACT': None,
    'REACENTH': None,
    'REACPHA': None,
    'REACCORD': None,
    'TEMPI': KeywordSpecification(
        'TEMPI', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'XMF': KeywordSpecification(
        'XMF', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'YMF': KeywordSpecification(
        'YMF', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'BSWAT': KeywordSpecification(
        'BSWAT',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'BSOIL': KeywordSpecification(
        'BSOIL',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'BPRES': KeywordSpecification(
        'BPRES',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'BTEMP': KeywordSpecification(
        'BTEMP',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'BMLSC': KeywordSpecification(
        'BMLSC',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'BSSOLID': KeywordSpecification(
        'BSSOLID',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'BHSOL': KeywordSpecification(
        'BHSOL',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'BDENS': KeywordSpecification(
        'BDENS',
        DataTypes.OBJECT_LIST,
        ObjectSpecification(terminated=True),
        (SECTIONS.SUMMARY,),
    ),
    'PERFORMANCE': KeywordSpecification('PERFORMANCE', None, None, (SECTIONS.SUMMARY,)),
    'RUNSUM': KeywordSpecification('RUNSUM', None, None, (SECTIONS.SUMMARY,)),
    'WINJGAS': KeywordSpecification(
        'WINJGAS',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            ['WELL', 'GAS_NATURE', 'SPEC_NAME', 'WELLSTREAM_NAME', 'SEPARATOR_STAGE'],
            cast(Sequence[DTypeString], ['text', 'text', 'text', 'text', 'int']),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'WINJTEMP': KeywordSpecification(
        'WINJTEMP',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            ['NAME', 'STEAM_QUALITY', 'FLUID_TEMP', 'FLUID_PRES', 'FLUID_ENTHALPY'],
            ['text', 'float', 'float', 'float', 'float'],
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'ROCKPROP': KeywordSpecification(
        'ROCKPROP',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'ROCK_TYPE_NUMBER',
                'INITIAL_PRESSURE',
                'ROCK_CONDUCTIVITY',
                'VOLUMETRIC_HEAT_CAPACITY',
                'HEAT_CAPACITY_TEMPERATURE_COEFFICIENT',
                'HEAT_LOSS_CALCULATION_METHOD',
            ],
            ('int', 'float', 'float', 'float', 'float', 'text'),
        ),
        (SECTIONS.GRID,),
    ),
    'ROCKCON': KeywordSpecification(
        'ROCKCON',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            [
                'ROCK_TYPE_NUMBER',
                'I1',
                'I2',
                'J1',
                'J2',
                'K1',
                'K2',
                'FACE',
                'ROCK_INFLUX_COEFFICIENT',
            ],
            ('int', 'int', 'int', 'int', 'int', 'int', 'int', 'text', 'float'),
        ),
        (SECTIONS.GRID,),
    ),
    'ZFACTOR': None,
    'THANALB': KeywordSpecification('THANALB', None, None, (SECTIONS.PROPS,)),
    'HEATVAP': None,
    'TEMPVD': KeywordSpecification(
        'TEMPVD',
        DataTypes.TABLE_SET,
        TableSpecification(
            ('DEPTH', 'T'),
            dtypes=['float', 'float'],
            domain=[0],
            number=_get_eql_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'GASVISCT': None,
    'OILVISCT': KeywordSpecification(
        'OILVISCT',
        DataTypes.TABLE_SET,
        TableSpecification(
            ('T', 'VISCOSITY'),
            dtypes=['float', 'float'],
            domain=[0],
            number=_get_pvt_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'RPTPRINT': KeywordSpecification(
        'RPTPRINT',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            [
                'STEP_SUMMARY',
                'FLUID_IN_PLACE_FIELD',
                'FLUID_IN_PLACE_REGION',
                'STATUS_OF_GROUPS',
                'RESERVED1',
                'STATUS_OF_WELLS',
                'STATUS_OF_WELL_COMPLETIONS',
                'SOLUTION_DATA',
                'CONVERGENCE_OF_NON-LINEAR_EQUATIONS',
                'WELL_POTENTIALS',
                'INTER_REGIONS_FLOW',
                'COMPONENT_FLOWS',
                'AQUIFER_FLOWS',
                'HEATER_CONNECTIONS',
                'RESERVED2',
                'RESERVED3',
                'RESERVED4',
                'NETWORK_STATUS',
            ],
            cast(Sequence[DTypeString], (['int'] * 18)),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'HWELLS': KeywordSpecification('HWELLS', None, None, (SECTIONS.RUNSPEC,)),
    'WSEGDIMS': KeywordSpecification(
        'WSEGDIMS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            [
                'N_MULTISEGMENT_WELLS',
                'N_SEGMENTS',
                'N_BRANCHES',
                'N_SEGMEN_CHORD_LINKS',
            ],
            ('int', 'int', 'int', 'int'),
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'HEATCRT': KeywordSpecification(
        'HEATCRT', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.GRID,)
    ),
    'SLGOF': KeywordSpecification(
        'SLGOF',
        DataTypes.TABLE_SET,
        TableSpecification(
            ['SL', 'KRGO', 'KROG', 'PCOG'],
            domain=[0],
            number=_get_sat_fun_regions_number,
        ),
        (SECTIONS.PROPS,),
    ),
    'ZMF': KeywordSpecification(
        'ZMF', DataTypes.ARRAY, ArraySpecification(float), (SECTIONS.SOLUTION,)
    ),
    'COMPORD': KeywordSpecification(
        'COMPORD',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            ['NAME', 'METHOD'],
            ('text', 'text'),
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'ROCKCOMP': KeywordSpecification(
        'ROCKCOMP',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            [
                'ROCK_COMPACTION_OPTION',
                'ROCK_REGIONS_NUM',
                'WATER_INDUCED_COMPACTION_FLAG',
                'TRANSMISSIBILITY_POROSITY_FUNCTION',
            ],
            ['text', 'int', 'text', 'text'],
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'NETWORK': KeywordSpecification(
        'NETWORK',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ['MAX_NODES', 'MAX_BRANCHES', 'MAX_NODE_BRANCHES'], ('int', 'int', 'int')
        ),
        (SECTIONS.RUNSPEC,),
    ),
    'TZONE': KeywordSpecification(
        'TZONE',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(('OIL', 'WATER', 'GAS'), ('text', 'text', 'text')),
        (SECTIONS.PROPS,),
    ),
    'ROCKOPTS': KeywordSpecification(
        'ROCKOPTS',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(
            ('OVERBURD_METHOD', 'REFERENCE_PRESSURE_OPTION', 'TABLE_REGION', '_'),
            ('text', 'text', 'text', 'text'),
        ),
        (SECTIONS.PROPS,),
    ),
    'RKTRMDIR': KeywordSpecification('RKTRMDIR', None, None, (SECTIONS.PROPS,)),
    'ROCKTAB': None,
    'ROCKNUM': KeywordSpecification(
        'ROCKNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'ENDNUM': KeywordSpecification(
        'ENDNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)
    ),
    'DATUM': KeywordSpecification(
        'DATUM',
        DataTypes.SINGLE_STATEMENT,
        StatementSpecification(columns=['DATUM_DEPTH'], dtypes=['float']),
        (SECTIONS.SOLUTION,),
    ),
    'PRVD': KeywordSpecification(
        'PRVD',
        DataTypes.TABLE_SET,
        TableSpecification(
            columns=['DEPTH', 'PRESSURE'], domain=[0], dtypes=['float', 'float']
        ),
        (SECTIONS.SOLUTION,),
    ),
    'AQUCT': KeywordSpecification(
        'AQUCT',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'NUMBER',
                'DATUM_DEPTH',
                'INITIAL_PRESSURE',
                'PERMEABILITY',
                'POROSITY',
                'TOTAL_COMPRESSIBILITY',
                'AQUIFER_INNER_RADIUS',
                'THICKNESS',
                'INFLUENCE_ANGLE',
                'PVTW_TABLE',
                'AQUTAB_TABLE',
                'INITIAL_SALT_CONCETRATION',
                'TEMPERATURE',
                'DENY_WATER_BACKFLOW',
            ],
            dtypes=[
                'int',
                'float',
                'float',
                'float',
                'float',
                'float',
                'float',
                'float',
                'float',
                'int',
                'int',
                'float',
                'float',
                'text',
            ],
        ),
        (SECTIONS.PROPS, SECTIONS.SCHEDULE, SECTIONS.GRID, SECTIONS.SOLUTION),
    ),
    'AQUANCON': KeywordSpecification(
        'AQUANCON',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'NUMBER',
                'I1',
                'I2',
                'J1',
                'J2',
                'K1',
                'K2',
                'FACE',
                'INFLUX_COEFFICIENT',
                'INFLUX_COEFFICIENT_MULTIPLIER',
                'ACTIVE_BLOCK_CONNECTIONS',
            ],
            dtypes=[
                'int',
                'int',
                'int',
                'int',
                'int',
                'int',
                'int',
                'text',
                'float',
                'float',
                'text',
            ],
        ),
        (SECTIONS.GRID, SECTIONS.SOLUTION),
    ),
    'HMMLCTAQ': KeywordSpecification(
        'HMMLCTAQ',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'NUMBER',
                'PERMEABILITY_MULTIPLIER',
                'INFLUENCE_ANGLE_MULTIPLIER',
                'AQUIFER_DEPTH_MULTIPLIER',
            ],
            dtypes=['int', 'float', 'float', 'float'],
        ),
        (SECTIONS.SOLUTION,),
    ),
    'BRANPROP': KeywordSpecification(
        'BRANPROP',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'DOWNTREE_NODE',
                'UPTREE_NODE',
                'VFP_TABLE_NUMBER',
                'ALQ',
            ],
            dtypes=['text', 'text', 'int', 'float'],
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'NODEPROP': KeywordSpecification(
        'NODEPROP',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'NAME',
                'PRESSURE',
                'UPTREE_NODE_CHOKE_FLAG',
                'GAS_LIFT_FLAG',
                'AUTOMATIC_CHOKE_GROUP',
                'GROUP',
            ],
            dtypes=['text', 'text', 'text', 'text', 'text', 'text'],
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'NWATREM': KeywordSpecification(
        'NWATREM',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=['NODE', 'WATER_REMOVAL_RATE', 'WATER_FLOW_FRACTION'],
            dtypes=['text', 'float', 'float'],
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'WTEST': KeywordSpecification(
        'WTEST',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'WELL',
                'TEST_INTERVAL',
                'CLOSURE_REASON',
                'TESTS_NUMBER',
                'STARTUP_TIME',
            ],
            dtypes=['text', 'float', 'text', 'int', 'float'],
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'NETCOMPA': KeywordSpecification(
        'NETCOMPA',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'INLET_NODE',
                'OUTLET_NODE',
                'GROUP',
                'PHASE',
                'VFP_TABLE',
                'ALQ',
                'GAS_CONSUMPTION_RATE',
                'EXTRACTED_GAS_CONSUMPTION_GROUP',
                'COMPRESSOR_TYPE',
                'COMPESSION_LEVELS_NUMBER',
                'ALQ_LEVEL1',
                'SWITCHING_SEQUENCE_NUMBER',
            ],
            dtypes=[
                'text',
                'text',
                'text',
                'text',
                'int',
                'float',
                'float',
                'text',
                'text',
                'int',
                'float',
                'int',
            ],
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'WEFAC': KeywordSpecification(
        'WEFAC',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'WELL',
                'WELL_EFFICIENCY_FACTOR',
                'INCLUDE_IN_EXTENDED_NETWORK_CALCULATIONS',
            ],
            dtypes=['text', 'float', 'text'],
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'WELLTRACK': KeywordSpecification(
        'WELLTRACK',
        DataTypes.NAMED_TABLE,
        TableSpecification(
            ['X', 'Y', 'Z', 'MD'],
            None,
            ['float', 'float', 'float', 'float'],
            header=None,
            number=1,
        ),
        (SECTIONS.SCHEDULE,),
    ),
    'MULTFLT': KeywordSpecification(
        'MULTFLT',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=['NAME', 'TRANSMISSIBILITY_MULT', 'THERMAL_MULT'],
            dtypes=('text', 'float', 'float'),
            terminated=True,
        ),
        (SECTIONS.GRID, SECTIONS.EDIT, SECTIONS.SCHEDULE),
    ),
    'EQUALS': KeywordSpecification(
        'EQUALS',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=(
                'ARR',
                'MULTIPLYER',
                'IMIN',
                'IMAX',
                'JMIN',
                'JMAX',
                'KMIN',
                'KMAX',
            ),
            dtypes=cast(list[DTypeString], ['text', 'float'] + ['int'] * 6),
            terminated=True,
        ),
        (
            SECTIONS.GRID,
            SECTIONS.EDIT,
            SECTIONS.REGIONS,
            SECTIONS.PROPS,
            SECTIONS.SOLUTION,
        ),
    ),
    'FLUXNUM': KeywordSpecification(
        'FLUXNUM', DataTypes.ARRAY, ArraySpecification(dtype=int), (SECTIONS.GRID,)
    ),
    'MULTREGT': KeywordSpecification(
        'MULTREGT',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=[
                'START_REGION',
                'END_REGION',
                'MULT',
                'DIRECTION',
                'NNC',
                'REGION_TYPE',
            ],
            dtypes=('int', 'int', 'float', 'text', 'text', 'text'),
            terminated=True,
        ),
        (SECTIONS.GRID, SECTIONS.EDIT, SECTIONS.SCHEDULE),
    ),
    'SWATINIT': KeywordSpecification(
        'SWATINIT', DataTypes.ARRAY, ArraySpecification(dtype=float), (SECTIONS.PROPS,)
    ),
    'TRACER': KeywordSpecification(
        'TRACER',
        DataTypes.STATEMENT_LIST,
        StatementSpecification(
            columns=['INDICATOR', 'FLUID'], dtypes=['text', 'text'], terminated=True
        ),
        (SECTIONS.PROPS,),
    ),
}


def _get_oilviscc_columns_factory(n_comp: IntType):
    def get_oilviscc_columns(header: pd.DataFrame) -> Sequence[str]:
        if header['TYPE'].values[0] == 'CORRELATION':
            if header['NAME'].values[0] == 'ASTM':
                return (
                    ['T1']
                    + [f'MU{i}_T1' for i in range(1, n_comp + 1)]
                    + ['T2']
                    + [f'MU{i}_T2' for i in range(1, n_comp + 1)]
                    + [f'MU{i}_TINF' for i in range(1, n_comp + 1)]
                )
            if header['NAME'].values[0] == 'ANDRADE':
                return (
                    ['T1']
                    + [f'MU{i}_T1' for i in range(1, n_comp + 1)]
                    + [f'MU{i}_TINF' for i in range(1, n_comp + 1)]
                )
            if header['NAME'].values[0] == 'VOGEL':
                return (
                    ['T1']
                    + [f'MU{i}_T1' for i in range(1, n_comp + 1)]
                    + ['T2']
                    + [f'MU{i}_T2' for i in range(1, n_comp + 1)]
                    + [f'MU{i}_TINF' for i in range(1, n_comp + 1)]
                )
            if header['NAME'].values[0] == 'LOG':
                return (
                    ['T1']
                    + [f'MU{i}_T1' for i in range(1, n_comp + 1)]
                    + ['T2']
                    + [f'MU{i}_T2' for i in range(1, n_comp + 1)]
                )
            raise ValueError(
                f'`NAME` should be `ASTM`, `ANDRADE`, `VOGEL`, `LOG` not `{header["NAME"].values[0]}`'
            )
        if header['TYPE'].values[0] == 'FORMULA':
            if header['NAME'].values[0] == 'ASTM':
                return (
                    [f'A{i}' for i in range(1, n_comp + 1)]
                    + [f'B{i}' for i in range(1, n_comp + 1)]
                    + [f'C{i}' for i in range(1, n_comp + 1)]
                )
            if header['NAME'].values[0] == 'ANDRADE':
                return [f'A{i}' for i in range(1, n_comp + 1)] + [
                    f'B{i}' for i in range(1, n_comp + 1)
                ]
            if header['NAME'].values[0] == 'VOGEL':
                return (
                    [f'A{i}' for i in range(1, n_comp + 1)]
                    + [f'B{i}' for i in range(1, n_comp + 1)]
                    + [f'C{i}' for i in range(1, n_comp + 1)]
                )
            if header['NAME'].values[0] == 'LOG':
                return [f'A{i}' for i in range(1, n_comp + 1)] + [
                    f'B{i}' for i in range(1, n_comp + 1)
                ]
            raise ValueError(
                f'`NAME` should be `ASTM`, `ANDRADE`, `VOGEL`, `LOG` not `{header["NAME"].values[0]}`'
            )
        raise ValueError(
            f'`TYPE` should be `FORMULA` or `CORRELATION` not `{header["TYPE"].values[0]}`'
        )

    return get_oilviscc_columns


def get_dynamic_keyword_specification(
    keyword: str, data: DataType
) -> KeywordSpecification:
    """
    Construct keyword specification.

    Parameters
    ----------
    keyword : str
        Keyword.
    data : DataType
        Loaded data.

    Returns
    -------
    KeywordSpecification
        Keyword Specification.

    """

    def _compositional_table(
        kw: str,
        column_name: str,
        data: DataType,
        number: Callable[[DataType | None], IntType] = _get_eos_regions_number,
    ):
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            kw,
            DataTypes.TABLE_SET,
            TableSpecification(
                [column_name + f'{i}' for i in range(1, n_comp + 1)],
                domain=None,
                number=number,
            ),
            (SECTIONS.PROPS,),
        )

    def _compositional_statement(
        kw: str, column_name: str, dtype: DTypeString, data: DataType
    ):
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')

        return KeywordSpecification(
            kw,
            DataTypes.SINGLE_STATEMENT,
            StatementSpecification(
                [column_name + f'{i}' for i in range(1, n_comp + 1)],
                [dtype for _ in range(n_comp)],
            ),
            (SECTIONS.PROPS,),
        )

    if keyword == 'ZMFVD':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        spec = KeywordSpecification(
            'ZMFVD',
            DataTypes.TABLE_SET,
            TableSpecification(
                ['DEPTH'] + [f'C{i}' for i in range(1, n_comp + 1)],
                [0],
                cast(Sequence[DTypeString], ['float'] * (n_comp + 1)),
                number=_get_eql_regions_number,
            ),
            (SECTIONS.PROPS,),
        )
        return spec
    if keyword == 'COORDSYS':
        n_res = None
        for d in data['RUNSPEC']:
            if d[0] == 'NUMRES':
                if not isinstance(d[1], pd.DataFrame):
                    raise ValueError('NUMRES value should be of type pandas.DataFrame.')
                n_res = d[1].N.values[0]  # pyright: ignore[reportAny]
                if not isinstance(n_res, int):
                    raise ValueError('N column in NUMRES value should be of type int.')

            if n_res is None:
                n_res = 1
            spec = KeywordSpecification(
                keyword,
                DataTypes.RECORDS,
                RecordsSpecification(
                    [
                        StatementSpecification(
                            [
                                'K1',
                                'K2',
                                'CIRCLE_COMPLETION',
                                'CONNECTION_BELOW',
                                'LATERAL_BLOCK_CONNECTION_LOWER_BOUND',
                                'LATERAL_BLOCK_CONNECTION_UPPER_BOUND',
                            ],
                            ['int', 'int', 'text', 'text', 'int', 'int'],
                        ),
                    ]
                    * n_res
                ),
                (SECTIONS.GRID,),
            )
            return spec
    if keyword == 'COMPVD':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            'COMPVD',
            DataTypes.TABLE_SET,
            TableSpecification(
                ['DEPTH']
                + [f'Z{i}' for i in range(1, n_comp + 1)]
                + ['LIQUID_FLAG']
                + ['P_SAT'],
                domain=[0],
                dtypes=cast(
                    Sequence[DTypeString],
                    ['float'] * (n_comp + 1) + ['int', 'float'],
                ),
                number=_get_eql_regions_number,
            ),
            (SECTIONS.PROPS,),
        )
    if keyword == 'GPTABLEN':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            'GPTABLEN',
            DataTypes.TABLE_SET,
            TableSpecification(
                ['C_HEAVY']
                + [f'OIL_RECOVERY_FRACTION{i}' for i in range(1, n_comp + 1)]
                + [f'NGL_RECOVERY_FRACTION{i}' for i in range(1, n_comp + 1)],
                domain=[0],
                header=StatementSpecification(
                    ['GPTABLE_NUM', 'HEAVY_C1', 'HEAVY_CLAST'], ['int', 'int', 'int']
                ),
                number=_get_gptable_number,
            ),
            (SECTIONS.SOLUTION, SECTIONS.SCHEDULE),
        )
    if keyword == 'PCRIT':
        return _compositional_table('PCRIT', 'P', data)
    if keyword == 'VCRIT':
        return _compositional_table('VCRIT', 'V', data)
    if keyword == 'ZCRIT':
        return _compositional_table('ZCRIT', 'Z', data)
    if keyword == 'VCRITVIS':
        return _compositional_table('VCRITVIS', 'V', data)
    if keyword == 'ZCRITVIS':
        return _compositional_table('ZCRITVIS', 'Z', data)
    if keyword == 'MW':
        return _compositional_table('MW', 'MW', data)
    if keyword == 'ACF':
        return _compositional_table('ACF', 'ACF', data)
    if keyword == 'TCRIT':
        return _compositional_table('TCRIT', 'T', data)
    if keyword == 'DREF':
        return _compositional_statement('DREF', 'D', 'float', data)
    if keyword == 'PREF':
        return _compositional_statement('PREF', 'P', 'float', data)
    if keyword == 'TREF':
        return _compositional_statement('TREF', 'T', 'float', data)
    if keyword == 'CREF':
        return _compositional_statement('CREF', 'C', 'float', data)
    if keyword == 'THERMEX1':
        return _compositional_statement('THERMEX1', 'X', 'float', data)
    if keyword == 'SPECHA':
        return _compositional_table('SPECHA', 'X', data)
    if keyword == 'SPECHB':
        return _compositional_table('SPECHB', 'X', data)
    if keyword == 'SPECHG':
        return _compositional_table('SPECHG', 'X', data)
    if keyword == 'SPECHS':
        return _compositional_table('SPECHG', 'X', data)
    if keyword == 'CVTYPE':
        return _compositional_statement('CVTYPE', 'T', 'text', data)
    if keyword == 'SDREF':
        return _compositional_table('SDREF', 'SD', data)
    if keyword == 'ZFACTOR':
        return _compositional_table('ZFACTOR', 'ZC', data)
    if keyword == 'HEATVAP':
        return _compositional_table('HEATVAP', 'AC', data)
    if keyword == 'GASVISCF':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            'GASVISCF',
            DataTypes.TABLE_SET,
            TableSpecification(
                [
                    f'{c}{i}'
                    for c, i in itertools.product(['A', 'B'], range(1, n_comp + 1))
                ],
                domain=None,
                number=_get_eos_regions_number,
            ),
            (SECTIONS.PROPS,),
        )
    if keyword == 'GASVISCT':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            'GASVISCT',
            DataTypes.TABLE_SET,
            TableSpecification(
                ['t', *[f'CP{i}' for i in range(1, n_comp + 1)]],
                domain=[0],
                number=_get_pvt_regions_number,
            ),
            (SECTIONS.PROPS,),
        )

    if keyword == 'KVCR':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            'KVCR',
            DataTypes.TABLE_SET,
            TableSpecification(
                [
                    f'{c}{i}'
                    for c, i in itertools.product(
                        ['A', 'B', 'C', 'D', 'E'], range(1, n_comp + 1)
                    )
                ],
                domain=None,
                number=_get_eos_regions_number,
            ),
            (SECTIONS.PROPS,),
        )
    if keyword == 'BIC':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        columns: list[str] = []
        for i in range(2, n_comp + 1):
            for j in range(1, i):
                columns.append(f'BIC{i}_{j}')
        return KeywordSpecification(
            'BIC',
            DataTypes.TABLE_SET,
            TableSpecification(columns, domain=None, number=_get_eos_regions_number),
            (SECTIONS.PROPS,),
        )
    if keyword == 'WELLSTRE':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            'WELLSTRE',
            DataTypes.STATEMENT_LIST,
            StatementSpecification(
                ['STREAM'] + [f'X{i}' for i in range(1, n_comp + 1)],
                cast(Sequence[DTypeString], ['text'] + ['float'] * n_comp),
            ),
            (SECTIONS.SCHEDULE,),
        )
    if keyword == 'OILVISCC':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            'OILVISCC',
            DataTypes.TABLE_SET,
            TableSpecification(
                _get_oilviscc_columns_factory(n_comp),
                None,
                header=StatementSpecification(['NAME', 'TYPE'], ['text', 'text']),
                number=_get_eos_regions_number,
            ),
            (SECTIONS.PROPS,),
        )
    for kw, c, t in zip(
        ('STOREAC', 'STOPROD', 'REACPHA', 'REACCORD'),
        ('C', 'C', 'P', 'O'),
        ('float', 'float', 'text', 'int'),
    ):
        if keyword == kw:
            n_comp = _get_ncomp(data)
            if n_comp is None:
                raise ValueError('No number of components information in `data`.')
            return KeywordSpecification(
                keyword,
                DataTypes.TABLE_SET,
                TableSpecification(
                    columns=[f'{t}{i}' for i in range(1, n_comp + 1)] + [f'{t}W'],
                    domain=None,
                    number=_get_reaction_number,
                    dtypes=cast(Sequence[DTypeString], [t] * (n_comp + 1)),
                ),
                (SECTIONS.PROPS,),
            )
    if keyword == 'REACPHA':
        n_comp = _get_ncomp(data)
        if n_comp is None:
            raise ValueError('No number of components information in `data`.')
        return KeywordSpecification(
            keyword,
            DataTypes.TABLE_SET,
            TableSpecification(
                columns=[f'P{i}' for i in range(1, n_comp + 1)],
                domain=None,
                number=_get_reaction_number,
                dtypes=cast(Sequence[DTypeString], ['text'] * n_comp),
            ),
            (SECTIONS.PROPS,),
        )
    if keyword == 'ROCKTAB':
        for key, _ in data['PROPS']:
            if key == 'RKTRMDIR':
                return KeywordSpecification(
                    'ROCKTAB',
                    DataTypes.TABLE_SET,
                    TableSpecification(
                        (
                            'PRESSURE',
                            'POROSITY',
                            'TRANSMISSIBILITY_MULTIPLIER_X',
                            'TRANSMISSIBILITY_MULTIPLIER_Y',
                            'TRANSMISSIBILITY_MULTIPLIER_Z',
                        ),
                        [0],
                        ('float', 'float', 'float', 'float', 'float'),
                        number=_get_rock_regions_number,
                    ),
                    (SECTIONS.PROPS,),
                )

        return KeywordSpecification(
            'ROCKTAB',
            DataTypes.TABLE_SET,
            TableSpecification(
                (
                    'PRESSURE',
                    'POROSITY',
                    'TRANSMISSIBILITY_MULTIPLIER',
                ),
                [0],
                (
                    'float',
                    'float',
                    'float',
                ),
                number=_get_rock_regions_number,
            ),
            (SECTIONS.PROPS,),
        )
    for kw, c in zip(('REACRATE', 'REACACT', 'REACENTH'), ('R', 'E', 'E')):
        if kw == keyword:
            n_react = _get_reaction_number(data)
            return KeywordSpecification(
                keyword,
                DataTypes.SINGLE_STATEMENT,
                StatementSpecification(
                    columns=[f'{c}{i}' for i in range(1, n_react + 1)],
                    dtypes=cast(Sequence[DTypeString], ['float'] * n_react),
                ),
                (SECTIONS.PROPS,),
            )
    else:
        raise ValueError(f'Specification can not be defined for keyword {keyword}.')


def _get_ncomp(data: DataType) -> IntType | None:
    """Get number of fluid components."""
    n_comp = None
    if 'RUNSPEC' in data:
        for d in data['RUNSPEC']:
            if d[0] == 'COMPS':
                if not isinstance(d[1], pd.DataFrame):
                    raise ValueError('COMPS value should be pandas.DataFrame.')
                n_comp = d[1].N.values[0]  # pyright: ignore[reportAny]
                if not isinstance(n_comp, IntType):
                    raise ValueError(
                        'N column in the "NCOMPS" should be of integer type.'
                    )
    if n_comp is not None:
        return n_comp  # pyright: ignore[reportUnknownVariableType]
    if 'PROPS' in data:
        for d in data['PROPS']:
            if d[0] == 'NCOMPS':
                if not isinstance(d[1], pd.DataFrame):
                    raise ValueError('NCOMPS value should be pandas.DataFrame.')
                n_comp = d[1].N.values[0]  # pyright: ignore[reportAny]
                if not isinstance(n_comp, IntType):
                    raise ValueError(
                        'N column in the "NCOMPS" should be of integer type.'
                    )
    return n_comp  # pyright: ignore[reportUnknownVariableType]
