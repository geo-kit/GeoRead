"""Utils for dumping keywords data."""

from collections.abc import Callable, Sequence
from contextlib import ExitStack
import copy
import numbers
import pathlib
from typing import Any, Protocol, cast
import numpy as np
import numpy.typing as npt
import pandas as pd
from ._data_directory import (
    INT_NAN,
    ArraySpecification,
    ArrayWithUnits,
    DataType,
    DataTypes,
    DATA_DIRECTORY,
    IntType,
    KeywordSpecification,
    NoDataSpecification,
    ObjectSpecification,
    ParametersSpecification,
    StringSpecification,
    TableSpecification,
    ValueType,
    get_dynamic_keyword_specification,
)

MAX_STRLEN = 40

INPLACE_ARRAYS = ['TSTEP']


class PWriteBuf(Protocol):
    """Protocol for writable buffer."""

    def write(self, s: str, /) -> int | None:
        """
        Write string to buffer.

        Parameters
        ----------
        s : str
            String to write.

        Returns
        -------
        int | None

        """
        pass


def format_string_val(
    val: pd.Timestamp | str,
    keyword_spec: StringSpecification | None | ObjectSpecification,
) -> str:
    """
    Format dates.

    Parameters
    ----------
    val : pd.Timestamp | str
        Value to format.
    keyword_spec : StringSpecification | None | ObjectSpecification
        Keyword specification.

    Returns
    -------
    str
        Formatted string.

    """
    if keyword_spec is not None and keyword_spec.date:
        if not isinstance(val, pd.Timestamp):
            raise ValueError('`val` should be of type pandas.Timestamp.')
        d = val.strftime('%d %b %Y').upper()
        if val.hour or val.minute or val.second:
            t = val.strftime('%H:%M:%S')
            return ' '.join((d, t))
        return d
    if not isinstance(val, str):
        raise ValueError('Value should be of type str.')
    return val


def _dump_string(keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf):
    if not (isinstance(val, str) or isinstance(val, pd.Timestamp)):
        raise ValueError('`val` should be of type `str` or `pandas.Timestamp`.')
    spec = keyword_spec.specification
    if not (isinstance(spec, StringSpecification) or spec is None):
        raise ValueError(
            '`keyword_spec.specification` should be of type StringSpecification.'
        )
    _ = buf.write('\n'.join([keyword_spec.keyword, format_string_val(val, spec), '/']))


def _dump_array(
    keyword_spec: KeywordSpecification,
    val: ValueType,
    buf: PWriteBuf,
    include_dir: pathlib.Path | None,
):
    if keyword_spec.keyword in INPLACE_ARRAYS:
        inplace = True
    else:
        inplace = False

    if not isinstance(val, np.ndarray):
        raise ValueError('`val` should be of type `numpy.ndarray`.')

    spec = keyword_spec.specification
    if not isinstance(spec, ArraySpecification):
        raise ValueError(
            '`keyword_spec.specification` should be of type ArraySpecification.'
        )

    if spec.dtype in (bool, int):
        fmt = '%d'
    else:
        fmt = '%f'
    if inplace:
        _dump_array_ascii(buf, val.reshape(-1), header=keyword_spec.keyword, fmt=fmt)
        _ = buf.write('/')
        return
    if include_dir is None:
        raise ValueError(
            '`include_dir` should be provided if array is not supposed to be dumped inplace.'
        )

    with open(include_dir / f'{keyword_spec.keyword}.inc', 'w') as inc_buf:
        _dump_array_ascii(
            inc_buf, val.reshape(-1), fmt=fmt, header=keyword_spec.keyword
        )
        _ = inc_buf.write('/')
    _ = buf.write(
        '\n'.join(
            (
                'INCLUDE',
                '"' + '/'.join((include_dir.name, f'{keyword_spec.keyword}.inc')) + '"',
            )
        )
    )
    _ = buf.write('\n/')


def _dump_table(keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf):
    if not isinstance(val, Sequence):
        raise ValueError('`val` should be of type `Sequence[pandas.DataFrame]`')

    _ = buf.write(keyword_spec.keyword)
    spec = keyword_spec.specification
    if not isinstance(spec, TableSpecification):
        raise ValueError(
            '`keyword_spec.specification` should be of type TableSpecification.'
        )

    domain = spec.domain
    for table in val:
        if spec.header:
            if not isinstance(table, Sequence):
                raise ValueError(
                    '`table` should be of type `Sequence[pandas.DataFrame]`'
                )
            header = table[1]
            data = table[0]
        else:
            header = None
            data = table
        if not isinstance(data, pd.DataFrame):
            raise ValueError('`val` should be of type `Sequence[pandas.DataFrame]`')
        domain_val = domain
        if header is not None:
            _ = buf.write('\n')

            if not isinstance(header, pd.DataFrame):
                raise ValueError('`header` should be of type `pandas.DataFrame.`')
            _dump_statement(header, buf, closing_slash=False, new_line=False)
        if domain_val is not None and len(domain_val) == 2:
            _dump_multitable(data, buf)
            continue
        _ = buf.write('\n')
        row_iterator = (
            data.itertuples() if domain is not None else data.itertuples(index=False)
        )
        for row in row_iterator:
            vals = list(row)
            vals = [_nan_to_none(v) for v in vals]  # pyright: ignore[reportAny]
            str_representaions = [
                _string_representation(v) if v is not None else '' for v in vals
            ]
            str_representaions = _replace_empty_vals(str_representaions)
            _ = buf.write('\t'.join([v for v in str_representaions]) + '\n')
        _ = buf.write('/')


def _dump_multitable(val: pd.DataFrame, buf: PWriteBuf):
    _ = buf.write('\n')
    for _, df in val.groupby(level=0):
        for i, (ind1, row) in enumerate(df.iterrows()):
            vals = row.values.tolist()
            if not isinstance(ind1, Sequence):
                raise ValueError('`val` should have multiindex.')
            if i == 0:
                vals = [*ind1] + vals
            else:
                vals = [ind1[1]] + vals
            vals = [_nan_to_none(v) for v in vals]  # pyright: ignore[reportArgumentType]
            str_representations = [
                _string_representation(v) if v is not None else '' for v in vals
            ]
            str_representations = _replace_empty_vals(str_representations)
            if i != 0:
                str_representations = [''] + str_representations
            if i == len(df) - 1:
                str_representations = str_representations + ['/']
            _ = buf.write('\t'.join(str_representations) + '\n')
    _ = buf.write('/')


def _dump_single_statement(
    keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf
):
    _ = buf.write(keyword_spec.keyword + '\n')
    if not isinstance(val, pd.DataFrame):
        raise ValueError('`val` should be of type `pandas.DataFrame`.')
    _dump_statement(val, buf, closing_slash=False)
    _ = buf.write('/')


def _dump_statement_list(
    keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf
):
    if not isinstance(val, pd.DataFrame):
        raise ValueError('`val` should be of type `pd.DataFrame`.')
    _ = buf.write(keyword_spec.keyword + '\n')
    for row in val.itertuples(index=False):
        _dump_statement(row, buf, closing_slash=True)
    _ = buf.write('/')


def _dump_records(keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf):
    _ = buf.write(keyword_spec.keyword + '\n')
    if not isinstance(val, Sequence):
        raise ValueError('`val` should be of type Sequence[pandas.DataFrame].')
    for v in val:
        if not isinstance(v, (pd.DataFrame, np.ndarray)):
            raise ValueError('`val` should be of type Sequence[pandas.DataFrame].')
        _dump_statement(v, buf, closing_slash=True)


def _dump_object_list(
    keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf
):
    _ = buf.write(keyword_spec.keyword + '\n')
    spec = keyword_spec.specification
    if not (isinstance(spec, ObjectSpecification) or spec is None):
        raise ValueError(
            '`keyword_spec.specification` should be of type `ObjectSpecification`.'
        )
    if not isinstance(val, Sequence):
        raise ValueError('`val` should be of type Sequence[str | pd.Timestamp]')

    for o in val:
        if not (isinstance(o, str) or isinstance(o, pd.Timestamp)):
            raise ValueError('`val` should be of type Sequence[str | pd.Timestamp]')
        _ = buf.write(f'{format_string_val(o, spec)}')
        if spec is not None and spec.terminated:
            _ = buf.write(' /')
        _ = buf.write('\n')
    _ = buf.write('/')


def _dump_parameters(
    keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf
):
    spec = keyword_spec.specification
    if not isinstance(spec, ParametersSpecification):
        raise ValueError(
            '`keyword_spec.specification` should be of type ParametersSpecification.'
        )
    if not isinstance(val, dict):
        raise ValueError

    if not all(isinstance(v, str) or v is None for v in val.values()):
        raise ValueError('`val` should be of type dict[str, str|None]')

    val = cast(dict[str, str | None], val)

    if spec.tabulated:
        return _dump_tabulated_parameters(keyword_spec, val, buf)

    _ = buf.write(keyword_spec.keyword + '\n')
    res = ' '.join([f'{k}' if v is None else f'{k}={v}' for k, v in val.items()])
    _ = buf.write(res)
    _ = buf.write('\n/')


def _dump_tabulated_parameters(
    keyword_spec: KeywordSpecification, val: dict[str, str | None], buf: PWriteBuf
):
    _ = buf.write(keyword_spec.keyword + '\n')
    for key, data in val.items():
        if not isinstance(data, str):
            raise ValueError('values of `val` should be strings.')
        _ = buf.write('\t'.join((key, data)) + '\n')
    _ = buf.write('/')


def _dump_array_with_units(
    keyword_spec: KeywordSpecification, val: ValueType, buf: PWriteBuf
):
    _ = buf.write(keyword_spec.keyword + '\n')
    if not isinstance(val, ArrayWithUnits):
        raise ValueError('`val` should be of type `ArrayWithUnits`')
    _ = buf.write(val.units + '\n')
    spec = keyword_spec.specification
    if not isinstance(spec, ArraySpecification):
        raise ValueError(
            '`keyword_spec.specification` should be of type `ArraySpecification`.'
        )
    if spec.dtype in (bool, int):
        fmt = '%d'
    else:
        fmt = '%g'
    _dump_array_ascii(buf, val.data.reshape(-1), fmt=fmt)
    _ = buf.write('/')


def _dump_no_data(
    keyword_spec: KeywordSpecification,
    buf: PWriteBuf,
):
    _ = buf.write(f'{keyword_spec.keyword}')
    spec = keyword_spec.specification
    if spec is not None and not isinstance(spec, NoDataSpecification):
        raise ValueError(
            '`keyword_spec.specification` should be None or of type `NoDataSpecification`.'
        )
    if spec is not None and spec.terminated:
        _ = buf.write('\n/')


DUMP_ROUTINES: dict[
    DataTypes | None,
    Callable[[KeywordSpecification, ValueType, PWriteBuf, pathlib.Path | None], None],
] = {
    DataTypes.OBJECT_LIST: lambda keyword_spec, val, buf, _: _dump_object_list(
        keyword_spec, val, buf
    ),
    DataTypes.STRING: lambda keyword_spec, val, buf, _: _dump_string(
        keyword_spec, val, buf
    ),
    DataTypes.STATEMENT_LIST: lambda keyword_spec, val, buf, _: _dump_statement_list(
        keyword_spec, val, buf
    ),
    DataTypes.PARAMETERS: lambda keyword_spec, val, buf, _: _dump_parameters(
        keyword_spec, val, buf
    ),
    DataTypes.ARRAY: _dump_array,
    DataTypes.TABLE_SET: lambda keyword_spec, val, buf, _=None: _dump_table(
        keyword_spec, val, buf
    ),
    None: lambda keyword_spec, _, buf, ___: _dump_no_data(keyword_spec, buf),
    DataTypes.SINGLE_STATEMENT: lambda keyword_spec, val, buf, _: (
        _dump_single_statement(keyword_spec, val, buf)
    ),
    DataTypes.RECORDS: lambda keyword_spec, val, buf, _: _dump_records(
        keyword_spec, val, buf
    ),
    DataTypes.ARRAY_WITH_UNITS: lambda keyword_spec, val, buf, _: (
        _dump_array_with_units(keyword_spec, val, buf)
    ),
}


def _dump_statement(
    val: pd.DataFrame | pd.Series | np.ndarray | tuple[Any, ...],  # pyright: ignore[reportExplicitAny]
    buf: PWriteBuf,
    closing_slash: bool = True,
    new_line: bool = True,
):
    if isinstance(val, pd.DataFrame):
        if val.shape[0] != 1:
            raise ValueError('Val shoud have exactly one row.')
        vals = [val[col][0] for col in val.columns]
    elif isinstance(val, pd.Series):
        vals = val.values
    else:
        vals = val
    vals = [_nan_to_none(v) for v in vals]  # pyright: ignore[reportAny]
    str_representaions = [
        _string_representation(v) if v is not None else '' for v in vals
    ]
    str_representaions = _replace_empty_vals(str_representaions)
    if len(str_representaions) == 0:
        str_representaions.append('*')
    result = '\t'.join(str_representaions)
    nl = '\n' if new_line else ''

    result += nl if not closing_slash else '/' + nl
    _ = buf.write(result)


def _string_representation(v: Any):  # pyright: ignore[reportAny, reportExplicitAny]
    if v is None:
        return ''
    if isinstance(v, numbers.Number):
        r = str(v)
        if 'e' in r:
            assert isinstance(v, (float, np.floating))
            return np.format_float_scientific(
                v, unique=True, trim='-', exp_digits=1
            ).upper()
        return r
    if isinstance(v, str):
        if any(symbol in v for symbol in ' \t'):
            return "'" + v + "'"
    return str(v)


def _replace_empty_vals(vals: Sequence[str]) -> list[str]:
    vals = copy.copy(vals)
    vals = list(vals)
    while True:
        start = None
        end = None
        for i, s in enumerate(vals):
            if s != '':
                if start is not None:
                    break
            else:
                if start is None:
                    start = i
                    end = i
                else:
                    end = i
        if start is None:
            break
        else:
            assert end is not None
            if end == len(vals) - 1:
                del vals[start:]
                continue
            if start == end:
                replacement = '*'
            else:
                replacement = f'{end - start + 1}*'
            vals[start : end + 1] = [replacement]
    return vals


def _nan_to_none(val: IntType | np.floating | str):
    if isinstance(val, (numbers.Number)) and np.isnan(val):
        return None
    if val == INT_NAN:
        return None
    if val == '':
        return None
    return val


def dump(
    data: DataType,
    path: pathlib.Path,
    inplace_scedule: bool = False,
    filename: str | None = None,
) -> None:
    """
    Dump model data.

    Parameters
    ----------
    data : DataType
        Model data.
    path : pathlib.Path
        Path to dump the model.
    inplace_scedule : bool, default False.
        Should schedule be dumped inplace.
    filename : str | None, default None.
        Name of the main model file, if None filename is taken from `TITLE` field in the `RUNSPEC`.

    """
    if not path.exists():
        path.mkdir()

    include_dir = path / 'include'

    if not include_dir.exists():
        include_dir.mkdir()

    if filename is None:
        for key, val in data['RUNSPEC']:
            if key == 'TITLE':
                filename = f'{val}.data'

    if filename is None:
        raise ValueError(
            'Filename is not specified and no TITLE keyword in RUNSPEC section.'
        )

    with ExitStack() as stack:
        buf = stack.enter_context(open(path / filename, 'w'))
        for section in (
            '',
            'RUNSPEC',
            'GRID',
            'EDIT',
            'PROPS',
            'REGIONS',
            'SOLUTION',
            'SUMMARY',
            'SCHEDULE',
        ):
            if section in data:
                if section != '':
                    _ = buf.write(f'{section}\n\n')
                if section == 'SCHEDULE' and not inplace_scedule:
                    schedule_path = include_dir / 'schedule.inc'
                    _ = buf.write('INCLUDE\n')
                    _ = buf.write(('"' + str(schedule_path.relative_to(path))) + '"')
                    _ = buf.write('\n/\n\n')
                    buf_tmp = stack.enter_context(open(schedule_path, 'w'))
                else:
                    buf_tmp = buf
                for key, val in data[section]:
                    spec = DATA_DIRECTORY[key]
                    if spec is None:
                        spec = get_dynamic_keyword_specification(key, data)
                    DUMP_ROUTINES[spec.type](spec, val, buf_tmp, include_dir)
                    _ = buf_tmp.write('\n\n')


def _dump_array_ascii(
    buffer: PWriteBuf,
    array: npt.NDArray[np.floating | np.integer | np.bool_],
    header: str | None = None,
    fmt: str = '%f',
    compressed: bool = True,
):
    """
    Write array-like data into an ASCII buffer.

    Parameters
    ----------
    buffer : PWriteBuf
        Destination buffer.
    array : 1d, array-like
        Array to be saved.
    header : str, optional
        String to be written line before the array.
    fmt : str or sequence of strs, optional
        Format to be passed into ``numpy.savetxt`` function. Default to '%f'.
    compressed : bool
        If True, uses compressed typing style

    """
    if header is not None:
        _ = buffer.write(header + '\n')

    if compressed:
        i = 0
        items_written = 0
        while i < len(array):
            count = 1
            while (i + count < len(array)) and (array[i + count] == array[i]):
                count += 1
            if count <= 4:
                _ = buffer.write(' '.join([fmt % array[i]] * count))
                items_written += count
            else:
                _ = buffer.write(''.join((str(count), '*', fmt % array[i])))
                items_written += 1
            i += count
            if items_written > MAX_STRLEN:
                _ = buffer.write('\n')
                items_written = 0
            elif i < len(array):
                _ = buffer.write(' ')
        _ = buffer.write('\n')
    else:
        for i in range(0, len(array), MAX_STRLEN):
            _ = buffer.write(' '.join([fmt % d for d in array[i : i + MAX_STRLEN]]))  # pyright: ignore[reportAny]
            _ = buffer.write('\n')
        _ = buffer.write('\n')
