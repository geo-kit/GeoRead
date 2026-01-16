"""Utils for data loading."""

from collections.abc import Callable, Iterator, Sequence, Generator
from contextlib import ExitStack
import copy
import itertools
import logging
import pathlib
import shlex
from types import TracebackType
from typing import Literal, Protocol, Self, TextIO, cast
import uuid
import re
import warnings
import chardet

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._data_directory import (
    DATA_DIRECTORY,
    INT_NAN,
    SECTIONS,
    ArraySpecification,
    ArrayWithUnits,
    DTypeString,
    DataType,
    DataTypes,
    IntType,
    NoDataSpecification,
    NpArray,
    ObjectSpecification,
    ParametersSpecification,
    RecordValueType,
    RecordsSpecification,
    SpecificationType,
    StatementSpecification,
    StringSpecification,
    TableSpecification,
    ValueType,
    get_dynamic_keyword_specification,
)


DEFAULT_ENCODINGS = ['utf-8', 'cp1251']


class PReadBuf(Protocol):
    """Protocol for readable buffer."""

    def __iter__(self) -> Iterator[str]:
        """
        Get sting iterator.

        Returns
        -------
        Iterator[str]
            Iterator.

        """
        ...

    def __next__(self) -> str:
        """
        Next string.

        Returns
        -------
        str
            string.

        """
        ...

    def prev(self) -> Self:
        """
        Move cursor to the previous string.

        Returns
        -------
        Self
            self.

        """
        ...


def _load_string(
    keyword_spec: SpecificationType | None, buf: PReadBuf
) -> str | pd.Timestamp:
    line = next(buf)
    if "'" in line:
        split = re.split(r"'(.*)'", line)
    elif '"' in line:
        split = re.split(r'"(.*)"', line)
    else:
        split = line.split('/')
        if len(split) > 1:
            split = ['', split[0], *['/' + s for s in split[1:]]]
        else:
            split = ['', split[0], '']

    terminated = False
    for i, s in enumerate(split):
        if i % 2 == 0:
            if '/' in s:
                terminated = True
                break
    if terminated:
        val = ''.join(split[:i] + [split[i].split('/')[0]])  # pyright: ignore[reportPossiblyUnboundVariable]
    else:
        val = line
        line = next(buf)
        if not line.startswith('/'):
            warnings.warn('Data was not properly terminated.')
            _ = buf.prev()
    val = re.sub(r'"(.*?)"', r'\1', val)
    val = re.sub(r'\'(.*?)\'', r'\1', val)
    val = val.strip()
    if keyword_spec is None:
        return val
    if not isinstance(keyword_spec, StringSpecification):
        raise ValueError(
            '`keyword_spec` should be None or of type StringSpecification.'
        )
    if keyword_spec.date:
        return _parse_date(val)
    return val


def _load_object_list(
    keyword_spec: SpecificationType | None, buf: PReadBuf
) -> list[str] | list[pd.Timestamp]:
    if not isinstance(keyword_spec, ObjectSpecification | None):
        raise ValueError('`keyword_spec` should be of type keyword_spec or None.')
    if keyword_spec is not None:
        terminated = keyword_spec.terminated
        is_dates = keyword_spec.date
    else:
        terminated = False
        is_dates = False
    res: list[str] | list[pd.Timestamp] = []
    while True:
        line = _get_expected_line(buf)
        split = line.split('/')
        val = split[0].strip(' \t\n\'""')
        if terminated and len(split) == 1:
            raise ValueError(f'Line "{line}" is not teminated with "/"')
        if val:
            if is_dates:
                for v in res:
                    if not isinstance(v, pd.Timestamp):
                        raise ValueError('`res` should be of type List[pd.Timestamp].')
                res = cast(list[pd.Timestamp], res)
                res.append(_parse_date(val))
            else:
                for v in res:
                    if not isinstance(v, str):
                        raise ValueError('`res` should be of type List[str].')
                res = cast(list[str], res)
                res.append(val)

        else:
            if len(split) == 1:
                raise ValueError('Object specification expected.')
        if len(split) > 1 and not terminated:
            break
        if line.startswith('/'):
            break
    return res


def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s)


def _load_table(
    keyword_spec: SpecificationType, buf: PReadBuf, data: DataType | None = None
) -> Sequence[pd.DataFrame] | Sequence[tuple[pd.DataFrame, pd.DataFrame]]:
    def _parse_val(val: str | int | float, t: Literal['int', 'float', 'text']):
        if t == 'int':
            if val == 'nan':
                return INT_NAN
            return int(val)
        if t == 'float':
            return float(val)
        return val

    def _empty_val(t: Literal['int', 'float', 'text']):
        if t == 'int':
            return INT_NAN
        if t == 'float':
            return np.nan
        else:
            raise ValueError('`t` should be `int` or `float`.')

    if not isinstance(keyword_spec, TableSpecification):
        raise ValueError('`keyword_spec` should be of type TableSpecification.')

    domain = keyword_spec.domain
    if callable(domain):
        domain = domain
    if domain is None:
        depth = 1
    else:
        depth = len(domain)
    if isinstance(keyword_spec.number, IntType):
        n = keyword_spec.number
    else:
        n = keyword_spec.number(data)
    table_data = _read_table_data(buf, depth, n)
    tables: list[pd.DataFrame] | list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for region_table_data in table_data:
        header = None
        if keyword_spec.header is not None:
            n_header = len(keyword_spec.header.columns)
            header_data = region_table_data[:n_header]
            header_data = [
                _parse_val(v, t)
                for v, t in zip(header_data, keyword_spec.header.dtypes)
            ]
            header = pd.DataFrame(
                [header_data], columns=list(keyword_spec.header.columns)
            )
            region_table_data = region_table_data[n_header:]
        if callable(keyword_spec.columns):
            if header is None:
                raise ValueError('`header` should be not None.')
            columns = keyword_spec.columns(header)
        else:
            columns = keyword_spec.columns
        n_attrs = len(columns)
        dtypes = keyword_spec.dtypes
        if isinstance(dtypes, str):
            dtypes = [keyword_spec.dtypes] * n_attrs
        dtypes = cast(Sequence[DTypeString], dtypes)
        if depth == 2:
            table_parts: list[list[float | int | str]] = []
            for d in region_table_data:
                n_rows = (len(d) - 1) / (n_attrs - 1)
                if not n_rows.is_integer():
                    raise ValueError(
                        'Number of element is not aligned with the number of attributes.'
                    )
                data_tmp: list[list[float | int | str]] = []
                for i in range(int(n_rows)):
                    data_tmp.append(
                        [_parse_val(d[0], dtypes[0])]
                        + [
                            _parse_val(v, t)
                            for v, t in zip(
                                d[i * (n_attrs - 1) + 1 : (i + 1) * (n_attrs - 1) + 1],
                                dtypes[1:],
                            )
                        ]
                    )
                table_parts += data_tmp
            table = pd.DataFrame(table_parts, columns=list(columns))
        else:
            if len(region_table_data) < n_attrs:
                tmp = [_empty_val(t) for t in dtypes[len(region_table_data) :]]
                region_table_data += tmp
            if len(region_table_data) % n_attrs > 0:
                raise ValueError(
                    'Number of values in table is not consistent wit number of columns.'
                )
            n_rows = int(len(region_table_data) / n_attrs)
            data_tmp = [
                list(
                    map(
                        _parse_val,
                        region_table_data[i * n_attrs : (i + 1) * n_attrs],
                        dtypes,
                    )
                )
                for i in range(n_rows)
            ]
            table = pd.DataFrame(data_tmp, columns=list(columns))
        if 'int' in dtypes:
            int_columns = [col for col, t in zip(columns, dtypes) if t == 'int']
            for col in int_columns:
                table[col] = table[col].fillna(INT_NAN)
                if (np.mod(table[col], 1) > 0).any():
                    raise ValueError('Noninteger value in integer column.')
                table[col] = table[col].astype(int)
        if keyword_spec.domain is not None:
            domain_attrs = [columns[i] for i in keyword_spec.domain]
            table = table.set_index(domain_attrs)
        if header is not None:
            for t in tables:
                if not isinstance(t, tuple):
                    raise ValueError(
                        '`tables` should be of type list[tuple[pd.DataFrame, pd.DataFrame]].'
                    )
            tables = cast(list[tuple[pd.DataFrame, pd.DataFrame]], tables)
            tables.append((table, header))
        else:
            for t in tables:
                if not isinstance(t, pd.DataFrame):
                    raise ValueError('`tables` should be of type list[pd.DataFrame].')
            tables = cast(
                list[pd.DataFrame], tables
            )  # # pyright: ignore[reportUnnecessaryCast]

            tables.append(table)
    return tables


def _load_single_statement(keyword_spec: SpecificationType, buffer: PReadBuf):
    if not isinstance(keyword_spec, StatementSpecification):
        raise ValueError('`keyword_spec` should be of type `StatementSpecification`.')

    columns = keyword_spec.columns
    column_types = keyword_spec.dtypes
    shift = 0
    full: list[str | None] = [None] * len(columns)
    while True:
        line = _get_expected_line(buffer)
        split = line.split('/')
        line = split[0].strip()
        vals = shlex.split(line)
        full, shift = parse_vals(full, shift, vals)
        if len(split) > 1:
            break
    df = pd.DataFrame(dict(zip(columns, full)), index=[0])
    if 'text' in column_types:
        text_columns = [col for col, dt in zip(columns, column_types) if dt == 'text']
        df[text_columns] = df[text_columns].map(
            cast(
                Callable[
                    [
                        str,
                    ],
                    str,
                ],
                lambda x: x.strip('\'"') if x is not None else x,
            )  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType]
        )
    if 'float' in column_types:
        float_columns = [col for col, dt in zip(columns, column_types) if dt == 'float']
        df[float_columns] = df[float_columns].astype(float)
    if 'int' in column_types:
        int_columns = [col for col, dt in zip(columns, column_types) if dt == 'int']
        df[int_columns] = df[int_columns].astype(float).fillna(INT_NAN).astype(int)
    return df


def _load_records(keyword_spec: SpecificationType, buffer: PReadBuf) -> RecordValueType:
    def _load_record(
        spec: StatementSpecification | ArraySpecification, buffer: PReadBuf
    ):
        if isinstance(spec, StatementSpecification):
            return _load_single_statement(spec, buffer)
        return _load_array(spec, buffer)

    if not isinstance(keyword_spec, RecordsSpecification):
        raise ValueError('`keyword_spec` shold be of type `RecordsSpecification`.')

    def _spec_generator(
        res: Sequence[
            pd.DataFrame
            | npt.NDArray[np.floating | np.integer | np.bool_]
            | tuple[pd.DataFrame, pd.DataFrame]
        ],
    ) -> Generator[StatementSpecification | ArraySpecification, None, None]:
        while True:
            try:
                get_new_spec = keyword_spec.get_next_specification

                if get_new_spec is None:
                    raise ValueError(
                        '`keyword_spec.get_next_specification` should not be None'
                    )
                spec = get_new_spec(res)
                yield spec
            except ValueError:
                break

    res: RecordValueType = []
    if keyword_spec.dynamic:
        spec_iterable = _spec_generator(res)
    else:
        spec_iterable = keyword_spec.specifications
        if spec_iterable is None:
            raise ValueError(
                '`keyword_spec.specifications` should not be None if not `keyword_spec.dynamic`.'
            )
    for spec in spec_iterable:
        res.append(_load_record(spec, buffer))
    return res


def _read_table_data(buffer: PReadBuf, depth: int, n: IntType) -> list[list[str]]:
    """
    Read numerical data for table.

    Parameters
    ----------
    buffer : PReadBuf
        String buffer to read.
    depth : int
        Depth of the table nesting (2 for multiindex table, 1 for normal table).
    n : IntType
        Number of tables to read.

    Returns
    -------
    List[np.ndarray] or List[List[np.ndarray]]
        List of numpy arrays (1 array for each region), if `depth==1`.
        List of lists of numpy array (1 array for each subtable, list of arrays
        for each region), if depth==2

    Raises
    ------
    ValueError
        If table block is not properly closed

    """
    data: list[list[str]] | list[list[list[str]]] = []
    if depth not in (1, 2):
        raise ValueError('`depth` should be 1 or 2.')
    for _ in range(depth):
        data = list(data)
    ind = [0] * depth
    group_end = True
    expr = re.compile(r'(\d*)\*(([^\s]*))')

    def _repl(match: re.Match[str]):
        num = match.groups()[0]
        val = match.groups()[1]
        if len(val) == 0:
            val = 'nan'
        num = int(num) if num else 1
        return ' '.join([val] * num)

    for line in buffer:
        line = line.strip()
        split = line.split('/')
        line = split[0]
        if len(line) > 0:
            cur_item = data
            line = expr.sub(_repl, line)
            for i in reversed(ind):
                if len(cur_item) == i:
                    if len(cur_item) > 0:
                        if isinstance(cur_item[0], str):
                            raise ValueError(
                                '`cur_item` should be of type `list[list[str]]'
                            )
                    cur_item = cast(list[list[str]] | list[list[list[str]]], cur_item)
                    cur_item.append([])

                cur_item = cur_item[i]
            values = line.split()
            if isinstance(cur_item, str):
                raise ValueError('`cur_item` should be of type `list[list[str]]')
            if len(cur_item) > 0:
                if isinstance(cur_item[0], str):
                    raise ValueError('`cur_item` should be of type `list[list[str]]')
            cur_item = cast(list[list[str]], cur_item)
            cur_item.append(values)
            group_end = False
        if len(split) > 1:
            if group_end:
                try:
                    ind[1] += 1
                    if len(data) == n:
                        break
                except IndexError:
                    _ = buffer.prev()
                    raise ValueError('Unexpected closing slash.')
                ind[0] = 0
            else:
                ind[0] += 1
                if len(data) == n and depth == 1:
                    break
            group_end = True

    if depth == 1:
        tmp_iter = [data]
    else:
        tmp_iter = data
    tmp_iter = cast(list[list[list[str]]], tmp_iter)
    for d in tmp_iter:
        for i, vals in enumerate(d):
            d[i] = list(itertools.chain(*vals))
    assert len(data) == n
    return data


def _load_array(keyword_spec: SpecificationType, buf: PReadBuf) -> NpArray:
    if not isinstance(keyword_spec, ArraySpecification):
        raise ValueError(keyword_spec, ArraySpecification)
    data = read_array(buf, dtype=keyword_spec.dtype)
    return data


def _load_array_with_units(keyword_spec: SpecificationType, buf: PReadBuf):
    if not isinstance(keyword_spec, ArraySpecification):
        raise ValueError('`keyword_spec` should be of type `ArraySpecification`.')
    line = next(buf)
    units = line.split()[0]
    _ = buf.prev()
    array = read_array(buf, dtype=keyword_spec.dtype, skip_first_word=True)
    return ArrayWithUnits(units, array)


def read_array(
    buffer: PReadBuf,
    dtype: type | None = None,
    compressed: bool = True,
    skip_first_word: bool = False,
) -> NpArray:
    """
    Read array data from a string buffer before first occurrence of '/' symbol.

    Parameters
    ----------
    buffer : buffer
        String buffer to read.
    dtype : dtype or None, default None.
        Defines dtype of an output array. If not specified, float array is returned.
    compressed : bool, default True.
        If True, A*B will be interpreted as B repeated A times.
    skip_first_word: bool, default False
        Should first word be skipped.

    Returns
    -------
    arr : ndarray
        Parsed array.

    """
    arr: list[NpArray] = []
    last_line = False
    if dtype is None:
        dtype = float
    for i, line in enumerate(buffer):
        if '/' in line:
            last_line = True
            line = line.split('/')[0]
        if i == 0 and skip_first_word:
            line = ' '.join(line.split()[1:])
        if compressed:
            x = decompress_array(line, dtype=dtype)
        else:
            x = np.fromstring(line.strip(), dtype=dtype, sep=' ')  # pyright: ignore[reportUnknownVariableType]
        if x.size:
            arr.append(x)  # pyright: ignore[reportUnknownArgumentType]
        if last_line:
            break
    return cast(NpArray, np.hstack(arr))


def decompress_array(s: str, dtype: type | None) -> NpArray:
    """Extract compressed numerical array from ASCII string. Interprets A*B as B repeated A times."""
    if dtype is None:
        dtype = float
    nums: list[float | int | bool] = []
    for x in s.split():
        try:
            val = [dtype(float(x))]
        except ValueError:
            k, val = x.split('*')
            val = [dtype(float(val))] * int(k)
        nums.extend(val)
    return np.array(nums)


def _load_parameters(
    keyword_spec: SpecificationType, buf: PReadBuf
) -> dict[str, str | None]:
    if not isinstance(keyword_spec, ParametersSpecification):
        raise ValueError('`keyword_spec` should be of type ParametersSpecification.')
    if keyword_spec.tabulated:
        return _load_parameters_tabulated(keyword_spec, buf)
    res: dict[str, str | None] = {}
    for line in buf:
        split = line.split('/')
        words = split[0].split()
        words = [w.strip('\'"') for w in words]
        for word in words:
            if '=' in word:
                key, val = word.split('=')
                res[key] = val
            else:
                res[word] = None
        if len(split) > 1:
            break
    return res


def _load_parameters_tabulated(_, buf: PReadBuf) -> dict[str, str | None]:
    res: dict[str, str | None] = {}
    for line in buf:
        split = line.split('/')
        if len(split) > 1 and split[0] == '':
            break
        words = split[0].split()
        if len(words) != 2:
            raise ValueError('There should be exactly two words on each line.')
        res[words[0]] = words[1]
        if len(split) > 1:
            break
    return res


def parse_vals(full: list[str | None], shift: int, vals: list[str]):
    """Parse values (unpack asterisk terms)."""
    full = copy.deepcopy(full)
    i = -1
    for i, v in enumerate(vals):
        if '*' in v:
            v = v.strip('\'"')
            if v == '*':
                continue
            try:
                n = int(v.split('*')[0])
                shift += n - 1
                if v.endswith('*'):
                    continue
                full[i + shift - n + 1 : i + shift + 1] = [v.split('*')[1]] * n
            except ValueError:
                full[i + shift] = v
        else:
            full[i + shift] = v
    return full, i + shift + 1


def _load_statement_list(keyword_spec: SpecificationType, buf: PReadBuf):
    """
    Parse Eclipse keyword data to dataframe.

    Parameters
    ----------
    keyword_spec : SpecificationType
        Keyword specification.
    buf : PReadBuf
        Buffer to read data from.

    Returns
    -------
    pd.Dataframe
        Loaded keyword dataframe.

    """
    statements: list[pd.DataFrame] = []
    while True:
        line = _get_expected_line(buf)
        if line.startswith('/'):
            break
        _ = buf.prev()
        statement = _load_single_statement(keyword_spec, buf)
        statements.append(statement)

    df = pd.concat(statements, ignore_index=True)
    return df


def _load_no_data(keyword_spec: SpecificationType | None, buf: PReadBuf):
    if keyword_spec is None:
        return None
    if not isinstance(keyword_spec, NoDataSpecification):
        raise ValueError(
            '`keyword_spec` should be of type `NoDataSpecification` or `None`.'
        )
    if not keyword_spec.terminated:
        return None
    line = next(buf)
    if not line.startswith('/'):
        raise ValueError('Data is not properly terminated.')
    return None


LOADERS: dict[
    DataTypes | None,
    Callable[[SpecificationType, PReadBuf, DataType | None], ValueType],
] = {
    None: lambda keyword_spec, buf, _: _load_no_data(keyword_spec, buf),
    DataTypes.STRING: lambda keyword_spec, buf, _: _load_string(keyword_spec, buf),
    DataTypes.OBJECT_LIST: lambda keyword_spec, buf, _: _load_object_list(
        keyword_spec, buf
    ),
    DataTypes.TABLE_SET: lambda keyword_spec, buf, data: _load_table(
        keyword_spec, buf, data
    ),
    DataTypes.ARRAY: lambda keyword_spec, buf, _: _load_array(keyword_spec, buf),
    DataTypes.PARAMETERS: lambda keyword_spec, buf, _: _load_parameters(
        keyword_spec, buf
    ),
    DataTypes.SINGLE_STATEMENT: lambda keyword_spec, buf, _: _load_single_statement(
        keyword_spec, buf
    ),
    DataTypes.STATEMENT_LIST: lambda keyword_spec, buf, _: _load_statement_list(
        keyword_spec, buf
    ),
    DataTypes.RECORDS: lambda keyword_spec, buf, _: _load_records(keyword_spec, buf),
    DataTypes.ARRAY_WITH_UNITS: lambda keyword_spec, buf, _: _load_array_with_units(
        keyword_spec, buf
    ),
}


class StringIteratorIO:
    """String iterator for text files."""

    def __init__(
        self,
        path: pathlib.Path,
        encoding: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize.

        Parameters
        ----------
        path : pathlib.Path
            Path to main model file.
        encoding : str | None, default None
            Encoding.
        logger : logging.Logger | None, default None
            Logger.

        """
        self._path: pathlib.Path = path
        if (encoding is not None) and encoding.startswith('auto'):
            encoding_tmp = encoding.split(':')
            if len(encoding_tmp) > 1:
                n_bytes = int(encoding_tmp[1])
            else:
                n_bytes = 5000

            with open(self._path, 'rb') as file:
                raw = file.read(n_bytes)
                self._encoding: str | None = chardet.detect(raw)['encoding']
        else:
            self._encoding = encoding
        self._line_number: int = 0
        self._f = None
        self._buffer: str = ''
        self._last_line: str | None = None
        self._include: StringIteratorIO | None = None
        self._on_last: bool = False
        if logger is None:
            logger = logging.getLogger(str(uuid.uuid4()))
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
        self._logger: logging.Logger = logger
        self._proposed_encodings: list[str] = DEFAULT_ENCODINGS.copy()
        self._stack: ExitStack | None = None

    @property
    def line_number(self) -> int:
        """Number of lines read."""
        if self._include is not None:
            return self._include.line_number
        return self._line_number

    @property
    def current_file(self) -> pathlib.Path:
        """
        File where cursor is located.

        Returns
        -------
        pathlib.Path
            Path to file.

        """
        if self._include is not None:
            return self._include.current_file
        return self._path

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self) -> str:
        """
        Return next string.

        Returns
        -------
        str
            Next string.

        """
        if self._include is not None:
            try:
                return next(self._include)
            except StopIteration:
                self._include = None
                self._logger.info(f'Continue reading {self.current_file}.')

        if self._on_last:
            self._on_last = False
            if self._last_line is None:
                raise ValueError('`_last_line` attribute should not be None.')
            return self._last_line
        try:
            if self._f is None:
                raise ValueError('`_f` should not be None.')
            line = next(self._f).split('--')[0].strip()
        except UnicodeDecodeError:
            return self._better_decoding()
        except StopIteration as e:
            self._logger.info(f'Finish reading {self.current_file}.')
            raise e
        self._line_number += 1
        if line:
            if line == 'INCLUDE':
                spec = DATA_DIRECTORY['INCLUDE']
                if spec is None:
                    raise ValueError(
                        'Specification for `INCLUDE` keyword should not be None.'
                    )
                path = LOADERS[DataTypes.STRING](spec.specification, self, None)
                if not isinstance(path, str):
                    raise ValueError('Path should be of type `str`.')
                self._include_file(path)
                return next(self)
            self._last_line = line
            return line
        return next(self)

    def _include_file(self, path: str | pathlib.Path):
        path = self._path.parent.joinpath(pathlib.Path(path))
        if self._stack is None:
            raise ValueError('`self._stack` should not be None.')
        with self._stack as stack:
            self._logger.info('INCLUDE keyword found.')
            self._include = stack.enter_context(
                StringIteratorIO(path, self._encoding, logger=self._logger)
            )
            self._stack = stack.pop_all()

    def _better_decoding(self) -> str:
        """Last chance to read line with default encodings."""
        try:
            enc = self._proposed_encodings.pop()
        except IndexError as err:
            raise ValueError(
                'Failed to decode at line {}'.format(self._line_number + 1)
            ) from err
        if enc == self._encoding:
            return self._better_decoding()
        self._f: TextIO | None = open(self._path, 'r', encoding=enc)  # pylint: disable=consider-using-with
        self._encoding = enc
        for _ in range(self._line_number):
            _ = next(self._f)
        return next(self)

    def prev(self):
        """Set current position to previous line."""
        if self._include is not None:
            _ = self._include.prev()
            return self
        if self._on_last:
            raise ValueError('Maximum cache depth is reached.')
        self._on_last = True
        return self

    def __enter__(self):
        """
        Enter context associated with the buffer.

        Adds corresponding context to the `ExitStack`.
        """
        with ExitStack() as stack:
            self._logger.info(f'Start reading {self._path}.')
            self._f = stack.enter_context(
                open(self._path, 'r', encoding=self._encoding)
            )  # pylint: disable=consider-u
            self._stack = stack.pop_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit the context. Closes all stack."""
        _ = exc_type, exc_val, exc_tb
        if self._stack is None:
            raise ValueError('`self._stack` should not be None.')
        self._stack.close()

    def read(self, n: int):
        """Read n characters."""
        while not self._buffer:
            try:
                self._buffer = next(self)
            except StopIteration:
                break
        result = self._buffer[:n]
        self._buffer = self._buffer[len(result) :]
        return result

    def skip_to(self, stop: Sequence[str] | str, *args):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        """Skip strings until stop token."""
        if isinstance(stop, str):
            stop = [stop]
        stop_pattern = '|'.join([x + '$' for x in stop])
        for line in self:
            if re.match(stop_pattern, line.strip(), *args):  # pyright: ignore[reportUnknownArgumentType]
                return


def _get_expected_line(buf: PReadBuf):
    try:
        line = next(buf)
    except StopIteration:
        raise ValueError('Buffer has ended earlier then expected.')
    return line


def load(
    path: pathlib.Path,
    logger: logging.Logger | None = None,
    encoding: str | None = None,
) -> DataType:
    """
    Load model data.

    Parameters
    ----------
    path : pathlib.Path
        Path to main model file.
    logger : logging.Logger | None, default None
        Logger.
    encoding : str | None, default None
        Encoding.

    Returns
    -------
    DataType
        Model data.

    """
    res: DataType = {}
    sections = [sec.value for sec in SECTIONS]
    if logger is None:
        logger = logging.getLogger(str(uuid.uuid4()))
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    filename = path.name

    logger.info(f'Start reading {filename}')
    cur_section = ''
    with StringIteratorIO(path, encoding=encoding, logger=logger) as lines:
        for line in lines:
            if not line:
                continue
            firstword = line.split(maxsplit=1)[0].upper()
            if firstword in sections:
                cur_section = firstword
                logger.info(f'Start {cur_section} section: line {lines.line_number}.')
                if cur_section not in res:
                    res[cur_section] = []
                continue
            if firstword in DATA_DIRECTORY:
                keyword_spec = DATA_DIRECTORY[firstword]
                if keyword_spec is None:
                    keyword_spec = get_dynamic_keyword_specification(firstword, res)
                keyword_sections = [sec.value for sec in keyword_spec.sections]
                if cur_section not in keyword_sections:
                    logger.warning(
                        f'Keyword {firstword} in section {cur_section}'
                        + f'is not supported (skipping): line {lines.line_number}'
                    )
                    continue
                logger.info(
                    f'Start reading keyword {firstword}: line {lines.line_number}.'
                )
                data = LOADERS[keyword_spec.type](
                    keyword_spec.specification, lines, res
                )
                if cur_section not in res:
                    res[cur_section] = []
                res[cur_section].append((firstword, data))
                logger.info(
                    f'Finish reading keyword {firstword}: line {lines.line_number}.'
                )
            elif firstword.startswith('/'):
                logger.info(f'Unnecessary "/" (skipping): line: {lines.line_number}.')
            else:
                logger.warning(
                    f'Keyword {firstword} in section {cur_section} '
                    + f'is not supported (skipping): line {lines.line_number}'
                )
    return res
