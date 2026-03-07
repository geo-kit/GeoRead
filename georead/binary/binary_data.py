"""Classes and functions for reading model binary files."""

from collections import UserDict, UserList
from collections.abc import Iterable
import pathlib
from typing import Literal

import numpy as np

from ._utils import read_sections, read_binary_data, decode

FileType = Literal['EGRID', 'INIT', 'UNRST', 'UNSMRY', 'X*', 'SMSPEC']


class BinaryAttribute:
    """Single attribute in binary file."""

    def __init__(
        self, path: pathlib.Path, name: str, data_type: str, start: int, n_elements: int
    ):
        """
        Initialize BinaryAtribute.

        Parameters
        ----------
        path : pathlib.Path
            Path to the model.
        name : str
            Attribute name.
        data_type : str
            Type of the attribute data.
        start : int
            Starting position in the file.
        n_elements : int
            Number of elements.

        """
        self._path: pathlib.Path = path
        self._name: str = name
        self._data_type: str = data_type
        self._start: int = start
        self._n_elements: int = n_elements

    @property
    def value(
        self,
    ) -> (
        np.ndarray[tuple[int], np.dtype[np.float64]]
        | np.ndarray[tuple[int], np.dtype[np.int_]]
        | np.ndarray[tuple[int], np.dtype[np.str_]]
    ):
        """
        Attribute value.

        Returns
        -------
        (
            np.ndarray[tuple[int], np.dtype[np.float64]]
            | np.ndarray[tuple[int], np.dtype[np.int_]]
            | np.ndarray[tuple[int], np.dtype[np.str_]]
        )
            Attribute value.

        """
        data = read_binary_data(
            self._path, self._data_type, self._start, self._n_elements
        )
        decoded_data = decode(data, self._data_type, self._n_elements)
        return decoded_data

    @property
    def name(self) -> str:
        """
        Attribute name.

        Returns
        -------
        str
            Attribute name.

        """
        return self._name


class BinaryFileData(UserList[BinaryAttribute]):
    """Binary file data."""

    def __init__(self, path: pathlib.Path):
        """
        Initialize BinaryFileData.

        Parameters
        ----------
        path : pathlib.Path
            Path to binary file.

        """
        super().__init__({})
        self._path: pathlib.Path = path
        data = read_sections(path)
        self._pos: int = 0
        for entry in data:
            self.append(
                BinaryAttribute(self._path, entry[0], entry[1], entry[2], entry[3])
            )

    @property
    def names(self) -> tuple[str, ...]:
        """Names of attributes."""
        return tuple([entry.name for entry in self])

    def tell(self) -> int:
        """
        Tell current position.

        Returns
        -------
        int
            Current position.

        """
        return self._pos

    def seek(self, pos: int) -> None:
        """
        Set position.

        Parameters
        ----------
        pos : int
            Position.

        """
        if pos >= len(self):
            raise ValueError()
        self._pos = pos

    def _find(self, iter: Iterable[int], name: str) -> int | None:
        for i in iter:
            if self[i].name == name:
                return i
        return None

    def find(self, name: str) -> int | None:
        """
        Find next attribute with specified name.

        Parameters
        ----------
        name : str
            Name to find.

        Returns
        -------
        int | None
            Position of the found attribute, None if not found.

        """
        return self._find(range(self._pos, len(self)), name)

    def find_prev(self, name: str) -> int | None:
        """
        Find previous attribute with specified name.

        Parameters
        ----------
        name : str
            Name to find.

        Returns
        -------
        int | None
            Position of the found attribute, None if not found.

        """
        return self._find(range(self._pos, 0, -1), name)

    def find_unique(self, name: str) -> int | None:
        """
        Find attribute with specified name and check that it is unique.

        Parameters
        ----------
        name : str
            Name to find.

        Returns
        -------
        int | None
            Position of the found attribute, None if not found.

        """
        values: list[int] = []
        for i in range(0, len(self)):
            if self[i].name == name:
                values.append(i)
                if len(values) > 1:
                    raise ValueError('Section `{name} is not unique.')
        if len(values) == 0:
            return None
        return values[0]

    def find_all(self, name: str) -> list[int]:
        """
        Find all attributes with specified name.

        Parameters
        ----------
        name : str
            Name to find.

        Returns
        -------
        list[int]
            Positions of the attribute.

        """
        res = [i for i, item in enumerate(self) if item.name == name]
        return res


class BinaryData(UserDict[FileType, BinaryFileData]):
    """Binary Data."""

    def __init__(self, path_to_results: pathlib.Path, basename: str):
        """
        Initialize BinaryData.

        Parameters
        ----------
        path_to_results : pathlib.Path
            Path to folder with binary file.
        basename : str
            Name of the model.

        """
        super().__init__()
        for ext in ('EGRID', 'INIT', 'UNRST', 'UNSMRY', 'SMSPEC'):
            filename = basename + f'.{ext}'
            found_files: list[pathlib.Path] = []
            for f in path_to_results.iterdir():
                if f.is_file() and f.name.lower() == filename.lower():
                    found_files.append(f)
            if len(found_files) > 1:
                raise ValueError(f'{path_to_results} contains multiple {ext} files.')
            if len(found_files) == 1:
                self[ext] = BinaryFileData(found_files[0])


def load(model_path: pathlib.Path) -> BinaryData | None:
    """
    Load binary data.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to the model .data file.

    Returns
    -------
    BinaryData | None
        Loaded data.

    """
    basename = model_path.stem
    results_dir = model_path.parent / 'RESULTS'
    if not results_dir.is_dir():
        return None
    if (results_dir / basename).is_dir():
        results_dir = results_dir / basename
    return BinaryData(results_dir, basename)
