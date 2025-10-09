from abc import ABC
from collections import UserDict, UserList
from collections.abc import Iterable
import pathlib
from typing import Literal, Mapping, TypeVar, override

from ._utils import read_header_and_section, read_binary_data, decode


FileType = Literal['EGRID', 'INIT', 'UNRST', 'UNSMRY']


class BinaryAttribute():
    def __init__(self, path: pathlib.Path, name: str,
                 data_type: str, start: int, n_elements: int):
        self._path: pathlib.Path = path
        self._name: str = name
        self._data_type: str = data_type
        self._start: int = start
        self._n_elements: int = n_elements
    @property
    def value(self):
        data = read_binary_data(self._path, self._data_type, self._start,
                                self._n_elements)
        decoded_data = decode(data, self._data_type, self._n_elements)
        return decoded_data
    @property
    def name(self) -> str:
        return self._name
        
class BinaryFileData(UserList[BinaryAttribute]):
    def __init__(self, path: pathlib.Path):
        super().__init__({})
        self._path: pathlib.Path = path
        _, data = read_header_and_section(path)
        self._pos = 0
        for entry in data:
            self.append(BinaryAttribute(
                self._path,
                entry[0],
                entry[1],
                entry[2],
                entry[3]))

    @property
    def names(self):
        [entry.name for entry in self]
    def tell(self) -> int:
        return self._pos
    
    def seek(self, pos: int):
        if pos >= len(self):
            raise ValueError()
        self._pos = pos

    def _find(self, iter: Iterable[int], name: str):
        for i in iter:
            if self[i].name == name:
                return i
        return None
        
    def find(self, name: str):
        return self._find(range(self._pos, len(self)), name)

    def find_prev(self, name: str):
        return self._find(range(self._pos, 0, -1), name)

class BinaryData(UserDict[FileType, BinaryFileData]):
    def __init__(self, ):
        super().__init__()

def load(model_path: pathlib.Path) -> BinaryData:
    pass
