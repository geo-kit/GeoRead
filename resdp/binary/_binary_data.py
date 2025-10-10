from collections import UserDict, UserList
from collections.abc import Iterable
import pathlib
from typing import Literal

from ._utils import read_sections, read_binary_data, decode

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
        data = read_sections(path)
        self._pos: int = 0
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
    def __init__(self, path_to_results: pathlib.Path, basename: str):
        super().__init__()
        filename = basename + '.EGRID'
        found_files: list[pathlib.Path] = []
        for f in path_to_results.iterdir():
            if f.is_file() and f.name.lower() == filename.lower():
                found_files.append(f)
        if len(found_files) > 1:
            raise ValueError(f'{path_to_results} contains multiple `EGRID` files.')
        if len(found_files) == 1:
            self['EGRID'] = BinaryFileData(found_files[0])

def load(model_path: pathlib.Path) -> BinaryData | None:
    basename = model_path.stem
    results_dir = model_path.parent / 'RESULTS'
    if not results_dir.is_dir():
        return None
    if (results_dir / basename).is_dir():
        results_dir = results_dir / basename
    return BinaryData(results_dir, basename)
