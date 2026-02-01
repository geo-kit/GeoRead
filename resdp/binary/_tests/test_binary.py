import pytest
import numpy as np
import tarfile
from pathlib import Path
import resdp.binary


@pytest.fixture
def egg_model(tmp_path: Path):
    """Extract files of Egg resevoir model and provide path to it."""
    tar_file_path = Path(__file__).parent / '_data' / 'egg.tar.gz'

    extract_dir = tmp_path / 'egg'
    extract_dir.mkdir()
    with tarfile.open(tar_file_path, 'r') as f:
        f.extractall(extract_dir, filter='data')

    return extract_dir / 'egg/Egg_Model_ECL.DATA'


@pytest.fixture
def egg_binary_data(egg_model: Path):
    """Load Egg model binary data."""
    return resdp.binary.load(egg_model)


def test_load_egg(egg_model: Path):
    """Test loading Egg model binary data."""
    binary_data = resdp.binary.load(egg_model)
    assert isinstance(binary_data, resdp.binary.BinaryData)


def test_all_parts_exist(egg_binary_data: resdp.binary.BinaryData):
    """Test that all existing files were loaded."""
    extensions: set[resdp.binary.FileType] = {
        'EGRID',
        'INIT',
        'UNSMRY',
        'SMSPEC',
        'UNRST',
    }
    assert extensions == set(egg_binary_data.keys())
    for ext in extensions:
        assert isinstance(egg_binary_data[ext], resdp.binary.BinaryFileData)


def test_find_unique(egg_binary_data: resdp.binary.BinaryData):
    """Test that `BinaryFileData.find_unique` correctly finds unique element."""
    assert egg_binary_data['EGRID'].find_unique('GRIDHEAD') == 1


def test_find_unique_no_value(egg_binary_data: resdp.binary.BinaryData):
    """Test that `BinaryFileData.find_unique` returns `None` when element is not presented."""
    assert egg_binary_data['EGRID'].find_unique('abc') is None


def test_find_unique_error(egg_binary_data: resdp.binary.BinaryData):
    """Test that `BinaryFileData.find_unique` raises exception when element is not unique."""
    with pytest.raises(ValueError):
        _ = egg_binary_data['UNSMRY'].find_unique('PARAMS')


def test_value(egg_binary_data: resdp.binary.BinaryData):
    """Test that loaded value is correct."""
    val = egg_binary_data['EGRID'][1]
    assert val.name == 'GRIDHEAD'
    assert (
        val.value
        == np.array(  # pyright: ignore[reportAny]
            [1, 60, 60, 7] + [0] * 20 + [1] * 2 + [0] + [1] * 6 + [0] * 67
        )
    ).all()


def test_seek(egg_binary_data: resdp.binary.BinaryData):
    """Test `BinaryFileData.seek`."""
    assert egg_binary_data['UNSMRY'].tell() == 0
    egg_binary_data['UNSMRY'].seek(3)
    assert egg_binary_data['UNSMRY'].tell() == 3


def test_find(egg_binary_data: resdp.binary.BinaryData):
    """Test `BinaryFileData.find`."""
    assert egg_binary_data['UNSMRY'].find('PARAMS') == 3
    egg_binary_data['UNSMRY'].seek(4)
    assert egg_binary_data['UNSMRY'].find('PARAMS') == 6
    egg_binary_data['UNSMRY'].seek(363)
    assert egg_binary_data['UNSMRY'].find('SEQHDR') is None


def test_find_prev(egg_binary_data: resdp.binary.BinaryData):
    """Test `BinaryFileData.find_prev`."""
    assert egg_binary_data['UNSMRY'].find_prev('PARAMS') is None
    egg_binary_data['UNSMRY'].seek(4)
    assert egg_binary_data['UNSMRY'].find_prev('PARAMS') == 3
