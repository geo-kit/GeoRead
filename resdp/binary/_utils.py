import math
import pathlib
from struct import unpack

import numpy as np

_DATA_TYPES = {
    'INTE': (4, 'i', 1000),
    'REAL': (4, 'f', 1000),
    'LOGI': (4, 'i', 1000),
    'DOUB': (8, 'd', 1000),
    'CHAR': (8, '8s', 105),
    'MESS': (8, '8s', 105),
}
for val in range(8, 40):
    _DATA_TYPES['C0{}'.format(val)] = (val, '{}s'.format(val), 105)

def _get_type_info(data_type):
    """Returns element size, format and element skip for the given data type.

    Parameters
    ----------
    data_type: str
        Should be a key from the DATA_TYPES

    Returns
    -------
    type_info: tuple
    """
    try:
        return _DATA_TYPES[data_type]
    except KeyError as exc:
        raise ValueError(f'Unknown datatype {data_type}') from exc
    

def read_header_and_section(path: pathlib.Path):
    with open(path, 'rb') as f:
        header = f.read(4)
        sections = []
        while True:
            name_bytes = f.read(8)
            if not name_bytes:
                break
            section_name = unpack('8s', name_bytes)[0].decode('ascii').strip().upper()
            n_elements = unpack('>i', f.read(4))[0]
            data_type = unpack('4s', f.read(4))[0].decode('ascii')
            f.read(8)
            element_size, fmt, element_skip = _get_type_info(data_type)
            start = f.tell()
            size = element_size * n_elements + 8 * (math.floor((n_elements - 1) / element_skip) + 1)
            f.seek(f.tell() + size)
            sections.append((section_name, data_type, start, n_elements))
    return header, sections

def read_binary_data(path: pathlib.Path, data_type: str, start: int, n_elements: int):
    element_size, fmt, element_skip = _get_type_info(data_type)
    size = element_size * n_elements + 8 * (math.floor((n_elements - 1) / element_skip) + 1)
    with open(path, 'rb') as f:
        _ = f.seek(start)
        data = f.read(size)
    return data

def decode(data: bytes, data_type: str, n_elements: int):
    element_size, fmt, element_skip = _get_type_info(data_type)
    n_skip = math.floor((n_elements - 1) / element_skip)
    skip_elements = 8 // element_size
    skip_elements_total = n_skip * skip_elements
    data_format = fmt * (n_elements + skip_elements_total)
    data_size = element_size * (n_elements + skip_elements_total)
    if data_type in ['INTE', 'REAL', 'LOGI', 'DOUB']:
        data_format = '>' + data_format
    decoded_section = list(unpack(data_format, data[: + data_size]))
    del_ind = np.repeat(np.arange(1, 1 + n_skip) * element_skip, skip_elements)
    del_ind += np.arange(len(del_ind))
    decoded_section = np.delete(decoded_section, del_ind)
    if data_type in ['CHAR', 'C008', 'C015']:
        decoded_section = np.char.decode(decoded_section, encoding='ascii')
    return decoded_section

def _decode_section(section):
    """Decodes section of a binary ECLIPSE file.

    Parameters
    ----------
    section: list

    Returns
    -------
    decoded_section: ndarray
    """
    n_elements, data_type, element_size, fmt, element_skip, binary_data = section

    n_skip = math.floor((n_elements - 1) / element_skip)
    skip_elements = 8 // element_size
    skip_elements_total = n_skip * skip_elements
    data_format = fmt * (n_elements + skip_elements_total)
    data_size = element_size * (n_elements + skip_elements_total)
    if data_type in ['INTE', 'REAL', 'LOGI', 'DOUB']:
        data_format = '>' + data_format
    decoded_section = list(unpack(data_format, binary_data[24: 24 + data_size]))
    del_ind = np.repeat(np.arange(1, 1 + n_skip) * element_skip, skip_elements)
    del_ind += np.arange(len(del_ind))
    decoded_section = np.delete(decoded_section, del_ind)
    if data_type in ['CHAR', 'C008', 'C015']:
        decoded_section = np.char.decode(decoded_section, encoding='ascii')
    return decoded_section
