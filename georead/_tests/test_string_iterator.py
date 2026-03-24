"""Test StringIteratorIO."""

import pathlib

import pytest

from georead._load_utils import StringIteratorIO

RESULT_LINES = (
    'line 1',
    'line 3',
    'inc1 line 1',
    'inc1 line 3',
    'line 7',
    'inc2 line 1',
    'inc3 line 1',
    'inc2 line 5',
)


@pytest.fixture
def iterator():
    """Provide iterator fixture."""
    data_file_path = (
        pathlib.Path(__file__).parent
        / 'data'
        / 'string_iterator_io_test_data'
        / 'test.data'
    )
    return StringIteratorIO(data_file_path)


def test_iterator(iterator: StringIteratorIO):
    """Test that iterator returns proper values."""
    with iterator:
        for line1, line2 in zip(iterator, RESULT_LINES):
            assert line1 == line2


def test_current_file(iterator: StringIteratorIO):
    """Test that iterators properly recognizes current file."""
    with iterator:
        for line in iterator:
            if line.startswith('inc1'):
                assert (
                    iterator.current_file.absolute()
                    == (
                        pathlib.Path(__file__).parent
                        / 'data'
                        / 'string_iterator_io_test_data'
                        / 'inc1.inc'
                    ).absolute()
                )
            else:
                assert (
                    iterator.current_file.absolute()
                    == (
                        pathlib.Path(__file__).parent
                        / 'data'
                        / 'string_iterator_io_test_data'
                        / 'test.data'
                    ).absolute()
                )


def test_line_number(iterator: StringIteratorIO):
    """Test that iterator properly recognizes current line number."""
    with iterator:
        assertions = [False, False]
        for line in iterator:
            if line == 'line 3':
                assert iterator.line_number == 3
                assertions[0] = True
            if line == 'inc1 line 1':
                assert iterator.line_number == 1
                assertions[1] = True

    for a in assertions:
        assert a
