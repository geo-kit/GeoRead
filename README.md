[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)

### 🌐 Multi-Language Support

**English** | [Русский](./translations/ru/README.md)

# GeoRead

Python library for reading reservoir model data.

## Features
* Read reservoir models in Eclipse (E100, E300) format.
* Read Eclipse binary files

## Usage
Reading model in Eclipse text format.
```python
from georead import load

data = load('path_to_model.DATA')
```
Reading model binary files.
```python
import georead.binary

binary_data = georead.binary.load('path_to_model.DATA')
```

For more details on API, see the [documantation](https://geo-kit.github.io/GeoRead/).

