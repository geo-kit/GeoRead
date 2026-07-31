[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)

### 🌐 Языки

[English](../../README.md) | **Русский**

# GeoRead

Python-библиотека для чтения данных геологических и гидродинамических моделей.

## Возможности
* Чтение моделей пласта в формате Eclipse (E100, E300).
* Чтение бинарных файлов Eclipse.

## Использование
Чтение модели в текстовом формате Eclipse.
```python
from georead import load
data = load('path_to_model.DATA')
```
Чтение бинарных файлов модели.
```python
import georead.binary
binary_data = georead.binary.load('path_to_model.DATA')
```

Подробнее об API — в [документации](https://geo-kit.github.io/GeoRead/).