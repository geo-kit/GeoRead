"""Reader for the on-disk output of the JutulDarcy driver (geocode/bin/jutul_run.jl)."""

from .load import JutulResults, load

__all__ = ['JutulResults', 'load']
