"""Load JutulDarcy run output (manifest.json + HDF5 files)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JutulResults:
    """Results of a JutulDarcy run."""

    case_dir: Path
    manifest: dict[str, object]
    dates: np.ndarray
    pressure: np.ndarray
    swat: np.ndarray | None
    soil: np.ndarray | None
    sgas: np.ndarray | None
    active_to_natural: np.ndarray
    wells: dict[str, pd.DataFrame]


def _read_array(f: h5py.File, name: str) -> np.ndarray:
    ds = f[name]
    if not isinstance(ds, h5py.Dataset):
        raise TypeError(f'{name} is not an HDF5 dataset.')
    return np.asarray(ds[...])  # pyright: ignore[reportUnknownArgumentType]


def _read_manifest(case_dir: Path) -> dict[str, object]:
    manifest = json.loads(  # pyright: ignore[reportAny]
        (case_dir / 'manifest.json').read_text()
    )
    if not isinstance(manifest, dict):
        raise TypeError('manifest.json must contain a JSON object.')
    return cast('dict[str, object]', manifest)


def _manifest_section(manifest: dict[str, object], name: str) -> dict[str, object]:
    section = manifest[name]
    if not isinstance(section, dict):
        raise TypeError(f"manifest section '{name}' must be an object.")
    return cast('dict[str, object]', section)


def _manifest_str(section: dict[str, object], name: str) -> str:
    value = section[name]
    if not isinstance(value, str):
        raise TypeError(f"manifest field '{name}' must be a string.")
    return value


def _manifest_columns(manifest: dict[str, object]) -> list[str]:
    value = _manifest_section(manifest, 'wells')['columns']
    if not isinstance(value, list):
        raise TypeError("manifest field 'wells.columns' must be a string list.")

    columns = cast('list[object]', value)
    if not all(isinstance(item, str) for item in columns):
        raise TypeError("manifest field 'wells.columns' must be a string list.")
    return cast('list[str]', columns)


def load(case_dir: Path | str) -> JutulResults:
    """
    Load a JutulDarcy run directory produced by geocode/bin/jutul_run.jl.

    State arrays have shape (n_steps + 1, n_active) with state0 first;
    saturations missing from the run (e.g. SGAS in a two-phase model) are None.
    """
    case_dir = Path(case_dir)
    manifest = _read_manifest(case_dir)
    states = _manifest_section(manifest, 'states')
    wells_manifest = _manifest_section(manifest, 'wells')

    sats: dict[str, np.ndarray | None] = {}
    with h5py.File(case_dir / _manifest_str(states, 'file'), 'r') as f:
        pressure = _read_array(f, '/pressure')
        for name in ('swat', 'soil', 'sgas'):
            sats[name] = _read_array(f, '/' + name) if '/' + name in f else None
        dates = _read_array(f, '/dates_iso8601').astype(str).astype('datetime64[s]')

    columns = _manifest_columns(manifest)
    wells: dict[str, pd.DataFrame] = {}
    with h5py.File(case_dir / _manifest_str(wells_manifest, 'file'), 'r') as f:
        group = f['wells']
        if not isinstance(group, h5py.Group):
            raise TypeError("'wells' is not an HDF5 group.")
        for name, ds in group.items():  # pyright: ignore[reportUnknownVariableType]
            wells[name] = pd.DataFrame(
                np.asarray(ds[...]),  # pyright: ignore[reportUnknownArgumentType]
                columns=columns,
            )

    with h5py.File(case_dir / 'cell_indices.h5', 'r') as f:
        active_to_natural = _read_array(f, '/active_to_natural')

    return JutulResults(
        case_dir=case_dir,
        manifest=manifest,
        dates=dates,
        pressure=pressure,
        swat=sats['swat'],
        soil=sats['soil'],
        sgas=sats['sgas'],
        active_to_natural=active_to_natural,
        wells=wells,
    )
