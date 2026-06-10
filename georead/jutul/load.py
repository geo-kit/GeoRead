"""Load JutulDarcy run output (manifest.json + HDF5 files)."""
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JutulResults:
    """Results of a JutulDarcy run."""
    case_dir: Path
    manifest: dict
    dates: np.ndarray
    pressure: np.ndarray
    swat: np.ndarray
    soil: np.ndarray
    sgas: np.ndarray
    active_to_natural: np.ndarray
    wells: dict


def load(case_dir):
    """Load a JutulDarcy run directory produced by geocode/bin/jutul_run.jl.

    State arrays have shape (n_steps + 1, n_active) with state0 first;
    saturations missing from the run (e.g. SGAS in a two-phase model) are None.
    """
    case_dir = Path(case_dir)
    manifest = json.loads((case_dir / "manifest.json").read_text())

    sats = {}
    with h5py.File(case_dir / manifest["states"]["file"], "r") as f:
        pressure = f["/pressure"][...]
        for name in ("swat", "soil", "sgas"):
            sats[name] = f["/" + name][...] if "/" + name in f else None
        raw = f["/dates_iso8601"][...]
        dates = np.array([s.decode() if isinstance(s, bytes) else s for s in raw],
                         dtype="datetime64[s]")

    columns = list(manifest["wells"]["columns"])
    wells = {}
    with h5py.File(case_dir / manifest["wells"]["file"], "r") as f:
        for name, ds in f["wells"].items():
            wells[name] = pd.DataFrame(ds[...], columns=columns)

    with h5py.File(case_dir / "cell_indices.h5", "r") as f:
        active_to_natural = f["/active_to_natural"][...]

    return JutulResults(
        case_dir=case_dir,
        manifest=manifest,
        dates=dates,
        pressure=pressure,
        swat=sats["swat"],
        soil=sats["soil"],
        sgas=sats["sgas"],
        active_to_natural=active_to_natural,
        wells=wells,
    )
