"""What counts as a numbered run folder, shared by ExperimentID and open_vault."""

import os
import re
from collections.abc import Iterable
from pathlib import Path

# four or more digits without extra leading zeros (f"{id:04d}" gives "10000" after 9999),
# optionally followed by "_suffix"
_ID = r"\d{4}|[1-9]\d{4,}"
ID_NAME = re.compile(_ID)
_RUN_NAME = re.compile(rf"({_ID})(?:_.*)?")


def list_dirs(root: Path) -> list[Path]:
    """Subdirectories of *root* in listing order (unspecified); empty if *root* is missing."""
    try:
        with os.scandir(root) as it:
            return [Path(e.path) for e in it if e.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        return []


def is_run_dir(path: Path) -> bool:
    """A folder that holds zarro output: vault.zarr, data.npz (NpzWriter) or metadata.yaml."""
    return (path / "vault.zarr").is_dir() or (path / "data.npz").is_file() or (path / "metadata.yaml").is_file()


def numbered(dirs: Iterable[Path]) -> list[tuple[int, Path]]:
    """Directories whose name is a run id (optionally with a suffix), highest id first."""
    found = [(int(m.group(1)), p) for p in dirs if (m := _RUN_NAME.fullmatch(p.name))]
    return sorted(found, key=lambda t: (t[0], t[1].name), reverse=True)


def latest_run_id(dirs: Iterable[Path]) -> int:
    """Largest id among run folders, 0 if there are none. Checks folder contents only from the top down."""
    return next((i for i, p in numbered(dirs) if is_run_dir(p)), 0)


def find_runs(dirs: Iterable[Path], run_id: int) -> list[Path]:
    """Run folders with exactly this id, e.g. ``0007`` and ``0007_fridge`` for id 7."""
    return [p for i, p in numbered(dirs) if i == run_id and is_run_dir(p)]
