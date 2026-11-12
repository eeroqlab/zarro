"""Read a zarro run directory (``vault.zarr`` plus ``metadata.yaml``) for analysis or viewing.

The reader only needs numpy, zarr and pyyaml. It understands what zarro writes:

- sweeps written with :class:`~zarro.core.ZarrWriter`: one coordinate array per
  control variable and one array per readout, shaped like the sweep grid plus the
  readout's trailing shape;
- time series written with :class:`~zarro.core.StreamingWriter`: a ``_time``
  array and one array per readout that grows along time.

A run can be opened while it is still being written. Zarr fills points that have
not been written with ``0.0``, so :meth:`Vault.written_mask` tells written from
unwritten points, and :meth:`Vault.values` returns NaN for the latter.

Example
-------
>>> from zarro import open_vault
>>> run = open_vault("data/0007")
>>> run.axes, run.shape
>>> signal = run.values("signal")           # NaN where not written yet
>>> x = run.coords[run.axes[0].controls[0]]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
import zarr

__all__ = ["Axis", "Readout", "Vault", "VaultError", "open_vault"]


class VaultError(Exception):
    """The directory is not a readable zarro run."""


@dataclass(frozen=True)
class Axis:
    """One control axis. A multi-control axis has several ``controls`` sharing one length."""

    index: int
    controls: tuple[str, ...]
    length: int
    units: dict[str, str | None] = None  # {name: unit_string_or_None}


@dataclass(frozen=True)
class Readout:
    """One recorded readout. ``shape`` is the trailing shape: ``()`` for a scalar."""

    name: str
    kind: str  # "scalar" | "trace" | "image"
    shape: tuple[int, ...]
    unit: str | list[str] | None = None
    contains: str | list[str] | None = None


def load_yaml(path: Path) -> dict:
    """Return the file's mapping, or ``{}`` if it is missing or half-written."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _infer_kind(trailing: tuple[int, ...]) -> str:
    return ("scalar", "trace", "image")[min(len(trailing), 2)]


def _print_table(rows, headers, title, tabulate) -> None:
    print(title)
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        # minimal fallback when tabulate is not installed
        col_widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt.format(*row))


class Vault:
    """A run directory opened for reading.

    Parameters
    ----------
    path : str or Path
        The run directory, the one that contains ``vault.zarr``.
    control : dict, optional
        ``{"axis0": [names], "axis1": [names], ...}`` to use when the vault does
        not record its own control axes (for example because ``metadata.yaml`` is
        gone and the zarr attributes were never written).
    readouts : dict, optional
        ``{name: {"kind", "unit", "shape", "contains"}}`` to use when the vault
        does not record its own readout specs.

    Attributes
    ----------
    kind : "sweep" or "monitor"
        A vault with a ``_time`` array is a monitor (StreamingWriter) run.
    metadata : metadata.yaml merged over the zarr attributes. The attributes are
        written when the writer finishes; metadata.yaml exists from the start.
    tags : list of str
    axes : control axes, outermost first (empty for monitors).
    coords : ``{control name: values}`` for every control variable.
    coord_axis : ``{control name: axis index}``.
    readouts : ``{name: Readout}``.
    """

    def __init__(self, path: str | Path, *, control: dict | None = None, readouts: dict | None = None):
        self.path = Path(path)
        vault = self.path / "vault.zarr"
        if not vault.is_dir():
            raise VaultError(f"{self.path} has no vault.zarr")
        try:
            self._group = zarr.open_group(str(vault), mode="r")
        except Exception as exc:
            raise VaultError(f"cannot open {vault}: {exc}") from exc

        self.metadata = {**dict(self._group.attrs), **load_yaml(self.path / "metadata.yaml")}
        self.tags = list(self.metadata.get("tags") or [])
        self.kind = "monitor" if "_time" in self._group else "sweep"

        self.axes: list[Axis] = []
        self.coords: dict[str, np.ndarray] = {}
        self.coord_axis: dict[str, int] = {}
        self.readouts: dict[str, Readout] = {}

        if self.kind == "sweep":
            self._load_axes(control)
        self._load_readouts(readouts)

    # ── structure ─────────────────────────────────────────────────────

    def _load_axes(self, fallback: dict | None) -> None:
        schema = self.metadata.get("schema")
        control = self.metadata.get("control") or (schema.get("control") if isinstance(schema, dict) else None)
        control = control or fallback
        if not control:
            raise VaultError(f"{self.path}: no control axes in metadata.yaml, the zarr attributes or the arguments")
        for key in sorted(control, key=lambda k: int(str(k).removeprefix("axis"))):
            value = control[key]
            if isinstance(value, list):          # old format: ["Vch", "Vm"]
                names = tuple(value)
                units = {n: None for n in names}
            else:                                # new format: {"Vch": {"unit": "V"}, ...}
                names = tuple(value.keys())
                units = {n: (value[n] or {}).get("unit") for n in names}
            missing = [n for n in names if n not in self._group]
            if missing:
                raise VaultError(f"{self.path}: coordinate array(s) {missing} not found in vault.zarr")
            values = {n: np.asarray(self._group[n][:]) for n in names}
            self.axes.append(Axis(len(self.axes), names, len(values[names[0]]), units))
            for n in names:
                self.coords[n] = values[n]
                self.coord_axis[n] = len(self.axes) - 1

    def _load_readouts(self, fallback: dict | None) -> None:
        schema = self.metadata.get("schema")
        specs = self.metadata.get("readout") or (schema.get("readout") if isinstance(schema, dict) else None)
        specs = specs or fallback
        if not specs:
            names = list(self.metadata.get("readouts") or []) or [
                n for n in self._group.array_keys() if n != "_time" and n not in self.coords
            ]
            specs = {n: {} for n in names}
        lead = len(self.shape) if self.kind == "sweep" else 1
        for name, spec in specs.items():
            if name not in self._group:
                continue
            trailing = tuple(self._group[name].shape[lead:])
            self.readouts[name] = Readout(
                name=name,
                kind=spec.get("kind") or _infer_kind(trailing),
                shape=trailing,
                unit=spec.get("unit"),
                contains=spec.get("contains"),
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Sweep grid shape, or ``(n_samples,)`` for a monitor (n grows while it runs)."""
        if self.kind == "monitor":
            return (int(self._group["_time"].shape[0]),)
        return tuple(a.length for a in self.axes)

    @property
    def time(self) -> np.ndarray:
        """Unix timestamps of a monitor run, one per sample."""
        if self.kind != "monitor":
            raise VaultError("only monitor runs have a time axis")
        return np.asarray(self._group["_time"][:])

    # ── data ──────────────────────────────────────────────────────────

    def read(self, name: str, selection=Ellipsis) -> np.ndarray:
        """Read a readout, indexed like the sweep grid, with ``0.0`` where nothing is written yet."""
        return np.asarray(self._group[name][selection])

    def values(self, name: str) -> np.ndarray:
        """Whole readout as floats, NaN where nothing has been written yet."""
        raw = self.read(name)
        out = raw.astype(raw.dtype if raw.dtype.kind == "f" else float)
        mask = self.written_mask(name)
        if not mask.all():
            full = mask.reshape(mask.shape + (1,) * (out.ndim - mask.ndim))
            out[~np.broadcast_to(full, out.shape)] = np.nan
        return out

    def written_mask(self, name: str, order: np.ndarray | None = None) -> np.ndarray:
        """Boolean array over the sweep grid: True where the point has been written.

        This is an estimate, because zarr does not store chunks that equal the
        fill value ``0.0``. If you know the run finished, every point is written
        and you don't need it. Monitor runs are always fully written.

        - Scalars are stored in plain chunks, so a point counts as written when
          its chunk file exists.
        - Trace and image readouts are stored in shards, so a point counts as
          written when any of its values is non-zero.
        - Points are assumed to be written in a fixed order, and every point
          visited before the last one found is also counted as written. A real
          ``0.0`` inside the scanned region therefore stays visible, but zeros
          after the last non-zero point cannot be told from unwritten ones.

        Parameters
        ----------
        order : integer array shaped like the grid, optional
            For every grid position, the step at which it is written. The
            default is C order, which is right for a plain nested sweep.
        """
        grid = self.shape
        if self.kind == "monitor":
            return np.ones(grid, dtype=bool)
        array = self._group[name]
        if array.shards is None:
            mask = self._chunk_mask(name, array, grid)
        else:
            data = np.asarray(array[...])
            trailing = tuple(range(len(grid), data.ndim))
            mask = (data != 0).any(axis=trailing) if trailing else data != 0
        if mask.any():
            if order is None:
                order = np.arange(mask.size).reshape(grid)
            mask |= order <= order[mask].max()
        return mask

    def _chunk_mask(self, name: str, array, grid: tuple[int, ...]) -> np.ndarray:
        mask = np.zeros(grid, dtype=bool)
        for idx in self._stored_chunks(self.path / "vault.zarr" / name / "c", array.ndim):
            region = tuple(
                slice(i * c, (i + 1) * c) for i, c in zip(idx[: len(grid)], array.chunks[: len(grid)])
            )
            mask[region] = True
        return mask

    @staticmethod
    def _stored_chunks(root: Path, ndim: int) -> list[tuple[int, ...]]:
        """Chunk indices present on disk, found by listing the directory tree once."""
        found: list[tuple[int, ...]] = []

        def walk(directory: str, prefix: tuple[int, ...]) -> None:
            try:
                entries = list(os.scandir(directory))
            except OSError:
                return
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                idx = prefix + (int(entry.name),)
                if len(idx) == ndim:
                    if entry.is_file():
                        found.append(idx)
                elif entry.is_dir():
                    walk(entry.path, idx)

        walk(str(root), ())
        return found

    def snapshot(self) -> None:
        """Print a human-readable summary of the vault."""
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None

        shape_str = "x".join(map(str, self.shape))
        date = self.metadata.get("date", "—")
        tags_str = ", ".join(self.tags) if self.tags else "—"

        print(f"Kind:  {self.kind:<10}  Shape: {shape_str:<12}  Date: {date}")
        print(f"Tags:  {tags_str}")
        print(f"Path:  {self.path}")

        if self.kind == "sweep":
            print()
            axis_rows = []
            for ax in self.axes:
                for name in ax.controls:
                    unit = (ax.units or {}).get(name) or "—"
                    axis_rows.append([f"axis{ax.index}", name, ax.length, unit])
            _print_table(
                axis_rows,
                headers=["axis", "variable", "length", "unit"],
                title="Control axes",
                tabulate=tabulate,
            )
        else:
            n = self.shape[0]
            print(f"\nTime axis:  {n} sample{'s' if n != 1 else ''}")

        print()
        readout_rows = [
            [
                name,
                r.kind,
                str(r.shape) if r.shape else "()",
                r.unit if r.unit is not None else "—",
                r.contains if r.contains is not None else "—",
            ]
            for name, r in self.readouts.items()
        ]
        _print_table(
            readout_rows,
            headers=["name", "kind", "shape", "unit", "contains"],
            title="Readouts",
            tabulate=tabulate,
        )

    def __repr__(self) -> str:
        shape = "x".join(map(str, self.shape))
        return f"{type(self).__name__}({self.path.name!r}, {self.kind}, shape={shape})"


def open_vault(path: str | Path) -> Vault:
    """Open the run directory at *path* for reading."""
    return Vault(path)
