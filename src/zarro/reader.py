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
"""

from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import yaml
import zarr

__all__ = ["Axis", "Readout", "Vault", "VaultError", "open_vault"]


_DIM_NAMES = ("sweep", "step", "frame", "volume")


def _dim_name(d: int, n_dims: int) -> str:
    """Human-readable role name for grid dimension *d* out of *n_dims* total.

    The last axis (highest index) is always "sweep", the second-to-last is
    "step", etc. — matching zarro's writing convention where the innermost
    loop has the highest axis index.
    """
    reverse = n_dims - 1 - d
    return _DIM_NAMES[reverse] if reverse < len(_DIM_NAMES) else f"dim{d}"


class VaultError(Exception):
    """The directory is not a readable zarro run."""


@dataclass
class Axis:
    """One control variable and its sweep coordinates.

    ``run.axes`` is a ``dict[str, Axis]`` keyed by variable name.
    For a multi-control axis (two variables swept together on the same grid
    dimension), both entries share the same ``dim`` value.
    """

    name: str
    values: np.ndarray   # coordinate array, shape (length,)
    unit: str | None
    length: int
    dim: int             # index of the grid dimension this variable spans



@dataclass(frozen=True)
class Readout:
    """One recorded readout. ``shape`` is the trailing shape: ``()`` for a scalar."""

    name: str
    kind: str  # "scalar" | "trace" | "image"
    shape: tuple[int, ...]
    unit: str | list[str] | None = None
    contains: str | list[str] | None = None
    trailing_dims: tuple[str, ...] = ()
    # Ordered trailing dimension names (e.g. ("component", "freq")).
    # Coordinate values are in Vault.trailing_coords[name][dim_name].


def load_yaml(path: Path) -> dict:
    """Return the file's mapping, or ``{}`` if it is missing or half-written."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _infer_kind(trailing: tuple[int, ...]) -> str:
    return ("scalar", "trace", "image")[min(len(trailing), 2)]


def _print_table(rows, headers, title) -> None:
    print(title)
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def _legacy_manifest(group, metadata: dict, kind: str, control: dict | None, readouts: dict | None) -> dict:
    """Build a manifest for vaults written before ``attrs["zarro"]`` existed.

    All old-format handling lives here: list- and dict-style control metadata,
    the ``metadata["schema"]`` nesting, the ``control=``/``readouts=`` fallbacks
    and shape/kind inference from the arrays themselves.
    """
    schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}

    axes = []
    if kind == "sweep":
        control = metadata.get("control") or schema.get("control") or control
        if not control:
            raise VaultError("no control axes in metadata.yaml, the zarr attributes or the arguments")
        for key in sorted(control, key=lambda k: int(str(k).removeprefix("axis"))):
            dim = int(str(key).removeprefix("axis"))
            value = control[key]   # old: ["Vch", "Vm"]; new: {"Vch": {"unit": "V"}, ...}
            for name in value:
                unit = None if isinstance(value, list) else (value[name] or {}).get("unit")
                axes.append({"name": name, "dim": dim, "unit": unit})

    specs = metadata.get("readout") or schema.get("readout") or readouts
    if not specs:
        taken = {a["name"] for a in axes} | {"_time"}
        specs = {n: {} for n in (list(metadata.get("readouts") or []) or
                                 [n for n in group.array_keys() if n not in taken and "." not in n])}
    lead = len({a["dim"] for a in axes}) if kind == "sweep" else 1
    out = []
    for name, spec in specs.items():
        if name not in group:
            continue
        shape = tuple(group[name].shape[lead:])
        out.append({
            "name": name,
            "kind": spec.get("kind") or _infer_kind(shape),
            "shape": list(shape),
            "unit": spec.get("unit"),
            "contains": spec.get("contains"),
            "trailing_dims": spec.get("trailing_dims") or [],
        })
    return {"version": 1, "kind": kind, "axes": axes, "readouts": out}


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
        The structure itself (axes, readouts) comes from the ``attrs["zarro"]`` manifest
        written at init, or from :func:`_legacy_manifest` for older vaults.
    tags : list of str
    axes : ``dict[str, Axis]`` keyed by variable name (empty for monitors).
        ``run.axes["Vch"].values``, ``.unit``, ``.length``, ``.dim``.
        Variables sharing a grid dimension have the same ``.dim`` value.
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

        attrs = dict(self._group.attrs)
        manifest = attrs.pop("zarro", None)
        self.metadata = {**attrs, **load_yaml(self.path / "metadata.yaml")}
        self.tags = list(self.metadata.get("tags") or [])
        self.kind = "monitor" if "_time" in self._group else "sweep"

        self.axes: dict[str, Axis] = {}
        self.readouts: dict[str, Readout] = {}
        self.trailing_coords: dict[str, dict[str, np.ndarray]] = {}
        self.trailing_units: dict[str, dict[str, list]] = {}
        # {readout_name: {dim_name: ["Hz","dBm","deg"]}} — per-label units for component dims

        if manifest is None:
            try:
                manifest = _legacy_manifest(self._group, self.metadata, self.kind, control, readouts)
            except VaultError as exc:
                raise VaultError(f"{self.path}: {exc}") from None
        self._build_axes(manifest["axes"])
        self._build_readouts(manifest["readouts"])

    # ── structure ─────────────────────────────────────────────────────

    def _build_axes(self, axes: list[dict]) -> None:
        missing = [a["name"] for a in axes if a["name"] not in self._group]
        if missing:
            raise VaultError(f"{self.path}: coordinate array(s) {missing} not found in vault.zarr")
        for a in axes:
            values = np.asarray(self._group[a["name"]][:])
            self.axes[a["name"]] = Axis(
                name=a["name"], values=values, unit=a.get("unit"), length=len(values), dim=a["dim"]
            )

    def _build_readouts(self, readouts: list[dict]) -> None:
        for r in readouts:
            name = r["name"]
            if name not in self._group:
                continue
            dims = [td for td in r.get("trailing_dims") or [] if isinstance(td, dict)]
            self.readouts[name] = Readout(
                name=name,
                kind=r["kind"],
                shape=tuple(r["shape"]),
                unit=r.get("unit"),
                contains=r.get("contains"),
                trailing_dims=tuple(td["name"] for td in dims),
            )
            for td in dims:
                dim = td["name"]
                if f"{name}.{dim}" in self._group:     # numeric coords live in their own array
                    self.trailing_coords.setdefault(name, {})[dim] = np.asarray(self._group[f"{name}.{dim}"][:])
                elif isinstance(td.get("values"), list):   # string labels live in the manifest
                    self.trailing_coords.setdefault(name, {})[dim] = np.asarray(td["values"])
                if isinstance(td.get("units"), list):
                    self.trailing_units.setdefault(name, {})[dim] = td["units"]

    @property
    def shape(self) -> tuple[int, ...]:
        """Sweep grid shape, or ``(n_samples,)`` for a monitor (n grows while it runs)."""
        if self.kind == "monitor":
            return (int(self._group["_time"].shape[0]),)
        dim_length: dict[int, int] = {}
        for ax in self.axes.values():
            dim_length.setdefault(ax.dim, ax.length)
        return tuple(dim_length[d] for d in sorted(dim_length))

    @property
    def dim_names(self) -> list[str]:
        """Ordered role names for each grid dimension.

        The last entry is always ``"sweep"`` (innermost loop), working outward:
        ``["sweep"]``, ``["step", "sweep"]``, ``["frame", "step", "sweep"]``, …
        """
        n = len(self.shape)
        return [_dim_name(d, n) for d in range(n)]

    @property
    def time(self) -> np.ndarray:
        """Unix timestamps of a monitor run, one per sample."""
        if self.kind != "monitor":
            raise VaultError("only monitor runs have a time axis")
        return np.asarray(self._group["_time"][:])

    # ── data ──────────────────────────────────────────────────────────

    def read(self, name: str, selection=Ellipsis, *, component: str | None = None) -> np.ndarray:
        """Read a readout, indexed like the sweep grid, with ``0.0`` where nothing is written yet.

        Parameters
        ----------
        component : str, optional
            For IMAGE readouts whose ``contains`` is a list of named layers,
            select one layer by name.  Returns the array with that layer's
            trailing dimension removed.  Example: ``component="mag"`` on a
            readout with ``contains=['fpts', 'mag', 'phase']`` and shape
            ``(3, 1601)`` gives an array of shape ``(sweep_grid..., 1601)``.
        """
        raw = np.asarray(self._group[name][selection])
        if component is not None:
            raw = self._select_component(name, component, raw)
        return raw

    def _select_component(self, name: str, component: str, data: np.ndarray) -> np.ndarray:
        r = self.readouts.get(name)
        contains = r.contains if r is not None else None
        if not isinstance(contains, list):
            # fall back to trailing_coords "component" labels (set via trailing_dims)
            comp_labels = self.trailing_coords.get(name, {}).get("component")
            if comp_labels is not None:
                contains = list(comp_labels)
            else:
                raise VaultError(
                    f"readout '{name}' has no named components "
                    f"(contains={contains!r}); component= is only valid when contains is a list"
                )
        if component not in contains:
            raise VaultError(
                f"component '{component}' not found in '{name}'; "
                f"available: {contains}"
            )
        idx = contains.index(component)
        # Layer axis = first trailing dim = data.ndim - n_trailing.
        # Works for TRACE with shape=(n_layers,) and IMAGE with shape=(n_layers, n_pts).
        n_trailing = len(r.shape) if r is not None else 2
        return np.take(data, idx, axis=data.ndim - n_trailing)

    def values(self, name: str, *, component: str | None = None) -> np.ndarray:
        """Whole readout as floats, NaN where nothing has been written yet.

        Parameters
        ----------
        component : str, optional
            Select one named layer of an IMAGE readout (see :meth:`read`).
        """
        raw = self.read(name, component=component)
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
        """Chunk indices present on disk."""
        parts = (p.relative_to(root).parts for p in root.rglob("*") if p.is_file())
        return [tuple(map(int, q)) for q in parts if len(q) == ndim and all(x.isdigit() for x in q)]

    def _grid_dims_and_coords(self) -> tuple[list[str], dict]:
        """Names and coordinates of the leading (grid or time) dimensions."""
        if self.kind == "monitor":
            return ["time"], {"time": ("time", self.time)}
        dim_axes: dict[int, list] = {}
        for ax in self.axes.values():
            dim_axes.setdefault(ax.dim, []).append(ax)
        dims, coords = [], {}
        for d in sorted(dim_axes):
            on_dim = dim_axes[d]
            dim_name = on_dim[0].name if len(on_dim) == 1 else f"dim{d}"
            dims.append(dim_name)
            coords.update({ax.name: (dim_name, ax.values) for ax in on_dim})
        return dims, coords

    def to_xarray(self, name: str, *, component: str | None = None,
                  trailing_coords: dict | None = None):
        """Return a readout as an :class:`xarray.DataArray` with labeled axes.

        Coordinate arrays from :attr:`axes` are attached as named coordinates.
        Monitor runs get a leading ``"time"`` dimension whose coordinate is the
        raw Unix timestamps (seconds).
        NaN is used for unwritten grid points (same as :meth:`values`).
        Units, when present, are stored in ``.attrs["units"]``.

        Trailing dimension names and coordinate values are resolved in this order:

        1. ``trailing_dims`` stored in the vault (written via ``ReadoutSpecs.trailing_dims``).
        2. ``trailing_coords`` keyword argument — a ``{dim_name: array}`` dict
           supplied by the caller (useful for older vaults that predate option 1).
        3. ``contains`` as a list → first trailing dim named ``"component"``
           with those labels as coordinate values.
        4. Generic fallback: ``"sample"`` (1 trailing dim) or ``"row"`` / ``"col"``
           (2 trailing dims, unselected IMAGE).

        Parameters
        ----------
        component : str, optional
            Select one named layer of an IMAGE readout before wrapping (see
            :meth:`read`).  The layer dimension is removed from the result.
        trailing_coords : dict, optional
            ``{dim_name: array}`` mapping to attach as coordinates on trailing
            dimensions.  Applied in insertion order to the trailing axes.

        Raises
        ------
        ImportError
            If xarray is not installed.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for to_xarray(): pip install xarray"
            ) from None

        data = self.values(name, component=component)
        rd = self.readouts.get(name)

        dims, coords = self._grid_dims_and_coords()
        n_grid = len(dims)

        # ── trailing dims ─────────────────────────────────────────────────
        n_trailing = data.ndim - n_grid

        # Priority 1: trailing_dims stored in vault
        vault_td = rd.trailing_dims if rd is not None else ()
        # After component= selection the component dim is gone; remove it
        if component is not None and vault_td and vault_td[0] == "component":
            vault_td = vault_td[1:]
        vault_tc = self.trailing_coords.get(name, {})

        if vault_td:
            td_names = list(vault_td[:n_trailing])
            # pad with generic names if metadata has fewer entries than trailing dims
            while len(td_names) < n_trailing:
                td_names.append(f"dim_{len(dims) + len(td_names)}")
            for dn in td_names:
                dims.append(dn)
                val = vault_tc.get(dn) if dn in vault_tc else (
                    trailing_coords.get(dn) if trailing_coords else None
                )
                if val is not None:
                    coords[dn] = np.asarray(val)

        # Priority 2: caller-supplied trailing_coords (old vaults)
        elif trailing_coords:
            for dn, val in list(trailing_coords.items())[:n_trailing]:
                dims.append(dn)
                coords[dn] = np.asarray(val)
            # pad if fewer entries than trailing dims
            while len(dims) - n_grid < n_trailing:
                dims.append(f"dim_{len(dims)}")

        # Priority 3: contains as list → component dim
        elif rd is not None and isinstance(rd.contains, list) and component is None:
            dims.append("component")
            coords["component"] = np.asarray(rd.contains)
            if n_trailing == 2:
                dims.append("sample")

        # Priority 4: generic fallback
        elif n_trailing == 1:
            dims.append("sample")
        elif n_trailing == 2:
            dims.extend(["row", "col"])

        # ── build DataArray ───────────────────────────────────────────────
        da = xr.DataArray(data, dims=dims, coords=coords)

        for ax in self.axes.values():
            if ax.unit and ax.name in da.coords:
                da.coords[ax.name].attrs["units"] = ax.unit
        if self.kind == "monitor":
            da.coords["time"].attrs["units"] = "s (unix epoch)"
        if rd is not None and rd.unit:
            da.attrs["units"] = rd.unit
        da.attrs["long_name"] = name
        if component is not None:
            da.attrs["component"] = component

        # Attach per-label units for component dims (e.g. {"fpts":"Hz","mag":"dBm"})
        ru = self.trailing_units.get(name, {})
        for dim_name, unit_list in ru.items():
            if component is not None and dim_name == "component":
                continue  # component dim was removed from the array
            tc_labels = self.trailing_coords.get(name, {}).get(dim_name)
            if tc_labels is not None and len(unit_list) == len(tc_labels):
                da.attrs[f"{dim_name}_units"] = {
                    str(label): u for label, u in zip(tc_labels, unit_list)
                }

        return da

    def snapshot(self) -> None:
        """Print a human-readable summary of the vault."""
        shape_str = "x".join(map(str, self.shape))
        date = self.metadata.get("date", "—")
        tags_str = ", ".join(self.tags) if self.tags else "—"
        ndim = len(self.shape)
        if self.kind == "sweep":
            outermost = _dim_name(0, ndim)   # slowest loop = highest-level name
            kind_str = f"{ndim}D {outermost} sweep" if ndim > 1 else "1D sweep"
        else:
            kind_str = self.kind

        print(f"Kind:  {kind_str:<18}  Shape: {shape_str:<12}  Date: {date}")
        print(f"Tags:  {tags_str}")
        print(f"Path:  {self.path}")

        if self.kind == "sweep":
            print()
            axis_rows = []
            for ax in self.axes.values():
                axis_rows.append([_dim_name(ax.dim, ndim), ax.name, ax.length, ax.unit or "—"])
            _print_table(
                axis_rows,
                headers=["role", "variable", "length", "unit"],
                title="Control axes",
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
        )

    def __repr__(self) -> str:
        shape = "x".join(map(str, self.shape))
        ndim = len(self.shape)
        if self.kind == "sweep":
            outermost = _dim_name(0, ndim)
            kind_str = f"{ndim}D {outermost} sweep" if ndim > 1 else "1D sweep"
        else:
            kind_str = self.kind
        return f"{type(self).__name__}({self.path.name!r}, {kind_str}, shape={shape})"


def open_vault(path: str | Path) -> Vault:
    """Open the run directory at *path* for reading.

    *path* may be an exact directory path **or** a 4-digit experiment ID (as a
    string like ``"0007"`` or an int like ``7``).  When the exact path does not
    exist and its final component is a 4-digit ID, the parent directory is
    searched for a folder whose name starts with that ID (e.g. ``"0007_blabla"``
    also matches ``"0007"``).  A :class:`VaultError` is raised if no match or
    more than one match is found.
    """
    if isinstance(path, int):
        path = Path(f"{path:04d}")
    path = Path(path)
    if not path.exists():
        stem = path.name
        if re.fullmatch(r"\d{4}", stem):
            parent = path.parent
            matches = [
                d for d in parent.iterdir()
                if d.is_dir() and (d.name == stem or d.name.startswith(stem + "_"))
            ]
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) == 0:
                raise VaultError(
                    f"no directory matching '{stem}*' found in {parent}"
                )
            else:
                names = sorted(m.name for m in matches)
                raise VaultError(
                    f"ambiguous experiment ID '{stem}': found {names}"
                )
    return Vault(path)
