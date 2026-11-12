# zarro Tutorial

A comprehensive guide to the `zarro` package for structured experimental data saving.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concepts](#2-core-concepts)
   - [DataKind](#21-datakind)
   - [WriteType](#22-writetype)
3. [Building a Schema](#3-building-a-schema)
   - [ControlVar](#31-controlvar)
   - [AxisSpecs](#32-axisspecs)
   - [ReadoutSpecs](#33-readoutspecs)
   - [MeasurementSchema](#34-measurementschema)
4. [Writing Data with ZarrWriter](#4-writing-data-with-zarrwriter)
   - [Initialization](#41-initialization)
   - [Pointwise writes](#42-pointwise-writes)
   - [Tracewise writes](#43-tracewise-writes)
   - [Imagewise writes](#44-imagewise-writes)
   - [Write all at once](#45-write-all-at-once)
5. [Data Kinds in Practice](#5-data-kinds-in-practice)
   - [SCALAR readouts](#51-scalar-readouts)
   - [TRACE readouts](#52-trace-readouts)
   - [IMAGE readouts](#53-image-readouts)
   - [Mixed readouts](#54-mixed-readouts)
6. [Metadata](#6-metadata)
   - [write_metadata](#61-write_metadata)
   - [write_yaml](#62-write_yaml)
   - [Tags](#63-tags)
7. [Adding Arrays After Initialization](#7-adding-arrays-after-initialization)
8. [Reading Data](#8-reading-data)
9. [Experiment ID Management](#9-experiment-id-management)
10. [NpzWriter](#10-npzwriter)
11. [Schemaless Writing](#11-schemaless-writing)
12. [StreamingWriter — Open-Ended Time Series](#12-streamingwriter--open-ended-time-series)
    - [When to use it](#121-when-to-use-it)
    - [Initialization](#122-initialization)
    - [Appending samples](#123-appending-samples)
    - [Flushing and closing](#124-flushing-and-closing)
    - [Reading streaming data](#125-reading-streaming-data)
    - [Mixed readout shapes](#126-mixed-readout-shapes)
    - [Manual timestamps](#127-manual-timestamps)
13. [Sharding and Performance](#13-sharding-and-performance)
14. [API Reference](#14-api-reference)
15. [Version Management](#15-version-management)

---

## 1. Overview

`zarro` provides a structured interface for saving experimental data from parameter sweeps. It wraps the [Zarr v3](https://zarr.readthedocs.io/) format with:

- A **declarative schema** that describes what axes you sweep and what you measure
- **Write helpers** that accept natural index tuples and dicts of arrays, hiding all Zarr indexing
- **Automatic chunking and sharding** tuned to your write pattern for efficient streaming I/O
- **Metadata** stored as both Zarr attributes (JSON) and a human-readable `metadata.yaml`

The typical workflow is:

```
define schema → create writer → run sweep loop → write metadata
```

---

## 2. Core Concepts

### 2.1 DataKind

`DataKind` is a `StrEnum` that describes the shape of a single measurement readout:

| Value | Meaning | Array shape per point |
|---|---|---|
| `DataKind.SCALAR` | Single number | `()` — a scalar |
| `DataKind.TRACE` | 1-D array | `(N,)` — e.g. a frequency sweep |
| `DataKind.IMAGE` | 2-D array | `(H, W)` — e.g. a detector frame |

```python
from zarro import DataKind

DataKind.SCALAR  # "scalar"
DataKind.TRACE   # "trace"
DataKind.IMAGE   # "image"
```

### 2.2 WriteType

`WriteType` is a `StrEnum` that tells the writer how data will arrive during the sweep loop. This controls chunk and shard geometry for optimal I/O:

| Value | When to use | Index tuple length |
|---|---|---|
| `WriteType.POINTWISE` | Write one point at a time | `len(control_shape)` |
| `WriteType.TRACEWISE` | Write one trace (row) at a time | `len(control_shape) - 1` |
| `WriteType.IMAGEWISE` | Write one image (plane) at a time | `len(control_shape) - 2` |
| `WriteType.ALL` | Write the entire array at once | — (no index needed) |

```python
from zarro import WriteType

WriteType.POINTWISE   # "pointwise"
WriteType.TRACEWISE   # "tracewise"
WriteType.IMAGEWISE   # "imagewise"
WriteType.ALL         # "all"
```

---

## 3. Building a Schema

### 3.1 ControlVar

`ControlVar` describes a single swept parameter — its name, the array of values it takes, and optional metadata.

```python
from zarro import ControlVar
import numpy as np

# Basic usage
freq = ControlVar(
    name="freq",
    values=np.linspace(4e9, 8e9, 1601),
    unit="Hz",
)

# With instrument tag
gate_voltage = ControlVar(
    name="Vgt",
    values=np.linspace(-0.5, 0.5, 101),
    unit="V",
    instrument="Keithley2400",
)

# Hysteresis sweep: values array is automatically doubled [forward, reverse]
Vbias = ControlVar(
    name="Vbias",
    values=np.linspace(0, 1, 50),
    unit="V",
    reverse=True,   # stored array becomes shape (100,): forward then backward
)
print(len(Vbias.values))  # 100
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable name (used as array key in vault) |
| `values` | `np.ndarray` | Sweep values |
| `unit` | `str` | Physical unit — stored in metadata |
| `reverse` | `bool` | If True, the values array is doubled: forward then backward sweep |
| `instrument` | `str` | Source instrument — stored in metadata |

### 3.2 AxisSpecs

`AxisSpecs` groups one or more `ControlVar` instances that share the **same length** into a single axis. Multiple control vars on the same axis are swept together (e.g. X and Y on a 2D stage).

```python
from zarro import AxisSpecs, ControlVar
import numpy as np

# Single variable per axis (most common)
axis0 = AxisSpecs([ControlVar("power", np.linspace(-30, 0, 20), unit="dBm")])
axis1 = AxisSpecs([ControlVar("freq",  np.linspace(4e9, 8e9, 1601), unit="Hz")])

# Two variables sharing the same axis (swept together)
x = ControlVar("x", np.linspace(0, 1, 50), unit="mm")
y = ControlVar("y", np.linspace(0, 1, 50), unit="mm")
axis_xy = AxisSpecs([x, y])   # x and y must have same length
```

All `ControlVar` instances in a single `AxisSpecs` must have the same `.length`. A `ValueError` is raised otherwise.

### 3.3 ReadoutSpecs

`ReadoutSpecs` describes a single measurement channel — its name, data kind, and optional metadata.

```python
from zarro import ReadoutSpecs, DataKind

# Scalar readout
lockin_x = ReadoutSpecs(
    name="lockin_X",
    kind=DataKind.SCALAR,
    unit="V",
    contains="in-phase signal",
)

# Trace readout (1601-point VNA trace)
s21 = ReadoutSpecs(
    name="S21",
    kind=DataKind.TRACE,
    shape=(1601,),
    unit="dBm",
    contains="transmission magnitude",
)

# Image readout (480 x 640 detector)
frame = ReadoutSpecs(
    name="camera",
    kind=DataKind.IMAGE,
    shape=(480, 640),
    unit="counts",
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Array key in vault |
| `kind` | `DataKind` | SCALAR, TRACE, or IMAGE |
| `unit` | `str \| list[str]` | Physical unit(s) |
| `shape` | `tuple[int, ...]` | **Required** for TRACE and IMAGE: trailing dimension sizes |
| `contains` | `str \| list[str]` | Human description of what is measured |

### 3.4 MeasurementSchema

`MeasurementSchema` combines control axes and readouts into a complete description of the measurement.

```python
from zarro import MeasurementSchema, AxisSpecs, ReadoutSpecs, ControlVar, DataKind, WriteType
import numpy as np

power  = ControlVar("power", np.linspace(-30, 0, 20),     unit="dBm")
freq   = ControlVar("freq",  np.linspace(4e9, 8e9, 1601), unit="Hz")

schema = MeasurementSchema(
    control=[AxisSpecs([power]), AxisSpecs([freq])],
    readout=[
        ReadoutSpecs("S21_mag",   DataKind.SCALAR, unit="dBm"),
        ReadoutSpecs("S21_phase", DataKind.SCALAR, unit="deg"),
        ReadoutSpecs("S21_trace", DataKind.TRACE,  shape=(1601,), unit="dBm"),
    ],
    writetype=WriteType.POINTWISE,
)

print(schema.axis_names)    # {'axis0': ['power'], 'axis1': ['freq']}
print(schema.readout_names) # ['S21_mag', 'S21_phase', 'S21_trace']
print(schema.meta)          # full metadata dict used in file headers
```

The resulting **control shape** is `(20, 1601)` — one dimension per `AxisSpecs`.

---

## 4. Writing Data with ZarrWriter

### 4.1 Initialization

```python
from zarro import ZarrWriter

writer = ZarrWriter(
    root="./data/run_001",          # directory to store vault.zarr and metadata.yaml
    schema=schema,                  # MeasurementSchema defined above
    tags=["vna", "cooldown_3"],     # optional free-form tags
    dtype="float32",                # numpy dtype for all data arrays
    overwrite=True,                 # overwrite existing files
    initialize_arrays=True,         # create Zarr arrays immediately
    target_file_size_bytes=5*1024*1024,  # target ~5 MB per shard
)
```

Setting `initialize_arrays=True` creates:
- One coordinate array per `ControlVar` (pre-filled with sweep values)
- One data array per `ReadoutSpecs` (pre-allocated, filled with zeros)
- A `metadata.yaml` file

If `initialize_arrays=False`, you can call `writer.add_array()` later to build up the vault manually.

### 4.2 Pointwise writes

Use when acquiring one scalar/trace/image at each grid point.

```python
# schema.writetype = WriteType.POINTWISE
# control_shape = (N_power, N_freq)

for i in range(N_power):
    for j in range(N_freq):
        mag, phase = vna.get_point()
        writer.write_point(
            index=(i, j),
            data={"S21_mag": mag, "S21_phase": phase},
        )
```

The `index` tuple must have length `len(control_shape)`.

### 4.3 Tracewise writes

Use when the inner axis is always swept as a unit (e.g. a VNA returns a full trace per trigger).

```python
# schema.writetype = WriteType.TRACEWISE
# control_shape = (N_power, N_freq)
# index length = len(control_shape) - 1 = 1

schema = MeasurementSchema(
    control=[AxisSpecs([power]), AxisSpecs([freq])],
    readout=[ReadoutSpecs("S21", DataKind.TRACE, shape=(1601,))],
    writetype=WriteType.TRACEWISE,
)
writer = ZarrWriter(root="./data/run_002", schema=schema,
                    overwrite=True, initialize_arrays=True)

for i in range(N_power):
    trace = vna.sweep()              # shape (1601,)
    writer.write_trace(
        index=(i,),
        data={"S21": trace},
    )
```

### 4.4 Imagewise writes

Use when the two innermost axes are always acquired together (e.g. a 2D raster image).

```python
# control_shape = (N_z, N_y, N_x)
# index length = len(control_shape) - 2 = 1

schema = MeasurementSchema(
    control=[AxisSpecs([z_var]), AxisSpecs([y_var]), AxisSpecs([x_var])],
    readout=[ReadoutSpecs("image", DataKind.SCALAR)],
    writetype=WriteType.IMAGEWISE,
)
writer = ZarrWriter(root="./data/run_003", schema=schema,
                    overwrite=True, initialize_arrays=True)

for k in range(N_z):
    img = camera.capture()           # shape (N_y, N_x)
    writer.write_image(
        index=(k,),
        data={"image": img},
    )
```

### 4.5 Write all at once

Use when data is generated in memory before saving (simulations, post-processing).

```python
schema = MeasurementSchema(
    control=[AxisSpecs([power]), AxisSpecs([freq])],
    readout=[ReadoutSpecs("S21", DataKind.SCALAR)],
    writetype=WriteType.ALL,
)
writer = ZarrWriter(root="./data/run_004", schema=schema,
                    overwrite=True, initialize_arrays=True)

full_data = simulate()               # shape (N_power, N_freq)
writer.write_all({"S21": full_data})
```

---

## 5. Data Kinds in Practice

### 5.1 SCALAR readouts

Each write delivers a single number per readout. The stored array shape equals the control shape.

```
control_shape = (20, 101)
S21_mag array shape → (20, 101)
```

### 5.2 TRACE readouts

Each write delivers a 1-D array per readout. The stored array shape is `control_shape + (trace_length,)`.

```
control_shape = (20, 101), trace shape (1601,)
S21 array shape → (20, 101, 1601)
```

```python
ReadoutSpecs("S21", DataKind.TRACE, shape=(1601,))
```

### 5.3 IMAGE readouts

Each write delivers a 2-D array per readout. The stored array shape is `control_shape + (H, W)`.

```
control_shape = (10,), image shape (480, 640)
frame array shape → (10, 480, 640)
```

```python
ReadoutSpecs("frame", DataKind.IMAGE, shape=(480, 640))
```

### 5.4 Mixed readouts

A single schema can contain multiple readout kinds simultaneously:

```python
schema = MeasurementSchema(
    control=[AxisSpecs([temp]), AxisSpecs([freq])],
    readout=[
        ReadoutSpecs("resistance",  DataKind.SCALAR, unit="Ohm"),
        ReadoutSpecs("S11",         DataKind.TRACE,  shape=(801,), unit="dBm"),
        ReadoutSpecs("noise_image", DataKind.IMAGE,  shape=(64, 64), unit="V/rtHz"),
    ],
    writetype=WriteType.POINTWISE,
)
```

In `write_point()`, supply whichever readouts were measured at that point; you do not have to provide all of them in a single call.

---

## 6. Metadata

### 6.1 write_metadata

`write_metadata()` writes both Zarr group attributes (JSON) and `metadata.yaml`. It automatically includes:
- `date` — ISO 8601 timestamp of when this method was called
- `schema` — control axis names and readout specs from the schema
- `tags` — the writer's tag list

```python
writer.write_metadata(meta={
    "operator": "alice",
    "sample":   "chip_B2",
    "cooldown": 3,
    "fridge":   "BlueFors_XLD",
})
```

Call `write_metadata()` once, typically **after** the sweep completes.

### 6.2 write_yaml

`write_yaml()` writes only the YAML file without touching Zarr attributes. Useful when you want a quick on-disk record without committing to Zarr attribute format.

```python
writer.write_yaml(meta={"status": "in_progress"})
```

If `overwrite=False` (the default) and `metadata.yaml` already exists, a `UserWarning` is emitted and the file is not overwritten.

### 6.3 Tags

Tags are free-form strings that can be set at construction time or mutated before calling `write_metadata()`:

```python
writer = ZarrWriter(root="...", schema=schema, tags=["vna"])
# ... run sweep ...
writer.tags.append("temperature_sweep")
writer.write_metadata()
```

---

## 7. Adding Arrays After Initialization

`add_array()` lets you attach derived or auxiliary arrays to an existing vault:

```python
writer = ZarrWriter(root="./data/run_001", overwrite=True)

# Compute something from raw data
background = np.load("background.npy")
writer.add_array(
    name="background",
    value=background,
    meta={"description": "dark-frame reference"},
)
```

If `overwrite=False` and an array with that name already exists, a `KeyError` is raised.

For writing multiple arrays at once without a schema, use `dump()`:

```python
writer.dump(
    data={"x": x_array, "y": y_array, "z": z_array},
    meta={"source": "post-processing"},
)
```

For a fully NPZ-like workflow (no schema, no streaming), see [Section 11](#11-schemaless-writing).

---

## 8. Reading Data

Use `open_vault` (or `Vault` directly) to open a run directory for analysis.
The reader works on finished runs, interrupted runs, and runs still in progress.

```python
from zarro import open_vault

run = open_vault("./data/run_001")

# Kind: "sweep" for ZarrWriter runs, "monitor" for StreamingWriter runs
print(run.kind)          # "sweep"
print(run.shape)         # (20, 1601)
print(run.tags)          # ["vna", "2K"]
print(run.metadata)      # full merged dict (zarr attrs + metadata.yaml)
```

### 8.1 Control axes and coordinates

```python
# Axes, outermost first
for ax in run.axes:
    print(ax.index, ax.controls, ax.length, ax.units)
    # e.g. 0  ('power',)  20  {'power': 'dBm'}

# Coordinate values as numpy arrays
freqs  = run.coords["freq"]    # shape (1601,)
powers = run.coords["power"]   # shape (20,)

# Which axis does a coordinate belong to?
run.coord_axis["freq"]    # 1
run.coord_axis["power"]   # 0
```

### 8.2 Readout metadata

```python
for name, r in run.readouts.items():
    print(name, r.kind, r.shape, r.unit, r.contains)
    # S21_mag  scalar  ()       dBm   None
    # S21      trace   (1601,)  V     S21
    # frame    image   (64,64)  None  None
```

### 8.3 Reading data arrays

```python
# Raw array — 0.0 where nothing has been written yet
all_mag = run.read("S21_mag")           # shape (20, 1601)
row_5   = run.read("S21_mag", (5,))     # shape (1601,)
point   = run.read("S21_mag", (3, 100)) # scalar

# NaN-masked array — NaN where unwritten (use this for mid-run reads)
signal = run.values("S21_mag")          # shape (20, 1601), NaN at unwritten points
```

### 8.4 Tracking progress mid-run

`written_mask` returns a boolean grid that is True where data has been written.
It works on running sweeps without blocking the writer.

```python
mask = run.written_mask("S21_mag")       # shape == run.shape
print(f"{mask.sum()} / {mask.size} points done")

# Custom visiting order (e.g. snake sweep: row 0 L→R, row 1 R→L, ...)
mask = run.written_mask("S21_mag", order=my_order_array)
```

**How it works:**

- **Scalar** readouts are stored in plain chunks: a point is written when its
  chunk file exists on disk.
- **Trace/image** readouts are stored in shards: a point is written when any
  of its values is non-zero (since zarr fills unwritten positions with 0.0).
- In both cases, every position visited *before* the last confirmed written
  position is also counted as written (order-prefix rule), so a genuine 0.0
  inside the scanned region remains visible.

### 8.5 Monitor runs (StreamingWriter)

```python
run = open_vault("./data/mon_001")
print(run.kind)    # "monitor"
print(run.shape)   # (n_samples,) — grows while the run is live
print(run.time)    # unix timestamps, shape (n_samples,)

x_values = run.values("x")   # shape (n_samples,), always fully written
```

The vault can be opened before the writer calls `close()`.
The readout shapes are inferred automatically from the array dimensions.

### 8.6 Fallback when metadata is missing

If `metadata.yaml` is gone and zarr attributes were never written, pass the
structure explicitly:

```python
run = Vault(
    "./data/run_001",
    control={"axis0": {"freq": {"unit": "Hz"}}, "axis1": {"power": {"unit": "dBm"}}},
    readouts={"S21_mag": {"kind": "scalar", "unit": "dBm"}},
)
```

### 8.7 Low-level zarr access

The underlying zarr group is always accessible for direct numpy-style reads:

```python
import zarr
z = zarr.open("./data/run_001/vault.zarr", mode="r")
print(list(z.keys()))
all_mag = z["S21_mag"][:]     # shape (20, 1601)
```

---

## 9. Experiment ID Management

`ExperimentID` manages a directory of sequentially numbered experiment folders:

```python
from zarro import ExperimentID

eid = ExperimentID(root="./data")

print(eid.current_id())   # e.g. 3
print(eid.next_id())      # 4
print(eid.next_dir())     # PosixPath('./data/0004')
print(eid.current_dir())  # PosixPath('./data/0003')

# Look up an existing folder by integer ID or string name
print(eid.get_dir_name(2))            # PosixPath('./data/0002')
print(eid.get_dir_name("special_run")) # PosixPath('./data/special_run')
```

Typical usage in a measurement script:

```python
eid = ExperimentID(root="./data")
run_dir = eid.next_dir()

writer = ZarrWriter(root=run_dir, schema=schema,
                    overwrite=False, initialize_arrays=True)
# ... run sweep ...
writer.write_metadata(meta={"id": eid.next_id()})
```

Folder names use zero-padded 4-digit integers (`0001`, `0002`, …) so they sort correctly in file explorers.

---

## 10. NpzWriter

`NpzWriter` is a simpler alternative that saves data as a single `.npz` file. Use it when the dataset fits in memory and you do not need streaming writes or Zarr's chunked access.

```python
from zarro import NpzWriter, MeasurementSchema, AxisSpecs, ReadoutSpecs, ControlVar, DataKind
import numpy as np

schema = MeasurementSchema(
    control=[AxisSpecs([ControlVar("freq", np.linspace(1e9, 5e9, 501))])],
    readout=[ReadoutSpecs("S21", DataKind.TRACE, shape=(501,))],
)

writer = NpzWriter(
    root="./data/npz_run",
    schema=schema,
    tags=["quick_check"],
    compressed=False,   # use np.savez_compressed for smaller files
    overwrite=True,
)

# Generate or acquire data
data = {"S21": np.random.rand(501, 501).astype("float32")}

writer.write(data)
writer.write_metadata(meta={"operator": "bob"})
```

Files created:
- `data.npz` — NumPy archive, loadable with `np.load()`
- `metadata.yaml` — human-readable metadata

```python
import numpy as np
npz = np.load("./data/npz_run/data.npz")
print(list(npz.keys()))   # ['S21']
s21 = npz["S21"]          # shape (501, 501)
```

---

## 11. Schemaless Writing

`ZarrWriter` (and `NpzWriter`) work without a schema when you just need to store a collection of arrays with metadata:

```python
from zarro import ZarrWriter
import numpy as np

writer = ZarrWriter(root="./data/schemaless", overwrite=True)

# write_npzlike: store multiple arrays with metadata
writer.dump(
    data={
        "frequencies": np.linspace(1e9, 5e9, 1001),
        "S21_db":      np.random.rand(1001),
        "S21_phase":   np.random.rand(1001),
    },
    meta={
        "description": "single VNA sweep",
        "date": "2026-04-10",
    },
)
```

You can also add arrays one at a time:

```python
writer.add_array("Vgt_values", np.linspace(-1, 1, 200))
writer.add_array("conductance", G_matrix)
```

---

## 12. StreamingWriter — Open-Ended Time Series

### 12.1 When to use it

`ZarrWriter` requires the full sweep grid to be known before acquisition begins. `StreamingWriter` is the right tool when:

- The number of samples is not known in advance (e.g. event-driven acquisition, user-terminated runs)
- Data arrives asynchronously or at irregular intervals
- You are logging instrument state over time alongside measurements

The trade-off: `StreamingWriter` can only append along a single time axis. There is no 2-D grid, no tracewise/imagewise distinction — just a sequence of samples, each optionally paired with a Unix timestamp.

### 12.2 Initialization

```python
from zarro import StreamingWriter

writer = StreamingWriter(
    root="./data/timeseries_001",
    readouts={
        "lockin_X": (),        # scalar per sample → stored shape (N,)
        "lockin_Y": (),
        "S21":      (1601,),   # trace per sample  → stored shape (N, 1601)
        "frame":    (480, 640) # image per sample  → stored shape (N, 480, 640)
    },
    chunk_size=256,            # samples buffered before each disk write
    dtype="float32",
    tags=["continuous", "lock-in"],
    overwrite=True,
)
```

At init, all arrays are created with size `0` along the time axis:

| Array | Initial shape | Final shape after N samples |
|---|---|---|
| `_time` | `(0,)` | `(N,)` |
| `lockin_X` | `(0,)` | `(N,)` |
| `S21` | `(0, 1601)` | `(N, 1601)` |
| `frame` | `(0, 480, 640)` | `(N, 480, 640)` |

### 12.3 Appending samples

```python
while instrument.is_running():
    x, y  = lockin.read_xy()
    trace = vna.sweep()          # shape (1601,)

    writer.append({
        "lockin_X": x,
        "lockin_Y": y,
        "S21":      trace,
    })
```

Each call to `append()` buffers the sample in memory. When the buffer reaches `chunk_size` samples, `flush()` is called automatically — resizing the Zarr arrays and writing the batch in one operation.

You do not have to supply every readout on every call. Readouts omitted from a call are simply not written for that sample, leaving zeros in the pre-allocated region. For sparse acquisition, supply all readouts each call and use `_time` to align them later.

### 12.4 Flushing and closing

`flush()` can be called manually at any time to commit buffered data:

```python
writer.flush()   # commit whatever is currently buffered
```

This is useful if you want partial results visible on disk during a long run, or as a safety measure before a risky operation.

`close()` flushes any remaining buffer and writes metadata:

```python
writer.close(meta={
    "operator": "alice",
    "sample":   "chip_B2",
    "notes":    "terminated after 3000 samples",
})
```

`close()` writes both Zarr group attributes and `metadata.yaml`, consistent with `ZarrWriter.write_metadata()`. Always call it at the end of acquisition.

> **Data loss on crash:** samples buffered since the last `flush()` are lost if the process terminates unexpectedly. Set `chunk_size=1` for maximum durability (one disk write per sample, slower), or call `flush()` explicitly at checkpoints.

### 12.5 Reading streaming data

The vault is a standard Zarr v3 group:

```python
import zarr
import numpy as np
from datetime import datetime

z = zarr.open("./data/timeseries_001/vault.zarr", mode="r")
print(list(z.keys()))
# ['_time', 'lockin_X', 'lockin_Y', 'S21']

# Timestamps
times = z["_time"][:]           # shape (N,), Unix float64
dt    = [datetime.fromtimestamp(t) for t in times]

# Scalar channels
x = z["lockin_X"][:]            # shape (N,)
y = z["lockin_Y"][:]

# Trace channel
traces = z["S21"][:]            # shape (N, 1601)
last_trace = z["S21"][-1]      # most recent trace, shape (1601,)

# Metadata
attrs = dict(z.attrs)
print(attrs["count"])           # total number of samples written
print(attrs["tags"])
```

### 12.6 Mixed readout shapes

A single `StreamingWriter` can hold scalars, traces, and images simultaneously:

```python
writer = StreamingWriter(
    root="./data/mixed",
    readouts={
        "temperature": (),           # thermocouple reading
        "resistance":  (),           # lock-in output
        "spectrum":    (1024,),      # spectrometer trace
        "camera":      (512, 512),   # camera frame
    },
    chunk_size=64,
)

for _ in range(n_steps):
    writer.append({
        "temperature": thermo.read(),
        "resistance":  lockin.read(),
        "spectrum":    spectrometer.acquire(),
        "camera":      cam.capture(),
    })

writer.close()
```

### 12.7 Manual timestamps

Pass your own timestamps when the acquisition hardware provides them, or when replaying logged data:

```python
import time

t0 = time.time()

for sample in hardware.stream():
    writer.append(
        data={"signal": sample.value},
        timestamp=sample.hardware_timestamp,   # float, Unix epoch
    )

writer.close()
```

If `timestamp` is omitted, `datetime.now().timestamp()` is used.

---

## 13. Sharding and Performance

`ZarrWriter` automatically chooses chunk and shard shapes based on `writetype` and `target_file_size_bytes`.

**Chunk shape** is aligned with the write pattern so each write touches exactly one chunk:

| WriteType | Chunk shape (control dims) |
|---|---|
| POINTWISE | `(1, 1, …, 1)` |
| TRACEWISE | `(1, 1, …, N_last)` |
| IMAGEWISE | `(1, 1, …, N_y, N_x)` |
| ALL | `control_shape` |

**Shard shape** aggregates multiple chunks into one file, reducing filesystem overhead. The number of shards is estimated from `target_file_size_bytes`:

```
num_shards = total_bytes // target_file_size_bytes
```

Tune `target_file_size_bytes` to match your storage system:
- Local NVMe SSD → 1–5 MB
- Network-attached storage (NAS) → 10–50 MB
- Object storage (S3) → 50–200 MB

```python
writer = ZarrWriter(
    root="./data/nas_run",
    schema=schema,
    target_file_size_bytes=20 * 1024 * 1024,   # 20 MB shards for NAS
    initialize_arrays=True,
)
```

---

## 14. API Reference

### DataKind

```
DataKind.SCALAR   # single value per point
DataKind.TRACE    # 1-D array per point
DataKind.IMAGE    # 2-D array per point
```

### WriteType

```
WriteType.POINTWISE   # write one point at a time
WriteType.TRACEWISE   # write one trace at a time
WriteType.IMAGEWISE   # write one image plane at a time
WriteType.ALL         # write entire array at once
```

### ControlVar

```python
ControlVar(
    name: str,
    values: np.ndarray,
    unit: str = None,
    reverse: bool = False,    # doubles array for hysteresis
    instrument: str = None,
)
# Property: .length → int
```

### AxisSpecs

```python
AxisSpecs(controls: list[ControlVar])
# Property: .length → int
# Property: .names  → list[str]
```

### ReadoutSpecs

```python
ReadoutSpecs(
    name: str,
    kind: DataKind,
    unit: str | list[str] = None,
    shape: tuple[int, ...] = None,    # required for TRACE and IMAGE
    contains: str | list[str] = None,
)
```

### MeasurementSchema

```python
MeasurementSchema(
    control: list[AxisSpecs],
    readout: list[ReadoutSpecs],
    writetype: WriteType = WriteType.POINTWISE,
)
# Property: .axis_names    → dict[str, list[str]]
# Property: .readout_names → list[str]
# Property: .meta          → dict  (auto-generated)
```

### ZarrWriter

```python
ZarrWriter(
    root: str | Path,
    schema: MeasurementSchema = None,
    tags: list[str] = [],
    dtype: str = "float32",
    overwrite: bool = False,
    initialize_arrays: bool = False,
    target_file_size_bytes: int = 5*1024*1024,
)

# Write methods
writer.write_point(index: tuple, data: dict)
writer.write_trace(index: tuple, data: dict)
writer.write_image(index: tuple, data: dict)
writer.write_all(data: dict)

# Metadata
writer.write_metadata(meta: dict = None)
writer.write_yaml(meta: dict = None)

# Array management
writer.add_array(name: str, value: np.ndarray, meta: dict = None)
writer.dump(data: dict, meta: dict = None)
```

### NpzWriter

```python
NpzWriter(
    root: str | Path,
    schema: MeasurementSchema = None,
    tags: list[str] = [],
    overwrite: bool = False,
    compressed: bool = False,
)

writer.write(data: dict[str, np.ndarray])
writer.write_metadata(meta: dict = None)
```

### ExperimentID

```python
ExperimentID(root: str | Path)

eid.current_id()          → int
eid.next_id()             → int
eid.next_dir()            → Path   # e.g. ./data/0004
eid.current_dir()         → Path
eid.get_dir_name(id)      → Path   # int or str
eid.get_folders()         → list[Path]
```

### StreamingWriter

```python
StreamingWriter(
    root: str | Path,
    readouts: dict[str, tuple],   # name -> trailing shape, e.g. {"x": (), "S21": (1601,)}
    chunk_size: int = 256,
    dtype: str = "float32",
    tags: list[str] = [],
    overwrite: bool = False,
)

writer.append(data: dict, timestamp: float = None)
writer.flush()
writer.close(meta: dict = None)
```

Arrays in vault:

| Name | Shape | dtype | Description |
|---|---|---|---|
| `_time` | `(N,)` | float64 | Unix timestamps |
| `<readout>` | `(N,) + trailing` | dtype | One entry per `readouts` key |

### Utility functions

```python
encode_value(x)    # numpy array/scalar → JSON-serializable Python type
serialize(x)       # full JSON encode/decode cycle using encode_value
compute_sharding(shape: tuple, num_shards: int) → tuple   # compute shard shape
```

---

### Reader

```python
from zarro import open_vault, Vault, VaultError
from zarro.reader import Axis, Readout
```

#### `open_vault(path)`

Convenience wrapper — returns `Vault(path)`.

#### `Vault(path, *, control=None, readouts=None)`

Open a run directory for reading.

| Attribute | Type | Description |
|---|---|---|
| `kind` | `"sweep"` or `"monitor"` | Determined by presence of `_time` array |
| `shape` | `tuple[int, ...]` | Grid shape, or `(n_samples,)` for monitors |
| `metadata` | `dict` | zarr attrs merged with `metadata.yaml` |
| `tags` | `list[str]` | |
| `axes` | `list[Axis]` | Outermost axis first; empty for monitors |
| `coords` | `dict[str, np.ndarray]` | Coordinate values per control variable |
| `coord_axis` | `dict[str, int]` | Axis index for each control variable |
| `readouts` | `dict[str, Readout]` | One entry per data array |
| `time` | `np.ndarray` | Unix timestamps (monitors only) |

Methods:

```python
vault.read(name, selection=Ellipsis) → np.ndarray  # 0.0 at unwritten points
vault.values(name) → np.ndarray                    # NaN at unwritten points
vault.written_mask(name, order=None) → np.ndarray  # bool grid
```

#### `Axis`

```python
@dataclass(frozen=True)
class Axis:
    index: int
    controls: tuple[str, ...]
    length: int
    units: dict[str, str | None]   # {name: unit_string_or_None}
```

#### `Readout`

```python
@dataclass(frozen=True)
class Readout:
    name: str
    kind: str                           # "scalar" | "trace" | "image"
    shape: tuple[int, ...]              # trailing shape; () for scalar
    unit: str | list[str] | None
    contains: str | list[str] | None
```

#### `VaultError`

Raised when the directory is not a readable zarro run, a required array is
missing, or metadata cannot be parsed without fallback arguments.

---

## 15. Version Management

Versioning is handled automatically by [setuptools-scm](https://setuptools-scm.readthedocs.io/). The version is derived from git tags — there is no version string to maintain manually in the source code.

### Checking the current version

```python
import zarro
print(zarro.__version__)
```

### Version string format

| Situation | Example |
|---|---|
| Exactly on a tag | `0.2.0` |
| Commits ahead of a tag | `0.2.1.dev4+gabcd1234` |
| Uncommitted changes present | `0.2.1.dev4+gabcd1234.d20261112` |

### Releasing a new version

```bash
# 1. Finish and commit all changes
git add -A && git commit -m "release v0.3.0"

# 2. Create the tag
git tag v0.3.0

# 3. Push branch and tag
git push --follow-tags

# 4. Reinstall locally so _version.py is regenerated
pip install -e .

# 5. Verify
python -c "import zarro; print(zarro.__version__)"  # → 0.3.0
```

> **Note:** step 4 (reinstall) is only needed locally after tagging. Regular code edits are picked up immediately by the editable install without any reinstall.

### Why --follow-tags?

`git push` alone sends branch commits but never tags. `--follow-tags` pushes the branch and any tags reachable from it in one command. You can make this the default permanently:

```bash
git config --global push.followTags true
```

After that, plain `git push` does everything.
