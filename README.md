# zarro

A Python package for structured experimental data saving, built on [Zarr v3](https://zarr.readthedocs.io/) with automatic sharding. Designed for multi-dimensional parameter sweeps with mixed readout types (scalars, traces, images).

## Features

- Declarative schema for control axes and measurement readouts
- Flexible write patterns: point-by-point, trace-by-trace, image-by-image, or all-at-once
- Automatic chunk and shard sizing for efficient I/O
- Human-readable YAML metadata alongside machine-readable Zarr attributes
- Hysteresis support via reversed control variable sweeps
- Experiment ID management for organized data directories

## Requirements

- Python >= 3.12
- numpy >= 2.4.2
- PyYAML >= 6.0.3
- zarr >= 3.1.5

## Installing

Clone the repository, then install dependencies and the package:

```shell
git clone <repository-url>
cd zarro
pip install -r requirements.txt
pip install -e .
```

Verify the installation:

```shell
python -c "from zarro import ZarrWriter; print('OK')"
```

## Quick start

```python
import numpy as np
from zarro import (
    DataKind, WriteType,
    ControlVar, AxisSpecs, ReadoutSpecs,
    MeasurementSchema, ZarrWriter,
)

# Define sweep axes
freq  = ControlVar("freq",  np.linspace(4e9, 8e9, 1601), unit="Hz")
power = ControlVar("power", np.linspace(-30, 0, 20),     unit="dBm")

# Define what you measure
schema = MeasurementSchema(
    control=[AxisSpecs([power]), AxisSpecs([freq])],
    readout=[ReadoutSpecs("S21", DataKind.SCALAR, unit="dBm")],
    writetype=WriteType.POINTWISE,
)

writer = ZarrWriter(root="./data/run_001", schema=schema,
                    overwrite=True, initialize_arrays=True)

for i, p in enumerate(power.values):
    for j, f in enumerate(freq.values):
        writer.write_point((i, j), {"S21": measure(p, f)})

writer.write_metadata(meta={"operator": "alice", "sample": "chip_B2"})
```

Read the data back:

```python
import zarr

z = zarr.open("./data/run_001/vault.zarr", mode="r")
print(list(z.keys()))       # ['freq', 'power', 'S21']
print(dict(z.attrs))        # metadata dict
s21 = z["S21"][:]           # numpy array, shape (20, 1601)
```

## Documentation

See [docs/docs.md](docs/docs.md) for a comprehensive guide covering all write patterns, data kinds, metadata management, experiment ID tracking, and the NPZ writer.

## License

MIT
