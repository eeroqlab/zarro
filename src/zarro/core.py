import yaml
import warnings
import numpy as np
import zarr
import json

from datetime import datetime
from enum import StrEnum
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


def encode_value(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):  # numpy scalars (np.float32, np.int64, etc.)
        return x.item()
    return x


def serialize(x):
    return json.loads(json.dumps(x, default=encode_value))


class DataKind(StrEnum):
    SCALAR = "scalar"
    TRACE = "trace"
    IMAGE = "image"

class WriteType(StrEnum):
    POINTWISE = "pointwise"
    TRACEWISE = "tracewise"
    IMAGEWISE = "imagewise"
    ALL       = "all"


@dataclass(frozen=True)
class ControlVar:
    name: str
    values: np.ndarray
    unit: Optional[str] = None
    reverse: Optional[bool] = False  # Whether the control variable is swept backwards (e.g. for hysteresis measurements); this is just metadata and does not affect the order of values in the array
    instrument: Optional[str] = None  # Optional field to specify the instrument/source of the control variable
    
    @property
    def length(self):
        return len(self.values)
    
    def __post_init__(self):
        if self.reverse:
            extended_values = np.zeros(2 * self.length)
            extended_values[:self.length] = self.values
            extended_values[self.length:] = self.values[::-1]
            self.values = extended_values


@dataclass(slots=True)
class AxisSpecs:
    """
    Defines a control axis for the measurement schema, consisting of one or more control variables that share the same length (number of points).

    Args:
        controls: list of ControlVar instances that define the control variables for this axis.
            All ControlVar instances in the list must have the same length (number of points) to ensure consistent indexing along this axis.
    """
    controls: list[ControlVar]
    length: int = field(init=False)

    def __post_init__(self):
        # FIX: replaced assert with ValueError so validation isn't stripped by python -O
        if len(self.controls) == 0:
            raise ValueError("At least one control variable is required")
        length_list = [len(c.values) for c in self.controls]
        if len(set(length_list)) > 1:
            raise ValueError(f"All ControlVar instances must have the same length, got: {length_list}")
        self.length = length_list[0] if length_list else 0

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.controls]


@dataclass(slots=True)
class ReadoutSpecs:
    name: str
    kind: DataKind
    unit: Optional[str | list[str]] = None
    shape: Optional[tuple[int, ...]] = None  # For trace/array data, specify the shape of the trailing dimensions (e.g. trace length or image height/width)
    contains: Optional[str | list[str]] = None


@dataclass(slots=True)
class MeasurementSchema:
    """Defines the structure of the measurement, including control axes and readout specifications.

    Args:
        control: list of AxisSpec instances that define the control axes for the measurement.
            Each AxisSpec instance must have a consistent length across its ControlVar instances.
        measure: list of ReadoutSpec instances that define the measurement readouts, including their names,
            data kinds (scalar, trace, image), and optional units and shapes.
        writetype: specifies the expected write pattern for the data, which can be used to optimize chunking and sharding strategies.
            For example, if writetype is POINTWISE, the writer can optimize for individual point writes.
    """
    control: list[AxisSpecs]
    readout: list[ReadoutSpecs]
    writetype: WriteType = WriteType.POINTWISE
    meta: dict = field(init=False)

    @property
    def axis_names(self) -> dict:  # FIX: was annotated as list[str] but returns a dict
        names = {}
        for i, axis in enumerate(self.control):
            names["axis" + str(i)] = axis.names
        return names

    @property
    def readout_names(self) -> list[str]:
        return [r.name for r in self.readout]

    def __post_init__(self):
        readout_meta = {}
        for r in self.readout:
            specs = asdict(r)
            specs.pop("name")
            readout_meta[r.name] = specs

        self.meta = {
            "control": self.axis_names,
            "readout": readout_meta
        }


@dataclass
class ExperimentID:
    root: Path | str
    
    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
    
    def get_folders(self):
        # Use a list comprehension to filter directories
        return [x for x in self.root.iterdir() if x.is_dir()]

    def current_id(self):
        folders = self.get_folders()
        counter = len(folders)
        if counter > 0:
            counter = max(counter, int(folders[-1].name[:4]))
        return counter
    
    def next_id(self):
        return self.current_id() + 1   

    def next_dir(self):
        name = str(self.next_id()).rjust(4, "0")
        return self.root / Path(name)
    
    def current_dir(self):
        name = str(self.current_id()).rjust(4, "0")
        return self.root / Path(name)
    
    def get_dir_name(self, id: int | str):
        if isinstance(id, str):
            return self.root / Path(id)
        else:
            return self.root / Path(str(id).rjust(4, "0"))


def compute_sharding(shape: tuple[int, ...], num_shards: int) -> tuple[int, ...]:
    """Compute shard shape given total array shape and desired number of shards.

    Args:
        shape: full shape of the array to be sharded
        num_shards: desired number of shards to split the array into

    Returns:
        shard_shape: shape of each shard, computed to approximately meet the target shard size
    """
    shard_shape = []
    escape_S = 1
    for Ni in shape:
        if (escape_S * Ni/(num_shards) > 1):
            Si =  escape_S * Ni // (num_shards)
            shard_shape.append(int(Si))
            break
        else:
            Si = 1
            shard_shape.append(Si)
            escape_S = escape_S * Ni

    if len(shard_shape) < len(shape):
        shard_shape.extend(shape[len(shard_shape):])

    return tuple(shard_shape)


@dataclass
class ZarrWriter:
    """
    Generalized streaming writer for multi-dimensional parameter sweeps with mixed scalar/trace readouts, using Zarr v3 sharding.

    Args:
        root: directory to store the Zarr and metadata files. The Zarr group will be created at root/vault.zarr.
        schema (Optional[MeasurementSchema]): defines the control axes and readout specifications.
            The writer uses this schema to initialize the Zarr arrays and
            optimize chunking/sharding strategies based on the writetype.
        tags (list[str]): optional list of tags to categorize or annotate the measurement.
        dtype (str): numpy dtype string for data arrays (default "float32"). Used for array creation and shard size estimation.
        target_file_size_bytes (int): desired shard file size in bytes (default 1MB); used to compute shard shapes for trace/array data
        overwrite (bool): if True, existing data will be overwritten; otherwise, it will be preserved and verified for consistency
        initialize_arrays (bool): if True, Zarr arrays will be created at initialization based on the schema.
            If False, arrays can be created later using add_array(). Default is false.

    Example usage:
    writer = ZarrWriter(
        root="path/to/data",
        schema=mschema,
        target_file_size_bytes=1 * 1024 * 1024,  # target ~1MB shards; adjust as needed
        overwrite=True,
    )
    """

    root: str | Path
    schema: Optional[MeasurementSchema] = None
    tags: list[str] = field(default_factory=list)

    # Behavior
    dtype: str = "float32"  # FIX: exposed dtype so itemsize can be used in sharding instead of hardcoded 4
    overwrite: bool = False
    initialize_arrays: bool = False  # Whether to create Zarr arrays at initialization based on the schema. If False, arrays can be created later using add_array().
    target_file_size_bytes: int = 5 * 1024 * 1024  # target ~5MB shards; adjust as needed

    # Not user controlled
    _control_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        # Local filesystem store (works on NAS paths)
        self.vault = zarr.group(str(self.root / "vault.zarr"))

        if self.schema is not None:
            self._control_shape = tuple(ax.length for ax in self.schema.control)

        if self.initialize_arrays:
            # Create/open arrays
            self._init_coord_arrays()
            self._init_data_arrays()
            # Write a small manifest (you can expand with full experiment metadata)
            self.write_yaml()


    def _init_coord_arrays(self):
        """Initialize coordinate arrays for control variables."""
        for ax in self.schema.control:
            for var in ax.controls:
                arr = self.vault.create_array(
                    name=var.name,
                    shape=var.values.shape,
                    dtype=self.dtype,  # FIX: use self.dtype instead of hardcoded "float32"
                    overwrite=self.overwrite,
                )
                arr[:] = np.asarray(var.values)


    def _init_data_arrays(self):
        """Initialize data arrays for measurement readouts."""

        for readout in self.schema.readout:

            match readout.kind:
                case DataKind.SCALAR:
                    shape = self._control_shape
                    shard_shape = None
                    chunk_shape = self._get_chunk_shape()

                case DataKind.TRACE:
                    shape = self._control_shape + (readout.shape[0],)
                    if self.schema.writetype == WriteType.ALL:
                        shard_shape = None
                    else:
                        shard_shape = self._get_shard_shape(shape, data_kind=DataKind.TRACE)
                    chunk_shape = self._get_chunk_shape() + (readout.shape[0],)
                    
                # FIX: IMAGE was previously unhandled (raised NotImplementedError)
                case DataKind.IMAGE:
                    shape = self._control_shape + readout.shape  # readout.shape is (H, W)
                    if self.schema.writetype == WriteType.ALL:
                        shard_shape = None
                    else:
                        shard_shape = self._get_shard_shape(shape, data_kind=DataKind.IMAGE)
                    chunk_shape = self._get_chunk_shape() + readout.shape

                case _:
                    raise NotImplementedError(f"Unsupported measurement kind: {readout.kind}")

            arr = self.vault.create_array(
                name=readout.name,
                shape=shape,
                shards=shard_shape,
                chunks=chunk_shape,
                dtype=self.dtype,  # FIX: use self.dtype instead of hardcoded "float32"
                overwrite=self.overwrite,
            )


    def _get_chunk_shape(self) -> tuple[int, ...]:
        """
        Determine the chunk shape for writing based on the writetype and control shape.
        """
        shape = self._control_shape

        match self.schema.writetype:
            case WriteType.POINTWISE:
                return (1,) * len(shape)

            case WriteType.TRACEWISE:
                return (1,) * (len(shape) - 1) + (shape[-1],)

            case WriteType.IMAGEWISE:
                # FIX: replaced assert with ValueError so validation isn't stripped by python -O
                if len(shape) <= 1:
                    raise ValueError("Control shape must have at least two dimensions for imagewise writes")
                return (1,) * (len(shape) - 2) + (shape[-1], shape[-2])

            case WriteType.ALL:
                return shape


    def _get_shard_shape(self, shape: tuple[int, ...], data_kind: DataKind) -> tuple[int, ...]:
        """
        Compute the shard shape for a given array shape and target shard size.
        """
        # Compute the total number of elements in the array
        total_elements = np.prod(shape)

        # FIX: use actual dtype itemsize instead of hardcoded 4 (which assumed float32)
        itemsize = np.dtype(self.dtype).itemsize
        num_shards = max(1, total_elements * itemsize // self.target_file_size_bytes)

        match self.schema.writetype:
            case WriteType.POINTWISE:
                index = slice(None)
                outro = slice(0)
            case WriteType.TRACEWISE:
                index = slice(None, -1)
                outro = slice(-1, None)
            case WriteType.IMAGEWISE:
                index = slice(None, -2)
                outro = slice(-2, None)
            # FIX: WriteType.ALL was missing, causing UnboundLocalError if reached
            case WriteType.ALL:
                return shape  # no sharding needed for full writes

        match data_kind:
            case DataKind.SCALAR:
                true_shape = shape
                shard_shape = compute_sharding(true_shape[index], num_shards)
                shard_shape += true_shape[outro]

            case DataKind.TRACE:
                true_shape = shape[:-1]  # control dims only
                shard_shape = compute_sharding(true_shape[index], num_shards)
                shard_shape += true_shape[outro] + (shape[-1],)

            # FIX: IMAGE was missing from this match (needed for _init_data_arrays IMAGE case)
            case DataKind.IMAGE:
                true_shape = shape[:-2]  # control dims only
                shard_shape = compute_sharding(true_shape[index], num_shards)
                shard_shape += true_shape[outro] + shape[-2:]

        return shard_shape


    def _write(self, index: tuple[int, ...], data: dict[str, np.ndarray], writetype: WriteType):
        """Generalized write method that dispatches based on writetype."""

        match writetype:
            case WriteType.POINTWISE:
                num_slices = 0
            case WriteType.TRACEWISE:
                num_slices = 1
            case WriteType.IMAGEWISE:
                num_slices = 2
            case _:
                raise ValueError(f"Unsupported writetype: {writetype}")

        # FIX: error message said "- 1" regardless of writetype; now uses actual num_slices
        if len(index) != len(self._control_shape) - num_slices:
            raise ValueError(f"index length {len(index)} does not match control dims {len(self._control_shape) - num_slices}")

        datakind_by_name = {r.name: r.kind for r in self.schema.readout}

        # Write each dataset
        for name, value in data.items():
            if name not in self.vault:
                raise KeyError(f"Unknown dataset '{name}'")

            arr = self.vault[name]
            value = np.asarray(value, dtype=arr.dtype)

            match datakind_by_name[name]:
                case DataKind.SCALAR:
                    self._check_input_shape(value, arr.shape[len(index):], name)  # FIX: re-enabled shape validation
                    index_tuple = index + (slice(None),) * (num_slices + 0)

                case DataKind.TRACE:
                    self._check_input_shape(value, arr.shape[len(index):], name)  # FIX: re-enabled shape validation
                    index_tuple = index + (slice(None),) * (num_slices + 1)

                case DataKind.IMAGE:
                    self._check_input_shape(value, arr.shape[len(index):], name)  # FIX: re-enabled shape validation
                    index_tuple = index + (slice(None),) * (num_slices + 2)

            arr[index_tuple] = value


    def write_point(self, index: tuple[int, ...], data: dict[str, np.ndarray]):
       self._write(index, data, writetype=WriteType.POINTWISE)


    def write_trace(self, index: tuple[int, ...], data: dict[str, np.ndarray]):
        self._write(index, data, writetype=WriteType.TRACEWISE)


    def write_image(self, index: tuple[int, ...], data: dict[str, np.ndarray]):
        self._write(index, data, writetype=WriteType.IMAGEWISE)


    def write_all(self, data: dict[str, np.ndarray]):
        # Write each dataset
        for name, value in data.items():
            if name not in self.vault:
                raise KeyError(f"Unknown dataset '{name}'")

            arr = self.vault[name]
            value = np.asarray(value, dtype=arr.dtype)
            self._check_input_shape(value, arr.shape, name)

            index_tuple = (slice(None),) * len(arr.shape)
            arr[index_tuple] = value


    def _check_input_shape(self, value: np.ndarray, expected_shape: tuple[int, ...], name: str):
        if value.shape != expected_shape:
            raise ValueError(f"Dataset '{name}' expects shape {expected_shape} but got {value.shape}")


    def write_yaml(self, meta: dict[str, str] = None):
        """Write a simple YAML manifest with experiment metadata, including control axis names and readout names.
        This is separate from the Zarr attributes and can be used for quick inspection or compatibility with other tools.

        Args:
            meta: additional metadata to include in the YAML file, as a dictionary of key-value pairs.
            This can be used to add custom metadata fields beyond the standard control and readout information.
        """
        metadata_path = self.root / "metadata.yaml"
        # FIX: was silently skipping without any indication; now emits a warning
        if metadata_path.exists() and not self.overwrite:
            warnings.warn(f"Metadata file already exists at {metadata_path} and overwrite=False; skipping write.")
            return

        metadata = {"tags": self.tags}
        if self.schema is not None:
            metadata |= self.schema.meta
        if meta:
            metadata |= meta  # Merge additional metadata if provided

        metadata_path.write_text(yaml.safe_dump(serialize(metadata), sort_keys=False))


    def write_metadata(self, meta: dict = None):
        """
        Write experiment metadata as Zarr group attributes. This can include control axis names, readout names,
        and any additional metadata provided by the user.
        Zarr attributes are stored as JSON-serializable key-value pairs and can be accessed programmatically when reading the Zarr group,
        making them useful for storing structured metadata alongside the data arrays.

        Args:
            meta: additional metadata to include in the Zarr attributes, as a dictionary of key-value pairs.
                This can be used to add custom metadata fields beyond the standard control and readout information.
        """
        metadata = {
            "date": datetime.now().isoformat(),  # FIX: was str(self.root), which stored the path not the date
            "schema": self.schema.meta if self.schema is not None else "none",
            "tags": self.tags
        }
        if meta:
            metadata |= meta  # Merge additional metadata if provided
        metadata = serialize(metadata)
        self.vault.attrs.put(metadata)  # Store metadata as a single JSON string in Zarr attributes; can be parsed back into structured form when reading
        self.write_yaml(meta)  # Also write a human-readable YAML manifest


    def add_array(self, name: str, value: np.ndarray | list, meta: dict[str] = None):
        """Add a new array to the vault after initialization. Useful for adding derived datasets."""
        if name in self.vault and not self.overwrite:
            raise KeyError(f"Dataset '{name}' already exists")

        value = np.asarray(value)

        arr = self.vault.create_array(
            name=name,
            shape=value.shape,
            dtype=value.dtype,
            overwrite=self.overwrite,
            attributes=meta
        )
        arr[:] = value


    def dump(self, data: dict, meta: dict = None):
        self.vault.attrs.put(meta if meta else {})  # Store per-array metadata in a separate attribute; can be accessed when reading to understand the structure of the data
        for key, value in data.items():
            self.add_array(name=key, value=value)


@dataclass
class NpzWriter:

    """Simple writer that saves data as .npz files, organized by control axis values. Not optimized for large datasets or complex schemas."""
    root: str | Path
    schema: Optional[MeasurementSchema] = None
    tags: list[str] = field(default_factory=list)

    #Behavior
    overwrite: bool = False
    compressed: bool = False

    # Not user controlled
    _control_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        if self.schema is not None:
            self._control_shape = tuple(ax.length for ax in self.schema.control)


    def write(self, data: dict[str, np.ndarray]):
        filename = self.root / "data.npz"  # FIX: removed unnecessary f-string prefix
        if self.overwrite and filename.exists():
            filename.unlink()  # Remove existing file if overwrite is True
        if self.compressed:
            np.savez_compressed(filename, **data)
        else:
            np.savez(filename, **data)


    def write_metadata(self, meta: dict = None):
        metadata_path = self.root / "metadata.yaml"

        metadata = {"tags": self.tags}
        if self.schema is not None:
            metadata |= self.schema.meta
        if meta:
            metadata |= meta  # Merge additional metadata if provided

        metadata_path.write_text(yaml.safe_dump(serialize(metadata), sort_keys=False))


@dataclass
class StreamingWriter:
    """
    Append-only writer for open-ended time-series data where the total number
    of samples is not known in advance.

    Arrays start at size 0 and grow along the time axis with each call to
    append(). Samples are buffered in memory and flushed to disk in batches
    of chunk_size to minimise the number of resize operations.

    A '_time' array is always written alongside the data, storing Unix
    timestamps (float64) for each sample.

    Args:
        root:        directory for vault.zarr and metadata.yaml
        readouts:    dict mapping readout name to its trailing shape.
                     Use an empty tuple () for scalars, (N,) for traces,
                     (H, W) for images.
                     Example: {"lockin_X": (), "S21": (1601,), "frame": (480, 640)}
        chunk_size:  number of samples per chunk along the time axis.
                     Also the buffer size — flush() is called automatically
                     once this many samples have been buffered.
        dtype:       numpy dtype for all data arrays (default "float32").
        tags:        free-form tags written to metadata on close().
        overwrite:   if True, an existing vault at root will be overwritten.
    """

    root: str | Path
    readouts: dict[str, tuple]
    chunk_size: int = 256
    dtype: str = "float32"
    tags: list[str] = field(default_factory=list)
    overwrite: bool = False

    _count: int = field(init=False, default=0)
    _buf_time: list = field(init=False, default_factory=list)
    _buf_data: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vault = zarr.group(str(self.root / "vault.zarr"))
        self._buf_data = {name: [] for name in self.readouts}

        # Timestamp coordinate array — always present
        self.vault.create_array(
            name="_time",
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype="float64",
            overwrite=self.overwrite,
        )

        for name, trailing in self.readouts.items():
            self.vault.create_array(
                name=name,
                shape=(0,) + trailing,
                chunks=(self.chunk_size,) + trailing,
                dtype=self.dtype,
                overwrite=self.overwrite,
            )

    def append(self, data: dict, timestamp: float = None):
        """Buffer one sample. Flushes to disk automatically when chunk_size is reached.

        Args:
            data:      dict mapping readout name to value (scalar, 1-D, or 2-D array).
            timestamp: Unix timestamp for this sample. Defaults to datetime.now().
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        self._buf_time.append(timestamp)
        for name, value in data.items():
            self._buf_data[name].append(np.asarray(value, dtype=self.dtype))
        if len(self._buf_time) >= self.chunk_size:
            self.flush()

    def flush(self):
        """Write all buffered samples to disk in a single resize + write operation."""
        n_new = len(self._buf_time)
        if n_new == 0:
            return
        n_old = self._count
        n_total = n_old + n_new

        self.vault["_time"].resize((n_total,))
        self.vault["_time"][n_old:n_total] = np.array(self._buf_time)

        for name, samples in self._buf_data.items():
            if not samples:
                continue
            arr = self.vault[name]
            arr.resize((n_total,) + arr.shape[1:])
            arr[n_old:n_total] = np.stack(samples)

        self._count = n_total
        self._buf_time = []
        self._buf_data = {name: [] for name in self.readouts}

    def close(self, meta: dict = None):
        """Flush remaining buffered samples and write metadata.

        Writes Zarr group attributes (JSON) and a metadata.yaml file,
        consistent with the convention used by ZarrWriter.

        Args:
            meta: additional key-value pairs to include in the metadata.
        """
        self.flush()
        metadata = {
            "date":     datetime.now().isoformat(),
            "count":    self._count,
            "tags":     self.tags,
            "readouts": {k: list(v) for k, v in self.readouts.items()},
        }
        if meta:
            metadata |= meta
        metadata = serialize(metadata)
        self.vault.attrs.put(metadata)
        (self.root / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
