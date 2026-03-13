import yaml
import numpy as np
import zarr
import json

from enum import StrEnum
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


def encode_value(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    # other special cases... 
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
    instrument: Optional[str] = None  # Optional field to specify the instrument/source of the control variable


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
        assert len(self.controls) > 0, "At least one control variable is required"
        length_list = [len(c.values) for c in self.controls]
        assert len(set(length_list)) <= 1  # Returns True if all elements in items are equal, False otherwise.
        self.length = length_list[0] if length_list else 0
    
    @property
    def names(self) -> list[str]:
        return [c.name for c in self.controls]


@dataclass(slots=True)
class ReadoutSpecs:
    name: str
    kind: DataKind
    unit: Optional[str] = None
    shape: Optional[tuple[int, ...]] = None  # For trace/array data, specify the shape of the trailing dimensions (e.g. trace length or image height/width)


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
    measure: list[ReadoutSpecs]
    writetype: WriteType = WriteType.POINTWISE

    @property
    def axis_names(self) -> list[str]:
        axis_names = {}
        for i, axis in enumerate(self.control[::-1]):
            axis_names["axis" + str(i) + " / " + str(axis.length)] = axis.names
        return axis_names
    
    @property
    def readout_names(self) -> list[str]:
        return [r.name for r in self.measure]


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
        root: directory to store the Zarr and metadata
        experiment_id: unique ID for this experiment/run
        schema: defines the control axes and readout specifications. Must be provided by the user to define the structure of the data.
        target_file_size_bytes: desired shard file size in bytes (default 1MB); used to compute shard shapes for trace/array data
        overwrite: if True, existing data will be overwritten; otherwise, it will be preserved and verified for consistency

    Example usage:
    writer = ZarrWriter(
        root="path/to/data",
        experiment_id="exp_001",
        schema=mschema,
        target_file_size_bytes=1 * 1024 * 1024,  # target ~1MB shards; adjust as needed
        overwrite=True,
    )
    """

    root: str | Path
    experiment_id: str
    schema: MeasurementSchema
    tags: list[str] = field(default_factory=list)

    # Behavior
    overwrite: bool = False
    initialize_arrays: bool = True  # Whether to create Zarr arrays at initialization based on the schema. If False, arrays can be created later using add_array().
    target_file_size_bytes: int = 5 * 1024 * 1024  # target ~5MB shards; adjust as needed

    # Not user controlled
    _control_shape: tuple[int, ...] = field(init=False)
    _dataset_map: dict[str, zarr.Array] = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root) / Path(self.experiment_id)
        self.root.mkdir(parents=True, exist_ok=True)

        # Local filesystem store (works on NAS paths)
        self.vault = zarr.group(str(self.root / "vault.zarr"))

        self._control_shape = tuple(ax.length for ax in self.schema.control)
        self._dataset_map = {}

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
                    dtype="float32",
                    overwrite=self.overwrite,
                )
                arr[:] = np.asarray(var.values)
    

    def _init_data_arrays(self):
        """Initialize data arrays for measurement readouts."""

        for readout in self.schema.measure:
            
            match readout.kind:
                case DataKind.SCALAR:
                    shape = self._control_shape
                    shard_shape = None # self._get_shard_shape(shape, data_kind=DataKind.SCALAR)
                    chunk_shape = self._get_chunk_shape()

                case DataKind.TRACE:
                    shape = self._control_shape + (readout.shape[0],)
                    if self.schema.writetype == WriteType.ALL:
                        shard_shape = None
                    else:
                        shard_shape = self._get_shard_shape(shape, data_kind=DataKind.TRACE)
                    chunk_shape = self._get_chunk_shape() + (readout.shape[0],)
                
                case _:
                    raise NotImplementedError(f"Unsupported measurement kind: {readout.kind}")

            arr = self.vault.create_array(
                name=readout.name,
                shape=shape,
                shards=shard_shape,
                chunks=chunk_shape,
                dtype="float32",
                overwrite=self.overwrite,
            )
            self._dataset_map[readout.name] = arr


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
                assert len(shape) > 1, "Control shape must have at least two dimensions for imagewise writes"
                return (1,) * (len(shape) - 2) + (shape[-1], shape[-2])

            case WriteType.ALL:
                return shape


    def _get_shard_shape(self, shape: tuple[int, ...], data_kind: DataKind) -> tuple[int, ...]:
        """
        Compute the shard shape for a given array shape and target shard size.
        """
        # Compute the total number of elements in the array
        total_elements = np.prod(shape)

        # Compute the number of shards needed to meet the target shard size
        num_shards = max(1, total_elements * 4 // self.target_file_size_bytes)

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

        match data_kind:
            case DataKind.SCALAR:
                true_shape = shape
                shard_shape = compute_sharding(true_shape[index], num_shards)
                shard_shape += true_shape[outro]
                
            case DataKind.TRACE:
                true_shape = shape[:-1]  # control dims only
                shard_shape = compute_sharding(true_shape[index], num_shards)
                shard_shape += true_shape[outro] + (shape[-1],)

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

        if len(index) != len(self._control_shape) - num_slices:
            raise ValueError(f"index length {len(index)} does not match control dims {len(self._control_shape) - 1}")
    
        datakind_by_name = {r.name: r.kind for r in self.schema.measure}

        # Write each dataset
        for name, value in data.items():
            if name not in self._dataset_map:
                raise KeyError(f"Unknown dataset '{name}'")
            
            arr = self._dataset_map[name]
            value = np.asarray(value, dtype=arr.dtype)

            match datakind_by_name[name]:
                case DataKind.SCALAR:
                    # self._check_input_shape(value, (), name)
                    index_tuple = index + (slice(None),) * (num_slices + 0)

                case DataKind.TRACE:
                    # self._check_input_shape(value, (arr.shape[-1],), name)
                    index_tuple = index + (slice(None),) * (num_slices + 1)

                case DataKind.IMAGE:
                    # self._check_input_shape(value, arr.shape[-2:], name)
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
            if name not in self._dataset_map:
                raise KeyError(f"Unknown dataset '{name}'")
            
            arr = self._dataset_map[name]
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
        if metadata_path.exists() and not self.overwrite:
            return
        
        metadata = {
            "control": self.schema.axis_names,
            "readout": self.schema.readout_names,
            "tags": self.tags
        }
        if meta:
            metadata |= meta  # Merge additional metadata if provided
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))


    def write_metadata(self, meta: dict[str, str] = None):
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
            "date": "2026-02-24",
            "experiment_id": self.experiment_id,
            "schema": asdict(self.schema),
            "tags": self.tags
        }
        if meta:
            metadata |= meta  # Merge additional metadata if provided
        metadata = serialize(metadata)
        self.vault.attrs.put(metadata)  # Store metadata as a single JSON string in Zarr attributes; can be parsed back into structured form when reading
        self.write_yaml(meta)  # Also write a human-readable YAML manifest


    def add_array(self, name: str, value: np.ndarray | list, meta: dict[str] = None):
        """Add a new array to the vault after initialization. Useful for adding derived datasets."""
        if name in self._dataset_map and not self.overwrite:
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
        self._dataset_map[name] = arr


    def write_npzlike(self, data: dict[str, np.ndarray], meta: dict[str, dict] = None):
        for key, value in data.items():
            self.add_array(name=key, value=value, meta=meta)


@dataclass
class NpzWriter:

    """Simple writer that saves data as .npz files, organized by control axis values. Not optimized for large datasets or complex schemas."""
    root: str | Path
    schema: MeasurementSchema
    tags: list[str] = field(default_factory=list)

    #Behavior
    overwrite: bool = True
    compressed: bool = True

    # Not user controlled
    _control_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        self._control_shape = tuple(ax.length for ax in self.schema.control)


    def write(self, data: dict[str, np.ndarray]):
        filename = self.root / f"data.npz"
        if self.overwrite and filename.exists():
            filename.unlink()  # Remove existing file if overwrite is True
        if self.compressed:
            np.savez_compressed(filename, **data)
        else:
            np.savez(filename, **data)


    def write_metadata(self, add_meta: dict[str, str] = None):
        metadata_path = self.root / "metadata.yaml"
        metadata = {
            "axis": self.schema.axis_names,
            "readout": self.schema.readout_names,
            "schema": self._control_shape
        }
        if add_meta:
            metadata |= add_meta  # Merge additional metadata if provided
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))


    def write_summary(self):
        summary_path = self.root / "summary.txt"

        with open(summary_path, "w") as f:
            f.write("Experiment Summary\n")
            f.write("------------------\n")

            f.write(f'\n{len(self.data)} measurements made.')

            f.write('\n')

            f.write(f'\nTags:')
            for tag in self.tags:
                f.write(f'   \n{tag}')

            f.write('\n')

            f.write('\nControl Axes:')
            for iax, ax in enumerate(self.schema.control):
                f.write(f'\n   Axis {iax}:')
                for ctrl in ax.controls:
                    f.write(f'\n      {ctrl.name} swept between {ctrl.values[0]:.3e} and {ctrl.values[-1]:.3e}')

            f.write('\n')


            f.write('\nMeasured:')
            for ro in self.schema.readout:
                f.write(f'\n   {ro.name} [{ro.unit}] of shape {ro.shape} stored')

            f.write('\n')