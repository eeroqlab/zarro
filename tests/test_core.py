"""
Tests for zarro/core.py.
Covers all write patterns shown in tutorials/demo.ipynb plus error cases and
the fixes applied in core (error messages, shape validation, date field, etc.).

Run with:  pytest tests/
"""

import json
import warnings

import numpy as np
import pytest
import yaml
import zarr

from zarro.core import (
    AxisSpecs,
    ControlVar,
    DataKind,
    MeasurementSchema,
    NpzWriter,
    ExperimentID,
    ReadoutSpecs,
    StreamingWriter,
    WriteType,
    ZarrWriter,
    compute_sharding,
    encode_value,
    serialize,
)

# ---------------------------------------------------------------------------
# Small dimensions so tests run fast
# ---------------------------------------------------------------------------
N_OUTER = 4    # Vgt axis (outer sweep)
N_INNER = 6    # Vch / Vm axis (inner sweep)
N_TRACE = 10   # number of frequency points in a trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_2d_schema(writetype: WriteType = WriteType.POINTWISE) -> MeasurementSchema:
    Vgt = ControlVar("Vgt", np.linspace(0, 1, N_OUTER, dtype=np.float32))
    Vch = ControlVar("Vch", np.linspace(0, 1, N_INNER, dtype=np.float32))
    Vm  = ControlVar("Vm",  np.linspace(0, 1, N_INNER, dtype=np.float32))
    return MeasurementSchema(
        control=[AxisSpecs([Vgt]), AxisSpecs([Vch, Vm])],
        readout=[
            ReadoutSpecs("lockin_X", DataKind.SCALAR, unit="V"),
            ReadoutSpecs("S21",      DataKind.TRACE,  shape=(N_TRACE,), unit="dBm"),
            ReadoutSpecs("temperature", DataKind.SCALAR, unit="K"),
        ],
        writetype=writetype,
    )


def make_1d_schema(writetype: WriteType = WriteType.POINTWISE) -> MeasurementSchema:
    Vch = ControlVar("Vch", np.linspace(0, 1, N_INNER, dtype=np.float32))
    Vm  = ControlVar("Vm",  np.linspace(0, 1, N_INNER, dtype=np.float32))
    return MeasurementSchema(
        control=[AxisSpecs([Vch, Vm])],
        readout=[
            ReadoutSpecs("lockin_X", DataKind.SCALAR, unit="V"),
            ReadoutSpecs("S21",      DataKind.TRACE,  shape=(N_TRACE,), unit="dBm"),
        ],
        writetype=writetype,
    )


# ===========================================================================
# ControlVar
# ===========================================================================

class TestControlVar:
    def test_basic_creation(self):
        v = ControlVar("Vch", np.linspace(0, 1, 5))
        assert v.name == "Vch"
        assert len(v.values) == 5

    def test_optional_fields_default_none(self):
        v = ControlVar("Vgt", np.array([1.0]))
        assert v.unit is None
        assert v.instrument is None

    def test_optional_fields_set(self):
        v = ControlVar("Vgt", np.array([1.0]), unit="V", instrument="DAC")
        assert v.unit == "V"
        assert v.instrument == "DAC"

    def test_reverse_doubles_values_forward_then_backward(self):
        v = ControlVar("Vb", np.array([0.0, 1.0, 2.0]), reverse=True)
        np.testing.assert_array_equal(v.values, [0, 1, 2, 2, 1, 0])
        assert v.length == 6

    def test_reverse_axis_length_and_writer_roundtrip(self, tmp_path):
        v = ControlVar("Vb", np.linspace(0, 1, 4, dtype=np.float32), reverse=True)
        schema = MeasurementSchema(
            control=[AxisSpecs([v])],
            readout=[ReadoutSpecs("x", DataKind.SCALAR)],
        )
        assert schema.control[0].length == 8
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        z = zarr.open_group(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["Vb"][:], v.values)


# ===========================================================================
# AxisSpecs
# ===========================================================================

class TestAxisSpecs:
    def test_length_computed_from_controls(self):
        v = ControlVar("Vch", np.linspace(0, 1, 7))
        ax = AxisSpecs([v])
        assert ax.length == 7

    def test_names_property(self):
        v1 = ControlVar("Vch", np.linspace(0, 1, 5))
        v2 = ControlVar("Vm",  np.linspace(0, 1, 5))
        ax = AxisSpecs([v1, v2])
        assert ax.names == ["Vch", "Vm"]

    def test_mismatched_lengths_raises_value_error(self):
        v1 = ControlVar("Vch", np.linspace(0, 1, 5))
        v2 = ControlVar("Vm",  np.linspace(0, 1, 7))
        with pytest.raises(ValueError, match="same length"):
            AxisSpecs([v1, v2])

    def test_empty_controls_raises_value_error(self):
        with pytest.raises(ValueError, match="At least one"):
            AxisSpecs([])


# ===========================================================================
# MeasurementSchema
# ===========================================================================

class TestMeasurementSchema:
    def test_meta_has_control_and_readout_keys(self):
        schema = make_2d_schema()
        assert "control" in schema.meta
        assert "readout" in schema.meta

    def test_axis_names_is_dict(self):
        schema = make_2d_schema()
        names = schema.axis_names
        assert isinstance(names, dict)

    def test_axis_names_content(self):
        schema = make_2d_schema()
        names = schema.axis_names
        assert "axis0" in names and "axis1" in names
        assert "Vgt" in names["axis0"]
        assert "Vch" in names["axis1"] and "Vm" in names["axis1"]

    def test_readout_names(self):
        schema = make_2d_schema()
        assert schema.readout_names == ["lockin_X", "S21", "temperature"]

    def test_empty_control_list_allowed(self):
        # tutorial uses control=[] for npz-like / image writes
        schema = MeasurementSchema(
            control=[],
            readout=[ReadoutSpecs("S34", DataKind.IMAGE, shape=(3, N_TRACE))],
            writetype=WriteType.ALL,
        )
        assert schema.meta["control"] == {}


# ===========================================================================
# Utility functions
# ===========================================================================

class TestEncodeValue:
    def test_ndarray_becomes_list(self):
        result = encode_value(np.array([1.0, 2.0]))
        assert isinstance(result, list)
        assert result == [1.0, 2.0]

    def test_numpy_scalar_becomes_python_scalar(self):
        result = encode_value(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_int_scalar(self):
        result = encode_value(np.int64(7))
        assert isinstance(result, int)

    def test_plain_values_pass_through(self):
        assert encode_value(42) == 42
        assert encode_value("hello") == "hello"
        assert encode_value(None) is None


class TestSerialize:
    def test_output_is_json_serializable(self):
        data = {"arr": np.array([1.0, 2.0]), "val": np.float64(3.0)}
        result = serialize(data)
        json.dumps(result)  # must not raise

    def test_nested_numpy_array(self):
        data = {"nested": {"arr": np.array([1, 2, 3])}}
        assert serialize(data)["nested"]["arr"] == [1, 2, 3]

    def test_numpy_scalar_in_dict(self):
        data = {"x": np.float32(1.5)}
        result = serialize(data)
        assert isinstance(result["x"], float)


class TestComputeSharding:
    def test_single_shard_returns_full_shape(self):
        assert compute_sharding((10, 20), 1) == (10, 20)

    def test_output_length_matches_input(self):
        shape = (10, 20, 30)
        shard = compute_sharding(shape, 5)
        assert len(shard) == len(shape)

    def test_all_dims_positive(self):
        shard = compute_sharding((100, 200, N_TRACE), 10)
        assert all(s >= 1 for s in shard)

    def test_empty_shape(self):
        assert compute_sharding((), 1) == ()


# ===========================================================================
# ZarrWriter — 2D scan (mirrors tutorial "Examples of 2D scans")
# ===========================================================================

class TestZarrWriter2D:

    def test_write_all_data_roundtrip(self, tmp_path):
        """2D scan / save all at once — tutorial pattern write_all(dict)."""
        schema = make_2d_schema(WriteType.ALL)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)

        S21    = np.random.rand(N_OUTER, N_INNER, N_TRACE).astype(np.float32)
        lockin = np.random.rand(N_OUTER, N_INNER).astype(np.float32)
        temp   = np.random.rand(N_OUTER, N_INNER).astype(np.float32)

        writer.write_all({"S21": S21, "lockin_X": lockin, "temperature": temp})

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["S21"][:],      S21)
        np.testing.assert_array_almost_equal(z["lockin_X"][:], lockin)
        np.testing.assert_array_almost_equal(z["temperature"][:], temp)

    def test_write_trace_data_roundtrip(self, tmp_path):
        """2D scan / save trace at each step — tutorial pattern write_trace((i,), dict)."""
        schema = make_2d_schema(WriteType.TRACEWISE)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)

        expected_S21   = np.random.rand(N_OUTER, N_INNER, N_TRACE).astype(np.float32)
        expected_lockin = np.random.rand(N_OUTER, N_INNER).astype(np.float32)
        expected_temp  = np.random.rand(N_OUTER, N_INNER).astype(np.float32)

        for i in range(N_OUTER):
            writer.write_trace((i,), {
                "S21":         expected_S21[i],
                "lockin_X":    expected_lockin[i],
                "temperature": expected_temp[i],
            })

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["S21"][:],      expected_S21)
        np.testing.assert_array_almost_equal(z["lockin_X"][:], expected_lockin)
        np.testing.assert_array_almost_equal(z["temperature"][:], expected_temp)

    def test_write_point_data_roundtrip(self, tmp_path):
        """2D scan / save at each point — tutorial pattern write_point((i,j), dict)."""
        schema = make_2d_schema(WriteType.POINTWISE)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)

        expected_S21    = np.random.rand(N_OUTER, N_INNER, N_TRACE).astype(np.float32)
        expected_lockin = np.random.rand(N_OUTER, N_INNER).astype(np.float32)
        expected_temp   = np.random.rand(N_OUTER, N_INNER).astype(np.float32)

        for i in range(N_OUTER):
            for j in range(N_INNER):
                writer.write_point((i, j), {
                    "S21":         expected_S21[i, j],
                    "lockin_X":    float(expected_lockin[i, j]),
                    "temperature": float(expected_temp[i, j]),
                })

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["S21"][:],      expected_S21)
        np.testing.assert_array_almost_equal(z["lockin_X"][:], expected_lockin)

    def test_coord_arrays_written_on_init(self, tmp_path):
        """Control variable arrays (Vgt, Vch, Vm) should appear in the vault."""
        schema = make_2d_schema(WriteType.ALL)
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        for name in ("Vgt", "Vch", "Vm"):
            assert name in z, f"Expected coord array '{name}' in vault"

    def test_add_array_after_write_all(self, tmp_path):
        """Tutorial: writer.add_array('Freqs', freq, meta={...}) after write_all."""
        schema = make_2d_schema(WriteType.ALL)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        freq = np.linspace(4e9, 8e9, N_TRACE, dtype=np.float64)
        writer.write_all({
            "S21": np.zeros((N_OUTER, N_INNER, N_TRACE), dtype=np.float32),
            "lockin_X": np.zeros((N_OUTER, N_INNER), dtype=np.float32),
            "temperature": np.zeros((N_OUTER, N_INNER), dtype=np.float32),
        })
        writer.add_array("Freqs", freq, meta={"unit": "Hz"})

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert "Freqs" in z
        np.testing.assert_array_almost_equal(z["Freqs"][:], freq)

    # --- error cases ---

    def test_wrong_index_length_raises(self, tmp_path):
        """Fixed: error message must use actual num_slices, not hardcoded 1."""
        schema = make_2d_schema(WriteType.POINTWISE)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        with pytest.raises(ValueError, match="index length"):
            # 2D control needs index (i, j); passing only (i,) should fail
            writer.write_point((0,), {
                "S21": np.zeros(N_TRACE),
                "lockin_X": 0.0,
                "temperature": 0.0,
            })

    def test_unknown_dataset_raises_key_error(self, tmp_path):
        schema = make_2d_schema(WriteType.POINTWISE)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        with pytest.raises(KeyError, match="nonexistent"):
            writer.write_point((0, 0), {"nonexistent": np.zeros(N_TRACE)})

    def test_wrong_shape_raises(self, tmp_path):
        """Fixed: shape validation is re-enabled in _write()."""
        schema = make_2d_schema(WriteType.POINTWISE)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        with pytest.raises(ValueError, match="expects shape"):
            writer.write_point((0, 0), {
                "S21":         np.zeros(N_TRACE + 5),  # wrong trace length
                "lockin_X":    0.0,
                "temperature": 0.0,
            })

    def test_write_all_wrong_shape_raises(self, tmp_path):
        schema = make_2d_schema(WriteType.ALL)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        with pytest.raises(ValueError, match="expects shape"):
            writer.write_all({
                "S21":         np.zeros((N_OUTER, N_INNER, N_TRACE + 1)),  # wrong
                "lockin_X":    np.zeros((N_OUTER, N_INNER)),
                "temperature": np.zeros((N_OUTER, N_INNER)),
            })


# ===========================================================================
# ZarrWriter — 1D scan (mirrors tutorial "Examples of 1D scans")
# ===========================================================================

class TestZarrWriter1D:

    def test_write_all_data_roundtrip(self, tmp_path):
        """1D scan / save all at once — tutorial pattern write_all(dict)."""
        schema = make_1d_schema(WriteType.ALL)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)

        S21    = np.random.rand(N_INNER, N_TRACE).astype(np.float32)
        lockin = np.random.rand(N_INNER).astype(np.float32)

        writer.write_all({"S21": S21, "lockin_X": lockin})

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["S21"][:],      S21)
        np.testing.assert_array_almost_equal(z["lockin_X"][:], lockin)

    def test_write_point_data_roundtrip(self, tmp_path):
        """1D scan / save at each point — tutorial pattern write_point((j,), dict)."""
        schema = make_1d_schema(WriteType.POINTWISE)
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)

        expected_S21    = np.random.rand(N_INNER, N_TRACE).astype(np.float32)
        expected_lockin = np.random.rand(N_INNER).astype(np.float32)

        for j in range(N_INNER):
            writer.write_point((j,), {
                "S21":      expected_S21[j],
                "lockin_X": float(expected_lockin[j]),
            })

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["S21"][:],      expected_S21)
        np.testing.assert_array_almost_equal(z["lockin_X"][:], expected_lockin)


# ===========================================================================
# ZarrWriter — IMAGEWISE (bug fix: chunk shape was transposed for non-square grids)
# ===========================================================================

def make_imagewise_schema(grid: tuple[int, int], readout: ReadoutSpecs) -> MeasurementSchema:
    a = ControlVar("a", np.arange(grid[0], dtype=np.float32))
    b = ControlVar("b", np.arange(grid[1], dtype=np.float32))
    return MeasurementSchema(
        control=[AxisSpecs([a]), AxisSpecs([b])],
        readout=[readout],
        writetype=WriteType.IMAGEWISE,
    )


class TestZarrWriterImagewise:

    # --- array creation succeeds for all readout kinds and grid shapes ---

    @pytest.mark.parametrize("grid", [(4, 4), (3, 4), (4, 3)])
    @pytest.mark.parametrize("ro", [
        ReadoutSpecs("r", DataKind.SCALAR),
        ReadoutSpecs("r", DataKind.TRACE,  shape=(8,)),
        ReadoutSpecs("r", DataKind.IMAGE,  shape=(3, 5)),
    ])
    def test_array_creation_2d_grid(self, tmp_path, grid, ro):
        schema = make_imagewise_schema(grid, ro)
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert "r" in z

    # --- chunk shape is (1,)*(n-2) + (rows, cols) [+ trailing] ---

    @pytest.mark.parametrize("grid,expected_chunk_base", [
        ((4, 4), (4, 4)),
        ((3, 4), (3, 4)),
        ((4, 3), (4, 3)),
    ])
    def test_chunk_shape_scalar_2d(self, tmp_path, grid, expected_chunk_base):
        schema = make_imagewise_schema(grid, ReadoutSpecs("r", DataKind.SCALAR))
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert z["r"].chunks == expected_chunk_base

    @pytest.mark.parametrize("grid,expected_chunk_base", [
        ((4, 4), (4, 4)),
        ((3, 4), (3, 4)),
    ])
    def test_chunk_shape_trace_2d(self, tmp_path, grid, expected_chunk_base):
        schema = make_imagewise_schema(grid, ReadoutSpecs("r", DataKind.TRACE, shape=(8,)))
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert z["r"].chunks == expected_chunk_base + (8,)

    @pytest.mark.parametrize("grid,expected_chunk_base", [
        ((4, 4), (4, 4)),
        ((3, 4), (3, 4)),
    ])
    def test_chunk_shape_image_2d(self, tmp_path, grid, expected_chunk_base):
        schema = make_imagewise_schema(grid, ReadoutSpecs("r", DataKind.IMAGE, shape=(3, 5)))
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert z["r"].chunks == expected_chunk_base + (3, 5)

    # --- 3D control grid (2, 3, 4): chunk is (1, 3, 4) + trailing ---

    def test_array_creation_3d_grid(self, tmp_path):
        a = ControlVar("a", np.arange(2, dtype=np.float32))
        b = ControlVar("b", np.arange(3, dtype=np.float32))
        c = ControlVar("c", np.arange(4, dtype=np.float32))
        for ro in [
            ReadoutSpecs("scalar", DataKind.SCALAR),
            ReadoutSpecs("trace",  DataKind.TRACE,  shape=(8,)),
            ReadoutSpecs("image",  DataKind.IMAGE,  shape=(3, 5)),
        ]:
            schema = MeasurementSchema(
                control=[AxisSpecs([a]), AxisSpecs([b]), AxisSpecs([c])],
                readout=[ro],
                writetype=WriteType.IMAGEWISE,
            )
            ZarrWriter(root=tmp_path / ro.name, schema=schema,
                       overwrite=True, initialize_arrays=True)
            z = zarr.open(str(tmp_path / ro.name / "vault.zarr"), mode="r")
            assert ro.name in z

    def test_chunk_shape_3d_grid(self, tmp_path):
        a = ControlVar("a", np.arange(2, dtype=np.float32))
        b = ControlVar("b", np.arange(3, dtype=np.float32))
        c = ControlVar("c", np.arange(4, dtype=np.float32))
        schema = MeasurementSchema(
            control=[AxisSpecs([a]), AxisSpecs([b]), AxisSpecs([c])],
            readout=[ReadoutSpecs("r", DataKind.TRACE, shape=(8,))],
            writetype=WriteType.IMAGEWISE,
        )
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert z["r"].chunks == (1, 3, 4, 8)

    # --- write_image round-trip for non-square 2D grid ---

    @pytest.mark.parametrize("grid", [(4, 4), (3, 4)])
    def test_write_image_roundtrip_scalar(self, tmp_path, grid):
        rows, cols = grid
        schema = make_imagewise_schema(grid, ReadoutSpecs("r", DataKind.SCALAR))
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        # IMAGEWISE on 2D control: index is () — write the full plane at once
        expected = np.random.rand(rows, cols).astype(np.float32)
        writer.write_image((), {"r": expected})
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["r"][:], expected)

    @pytest.mark.parametrize("grid", [(4, 4), (3, 4)])
    def test_write_image_roundtrip_trace(self, tmp_path, grid):
        rows, cols = grid
        schema = make_imagewise_schema(grid, ReadoutSpecs("r", DataKind.TRACE, shape=(8,)))
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        expected = np.random.rand(rows, cols, 8).astype(np.float32)
        writer.write_image((), {"r": expected})
        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_almost_equal(z["r"][:], expected)


# ===========================================================================
# ZarrWriter — write_npzlike (tutorial "Writing npz_like")
# ===========================================================================

class TestDump:

    def test_dump_arrays_in_vault(self, tmp_path):
        """Schemaless write via dump() without initialize_arrays."""
        writer = ZarrWriter(root=tmp_path, overwrite=True)
        data = {
            "S21":         np.random.rand(N_OUTER, N_INNER, N_TRACE),
            "temperature": np.random.rand(N_OUTER, N_INNER),
        }
        writer.dump(data)

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert "S21" in z
        assert "temperature" in z
        np.testing.assert_array_almost_equal(z["S21"][:], data["S21"])

    def test_dump_image(self, tmp_path):
        """Schemaless write of a 2-D array via dump()."""
        writer = ZarrWriter(root=tmp_path, overwrite=True)
        S34 = np.random.rand(3, N_TRACE)
        writer.dump({"S34": S34})

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        assert "S34" in z
        np.testing.assert_array_almost_equal(z["S34"][:], S34)

    def test_add_array_duplicate_without_overwrite_raises(self, tmp_path):
        writer = ZarrWriter(root=tmp_path, overwrite=False)
        writer.add_array("Freqs", np.zeros(10))
        with pytest.raises(KeyError, match="already exists"):
            writer.add_array("Freqs", np.zeros(10))



# ===========================================================================
# Metadata — ZarrWriter
# ===========================================================================

class TestZarrWriterMetadata:

    def test_zarr_attrs_keys_present(self, tmp_path):
        """write_metadata() stores date, schema, tags in zarr attrs."""
        schema = make_2d_schema()
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True)
        writer.tags = ["vna", "temperature"]
        writer.write_metadata()

        z    = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        attrs = dict(z.attrs)
        assert "date"   in attrs
        assert "schema" in attrs
        assert "tags"   in attrs
        assert attrs["tags"] == ["vna", "temperature"]

    def test_date_field_is_not_path(self, tmp_path):
        """Fixed: write_metadata() previously stored str(self.root) as the date."""
        schema = make_2d_schema()
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True)
        writer.write_metadata()

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        date_val = dict(z.attrs)["date"]
        assert str(tmp_path) not in date_val, "date field must not contain the root path"

    def test_date_field_is_iso_format(self, tmp_path):
        """date value should be parseable as an ISO datetime string."""
        from datetime import datetime
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True).write_metadata()

        z = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        date_val = dict(z.attrs)["date"]
        datetime.fromisoformat(date_val)  # must not raise

    def test_custom_meta_merged_into_attrs(self, tmp_path):
        schema = make_2d_schema()
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=True)
        writer.write_metadata(meta={"operator": "alice", "chip": "B2"})

        z     = zarr.open(str(tmp_path / "vault.zarr"), mode="r")
        attrs = dict(z.attrs)
        assert attrs.get("operator") == "alice"
        assert attrs.get("chip")     == "B2"

    def test_write_yaml_creates_file(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True).write_yaml()

        yaml_path = tmp_path / "metadata.yaml"
        assert yaml_path.exists()
        content = yaml.safe_load(yaml_path.read_text())
        assert "control" in content
        assert "readout" in content

    def test_write_yaml_warns_on_existing_file_no_overwrite(self, tmp_path):
        """Fixed: write_yaml() used to silently skip; now emits a UserWarning."""
        schema = make_2d_schema()
        writer = ZarrWriter(root=tmp_path, schema=schema, overwrite=False)
        writer.write_yaml()  # first write — creates the file

        with pytest.warns(UserWarning, match="overwrite=False"):
            writer.write_yaml()  # second write — should warn, not silently skip

    def test_write_metadata_also_writes_yaml(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True).write_metadata()
        assert (tmp_path / "metadata.yaml").exists()


# ===========================================================================
# NpzWriter (mirrors tutorial "Using NpzWriter")
# ===========================================================================

class TestNpzWriter:

    def test_write_creates_npz_file(self, tmp_path):
        NpzWriter(root=tmp_path).write({"x": np.zeros(10)})
        assert (tmp_path / "data.npz").exists()

    def test_write_data_roundtrip(self, tmp_path):
        original = {
            "S21":         np.random.rand(N_OUTER, N_INNER, N_TRACE).astype(np.float32),
            "temperature": np.random.rand(N_OUTER, N_INNER).astype(np.float32),
            "lockin_X":    np.random.rand(N_OUTER, N_INNER).astype(np.float32),
        }
        NpzWriter(root=tmp_path).write(original)

        loaded = np.load(tmp_path / "data.npz")
        for key, arr in original.items():
            np.testing.assert_array_almost_equal(loaded[key], arr)

    def test_write_compressed_creates_file(self, tmp_path):
        NpzWriter(root=tmp_path, compressed=True).write({"x": np.zeros(100)})
        assert (tmp_path / "data.npz").exists()

    def test_overwrite_replaces_file(self, tmp_path):
        writer = NpzWriter(root=tmp_path, overwrite=True)
        writer.write({"x": np.ones(5)})
        writer.write({"y": np.zeros(5)})  # overwrites

        loaded = np.load(tmp_path / "data.npz")
        assert "y" in loaded
        # old key should not be present after overwrite
        assert "x" not in loaded

    def test_write_metadata_creates_yaml(self, tmp_path):
        schema = make_2d_schema()
        writer = NpzWriter(root=tmp_path, schema=schema, tags=["test"])
        writer.write_metadata()

        yaml_path = tmp_path / "metadata.yaml"
        assert yaml_path.exists()
        content = yaml.safe_load(yaml_path.read_text())
        assert content["tags"] == ["test"]
        assert "control" in content
        assert "readout" in content

    def test_write_metadata_custom_fields(self, tmp_path):
        writer = NpzWriter(root=tmp_path)
        writer.write_metadata(meta={"experiment": "demo", "date": "2026-03-25"})

        content = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
        assert content["experiment"] == "demo"


# ===========================================================================
# ReadoutSpecs validation and normalization
# ===========================================================================

class TestReadoutSpecsValidation:

    def test_scalar_with_or_without_empty_shape_ok(self):
        ReadoutSpecs("x", DataKind.SCALAR)
        ReadoutSpecs("x", DataKind.SCALAR, shape=())

    @pytest.mark.parametrize("kind, shape", [
        (DataKind.SCALAR, (3,)),
        (DataKind.TRACE, None),
        (DataKind.TRACE, ()),
        (DataKind.TRACE, (2, 3)),
        (DataKind.IMAGE, (4,)),
        (DataKind.IMAGE, (2, 3, 4)),
    ])
    def test_kind_shape_mismatch_raises(self, kind, shape):
        with pytest.raises(ValueError, match="needs a"):
            ReadoutSpecs("x", kind, shape=shape)

    def test_kind_given_as_string_is_coerced(self):
        r = ReadoutSpecs("x", "trace", shape=(3,))
        assert r.kind is DataKind.TRACE

    def test_shape_list_is_tuple(self):
        assert ReadoutSpecs("x", DataKind.TRACE, shape=[3]).shape == (3,)

    def test_contains_list_promoted_to_component_dim(self):
        r = ReadoutSpecs("x", DataKind.TRACE, shape=(2,), contains=["X", "Y"])
        assert r.trailing_dims == [{"name": "component", "values": ["X", "Y"]}]

    def test_contains_list_on_scalar_is_not_promoted(self):
        r = ReadoutSpecs("x", DataKind.SCALAR, contains=["a", "b"])
        assert r.trailing_dims is None

    def test_contains_string_not_promoted(self):
        assert ReadoutSpecs("x", DataKind.TRACE, shape=(2,), contains="desc").trailing_dims is None

    def test_explicit_trailing_dims_beat_contains(self):
        r = ReadoutSpecs("x", DataKind.TRACE, shape=(2,), contains=["X", "Y"],
                         trailing_dims=[{"name": "ch", "values": ["a", "b"]}])
        assert r.trailing_dims[0]["name"] == "ch"

    def test_contains_length_must_match_shape(self):
        with pytest.raises(ValueError, match="expected \\(2,\\)"):
            ReadoutSpecs("x", DataKind.TRACE, shape=(2,), contains=["X", "Y", "Z"])

    def test_too_many_trailing_dims_raises(self):
        with pytest.raises(ValueError, match="trailing_dims"):
            ReadoutSpecs("x", DataKind.TRACE, shape=(3,),
                         trailing_dims=[{"name": "a"}, {"name": "b"}])

    def test_partial_trailing_dims_allowed(self):
        r = ReadoutSpecs("x", DataKind.IMAGE, shape=(3, 4), trailing_dims=[{"name": "component"}])
        assert len(r.trailing_dims) == 1

    def test_numeric_values_length_must_match(self):
        with pytest.raises(ValueError, match="expected \\(5,\\)"):
            ReadoutSpecs("x", DataKind.TRACE, shape=(5,),
                         trailing_dims=[{"name": "freq", "values": np.arange(4.0)}])

    def test_units_length_must_match(self):
        with pytest.raises(ValueError, match="needs 2 units"):
            ReadoutSpecs("x", DataKind.TRACE, shape=(2,),
                         trailing_dims=[{"name": "component", "values": ["a", "b"], "units": ["V"]}])

    def test_duplicate_names_raise(self):
        with pytest.raises(ValueError, match="duplicate"):
            ReadoutSpecs("x", DataKind.IMAGE, shape=(2, 2),
                         trailing_dims=[{"name": "a"}, {"name": "a"}])

    def test_entry_without_name_raises(self):
        with pytest.raises(ValueError, match="'name'"):
            ReadoutSpecs("x", DataKind.TRACE, shape=(2,), trailing_dims=[{"values": ["a", "b"]}])

    def test_string_ndarray_values_become_list(self):
        r = ReadoutSpecs("x", DataKind.TRACE, shape=(2,),
                         trailing_dims=[{"name": "c", "values": np.array(["a", "b"])}])
        assert r.trailing_dims[0]["values"] == ["a", "b"]

    def test_callers_dicts_not_mutated(self):
        td = [{"name": "c", "values": np.array(["a", "b"]), "units": ("V", "A")}]
        ReadoutSpecs("x", DataKind.TRACE, shape=(2,), trailing_dims=td)
        assert isinstance(td[0]["values"], np.ndarray) and td[0]["units"] == ("V", "A")

    def test_schema_manifest_strips_numeric_values_keeps_labels_and_units(self):
        ctl = AxisSpecs([ControlVar("a", np.arange(2.0))])
        r = ReadoutSpecs("r", DataKind.IMAGE, shape=(2, 3), unit="V", trailing_dims=[
            {"name": "component", "values": ["m", "p"], "units": ["dB", "deg"]},
            {"name": "freq", "unit": "Hz", "values": np.arange(3.0)},
        ])
        m = MeasurementSchema(control=[ctl], readout=[r]).manifest["readouts"][0]
        assert m["kind"] == "image" and m["shape"] == [2, 3] and m["unit"] == "V"
        assert m["trailing_dims"] == [
            {"name": "component", "values": ["m", "p"], "units": ["dB", "deg"]},
            {"name": "freq", "unit": "Hz"},
        ]


# ===========================================================================
# StreamingWriter with ReadoutSpecs
# ===========================================================================

FREQ = np.linspace(4e9, 8e9, 5)


class TestStreamingWriterSpecs:

    def _attrs(self, path):
        return dict(zarr.open_group(str(path / "vault.zarr"), mode="r").attrs)

    def test_tuple_only_unchanged(self, tmp_path):
        StreamingWriter(root=tmp_path, readouts={"x": (), "sig": (4,)})
        m = {r["name"]: r for r in self._attrs(tmp_path)["zarro"]["readouts"]}
        assert m["x"] == {"name": "x", "kind": "scalar", "shape": [], "unit": None,
                          "contains": None, "trailing_dims": []}
        assert m["sig"]["kind"] == "trace" and m["sig"]["shape"] == [4]

    def test_spec_fields_in_manifest_at_init(self, tmp_path):
        spec = ReadoutSpecs("S21", DataKind.TRACE, shape=(5,), unit="dBm",
                            trailing_dims=[{"name": "freq", "unit": "Hz", "values": FREQ}])
        StreamingWriter(root=tmp_path, readouts={"S21": spec})   # not closed
        m = self._attrs(tmp_path)["zarro"]["readouts"][0]
        assert (m["kind"], m["shape"], m["unit"]) == ("trace", [5], "dBm")
        assert m["trailing_dims"] == [{"name": "freq", "unit": "Hz"}]   # numeric values stripped

    def test_numeric_trailing_array_written_and_data_array_shape(self, tmp_path):
        spec = ReadoutSpecs("S21", DataKind.TRACE, shape=(5,),
                            trailing_dims=[{"name": "freq", "values": FREQ}])
        sw = StreamingWriter(root=tmp_path, readouts={"S21": spec})
        g = zarr.open_group(str(tmp_path / "vault.zarr"), mode="r")
        np.testing.assert_array_equal(g["S21.freq"][:], FREQ)
        assert g["S21"].shape == (0, 5)
        sw.append({"S21": np.ones(5)}); sw.flush()
        assert zarr.open_group(str(tmp_path / "vault.zarr"), mode="r")["S21"].shape == (1, 5)

    def test_name_mismatch_raises_before_creating_anything(self, tmp_path):
        spec = ReadoutSpecs("other", DataKind.SCALAR)
        with pytest.raises(ValueError, match="does not match"):
            StreamingWriter(root=tmp_path / "run", readouts={"x": spec})
        assert not (tmp_path / "run" / "vault.zarr").exists()

    def test_close_keeps_readouts_shape_mapping(self, tmp_path):
        spec = ReadoutSpecs("S21", DataKind.TRACE, shape=(5,))
        sw = StreamingWriter(root=tmp_path, readouts={"x": (), "S21": spec, "f": (2, 3)})
        sw.append({"x": 1.0, "S21": np.ones(5), "f": np.ones((2, 3))})
        sw.close()
        assert self._attrs(tmp_path)["readouts"] == {"x": [], "S21": [5], "f": [2, 3]}
        assert "zarro" in self._attrs(tmp_path)

    def test_mixed_dict(self, tmp_path):
        spec = ReadoutSpecs("v", DataKind.SCALAR, unit="V")
        StreamingWriter(root=tmp_path, readouts={"v": spec, "raw": (3,)})
        m = {r["name"]: r for r in self._attrs(tmp_path)["zarro"]["readouts"]}
        assert m["v"]["unit"] == "V" and m["raw"]["unit"] is None and m["raw"]["kind"] == "trace"

    def test_tuple_path_still_accepts_3d_shape(self, tmp_path):
        StreamingWriter(root=tmp_path, readouts={"vol": (2, 2, 2)})
        assert self._attrs(tmp_path)["zarro"]["readouts"][0]["kind"] == "image"


# ===========================================================================
# Coordinate precision: control coordinates are float64, readouts follow dtype
# ===========================================================================

GHZ = np.array([5.123456789e9, 5.123456790e9, 5.123456791e9])   # 1 Hz apart; float32 merges them


def _ghz_writer(tmp_path, control, dtype="float32", readout=None):
    schema = MeasurementSchema(
        control=control,
        readout=readout or [ReadoutSpecs("x", DataKind.SCALAR)],
    )
    return ZarrWriter(root=tmp_path, schema=schema, dtype=dtype, overwrite=True, initialize_arrays=True)


def _group(tmp_path):
    return zarr.open_group(str(tmp_path / "vault.zarr"), mode="r")


class TestCoordinateDtype:

    def test_float32_would_have_merged_the_points(self):
        assert len(np.unique(GHZ.astype(np.float32))) == 1     # premise of the fix

    def test_1d_ghz_roundtrip_distinct_float64(self, tmp_path):
        _ghz_writer(tmp_path, [AxisSpecs([ControlVar("f", GHZ, unit="Hz")])])
        z = _group(tmp_path)
        assert z["f"].dtype == np.float64
        np.testing.assert_array_equal(z["f"][:], GHZ)
        assert np.all(np.diff(z["f"][:]) == 1.0)

    def test_2d_ghz_roundtrip_distinct_float64(self, tmp_path):
        g2 = GHZ + 1e9
        _ghz_writer(tmp_path, [AxisSpecs([ControlVar("f1", GHZ)]), AxisSpecs([ControlVar("f2", g2)])])
        z = _group(tmp_path)
        for name, expected in (("f1", GHZ), ("f2", g2)):
            assert z[name].dtype == np.float64
            np.testing.assert_array_equal(z[name][:], expected)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "float16"])
    def test_readout_follows_dtype_coords_stay_float64(self, tmp_path, dtype):
        _ghz_writer(tmp_path, [AxisSpecs([ControlVar("f", GHZ)])], dtype=dtype)
        z = _group(tmp_path)
        assert z["x"].dtype == np.dtype(dtype)
        assert z["f"].dtype == np.float64

    def test_default_readout_dtype_is_float32(self, tmp_path):
        _ghz_writer(tmp_path, [AxisSpecs([ControlVar("f", GHZ)])])
        assert _group(tmp_path)["x"].dtype == np.float32

    def test_multi_control_axis_both_float64(self, tmp_path):
        a = ControlVar("fa", GHZ)
        b = ControlVar("fb", GHZ + 2e9)
        _ghz_writer(tmp_path, [AxisSpecs([a, b])])
        z = _group(tmp_path)
        np.testing.assert_array_equal(z["fa"][:], GHZ)
        np.testing.assert_array_equal(z["fb"][:], GHZ + 2e9)
        assert z["fa"].dtype == z["fb"].dtype == np.float64

    def test_reverse_control_var_float64_forward_then_backward(self, tmp_path):
        _ghz_writer(tmp_path, [AxisSpecs([ControlVar("f", GHZ, reverse=True)])])
        z = _group(tmp_path)
        assert z["f"].dtype == np.float64
        np.testing.assert_array_equal(z["f"][:], np.concatenate([GHZ, GHZ[::-1]]))

    def test_integer_control_values_stored_exactly_as_float64(self, tmp_path):
        _ghz_writer(tmp_path, [AxisSpecs([ControlVar("n", np.arange(5))])])
        z = _group(tmp_path)
        assert z["n"].dtype == np.float64
        np.testing.assert_array_equal(z["n"][:], np.arange(5))

    def test_coordinate_arrays_are_not_sharded(self, tmp_path):
        w = _ghz_writer(tmp_path, [AxisSpecs([ControlVar("f", GHZ)])], dtype="float16")
        assert w.vault["f"].shards is None

    def test_trailing_dims_keep_their_own_dtype(self, tmp_path):
        spec = ReadoutSpecs("S", DataKind.TRACE, shape=(3,), trailing_dims=[
            {"name": "freq", "values": GHZ}])
        _ghz_writer(tmp_path, [AxisSpecs([ControlVar("a", np.arange(2.0))])], readout=[spec])
        z = _group(tmp_path)
        assert z["S.freq"].dtype == np.float64
        np.testing.assert_array_equal(z["S.freq"][:], GHZ)

    def test_streaming_writer_unchanged(self, tmp_path):
        sw = StreamingWriter(root=tmp_path, readouts={"x": ()}, dtype="float32")
        sw.append({"x": 1.0}, timestamp=1.7e9 + 0.123456)
        sw.close()
        z = _group(tmp_path)
        assert z["_time"].dtype == np.float64 and z["x"].dtype == np.float32
        assert z["_time"][0] == 1.7e9 + 0.123456
        assert sorted(k for k in z.array_keys()) == ["_time", "x"]


# ===========================================================================
# ExperimentID
# ===========================================================================

def _run(root, name, artifact="vault.zarr"):
    """A run folder: numbered directory holding zarro output."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if artifact == "vault.zarr":
        (d / artifact).mkdir()
    elif artifact:
        (d / artifact).write_text("")
    return d


def _listing(monkeypatch, order):
    """Force the directory listing order (the filesystem's order is unspecified)."""
    from zarro import core
    real = core.list_dirs
    monkeypatch.setattr(core, "list_dirs", lambda root: order(real(root)))


class TestExperimentID:

    def test_empty_root(self, tmp_path):
        eid = ExperimentID(tmp_path / "new")          # also creates the root
        assert eid.current_id() == 0 and eid.next_id() == 1
        assert eid.next_dir() == tmp_path / "new" / "0001"
        assert eid.current_dir() == tmp_path / "new" / "0000"

    def test_normal_numbering(self, tmp_path):
        for n in ("0001", "0002", "0003"):
            _run(tmp_path, n)
        eid = ExperimentID(tmp_path)
        assert eid.current_id() == 3 and eid.next_id() == 4
        assert eid.next_dir() == tmp_path / "0004" and eid.current_dir() == tmp_path / "0003"

    def test_non_numeric_folders_ignored(self, tmp_path, monkeypatch):
        _run(tmp_path, "0001"); _run(tmp_path, "0002")
        for name in ("notes", "zzz", ".git", "__pycache__", "note"):
            (tmp_path / name).mkdir()
        for order in (sorted, lambda d: sorted(d, reverse=True)):   # a non-numeric folder listed last
            _listing(monkeypatch, order)
            assert ExperimentID(tmp_path).current_id() == 2

    def test_year_like_folder_is_not_a_run(self, tmp_path):
        _run(tmp_path, "0004")
        (tmp_path / "2026_backup").mkdir()
        (tmp_path / "20260101").mkdir()
        assert ExperimentID(tmp_path).next_id() == 5

    def test_year_like_folder_with_a_vault_is_a_run(self, tmp_path):
        _run(tmp_path, "0004"); _run(tmp_path, "2026_backup")
        assert ExperimentID(tmp_path).current_id() == 2026

    def test_gaps_use_the_largest_id_not_the_count(self, tmp_path):
        _run(tmp_path, "0001"); _run(tmp_path, "0004")        # 0002, 0003 were deleted
        for i in range(5):
            (tmp_path / f"other{i}").mkdir()                   # unrelated folders must not raise the floor
        assert ExperimentID(tmp_path).current_id() == 4

    def test_deleting_the_newest_run_frees_its_id(self, tmp_path):
        for n in ("0001", "0002", "0003"):
            _run(tmp_path, n)
        import shutil
        shutil.rmtree(tmp_path / "0003")
        assert ExperimentID(tmp_path).next_id() == 3

    @pytest.mark.parametrize("name, order", [
        ("ascending", sorted),
        ("descending", lambda d: sorted(d, reverse=True)),
        ("shuffled", lambda d: sorted(d, key=lambda p: hash(p.name) % 7)),
    ])
    def test_independent_of_listing_order(self, tmp_path, monkeypatch, name, order):
        for n in ("0001", "0003"):
            _run(tmp_path, n)
        _listing(monkeypatch, order)
        eid = ExperimentID(tmp_path)
        assert eid.current_id() == 3 and eid.next_dir() == tmp_path / "0004"

    def test_unsorted_listing_with_get_folders_patched(self, tmp_path, monkeypatch):
        for n in ("0001", "0003"):
            _run(tmp_path, n)
        monkeypatch.setattr(ExperimentID, "get_folders", lambda self: [tmp_path / "0003", tmp_path / "0001"])
        assert ExperimentID(tmp_path).next_id() == 4

    def test_suffixed_folders(self, tmp_path):
        _run(tmp_path, "0001"); _run(tmp_path, "0007_fridge")
        eid = ExperimentID(tmp_path)
        assert eid.current_id() == 7 and eid.next_dir() == tmp_path / "0008"

    def test_same_id_with_and_without_suffix(self, tmp_path):
        _run(tmp_path, "0007"); _run(tmp_path, "0007_fridge")
        assert ExperimentID(tmp_path).current_id() == 7

    def test_numbered_folder_without_output_is_not_a_run(self, tmp_path):
        _run(tmp_path, "0001")
        (tmp_path / "0002").mkdir()                            # e.g. a manual mkdir or a crash before the writer
        eid = ExperimentID(tmp_path)
        assert eid.current_id() == 1 and eid.next_dir() == tmp_path / "0002"   # id is reused

    def test_numbered_folder_with_only_a_vault_counts(self, tmp_path):
        """A run that failed after the writer created vault.zarr but before any data keeps its id."""
        _run(tmp_path, "0001"); _run(tmp_path, "0002")
        assert ExperimentID(tmp_path).next_id() == 3

    @pytest.mark.parametrize("artifact", ["data.npz", "metadata.yaml"])
    def test_npz_and_metadata_only_runs_count(self, tmp_path, artifact):
        _run(tmp_path, "0001"); _run(tmp_path, "0002", artifact=artifact)
        assert ExperimentID(tmp_path).current_id() == 2

    def test_real_npz_writer_run_counts(self, tmp_path):
        eid = ExperimentID(tmp_path)
        schema = MeasurementSchema(control=[AxisSpecs([ControlVar("a", np.arange(2.0))])],
                                   readout=[ReadoutSpecs("x", DataKind.SCALAR)])
        NpzWriter(root=eid.next_dir(), schema=schema, overwrite=False).write({"x": np.arange(2.0)})
        assert eid.current_id() == 1 and eid.next_dir() == tmp_path / "0002"

    def test_vault_zarr_must_be_a_directory(self, tmp_path):
        d = tmp_path / "0001"; d.mkdir()
        (d / "vault.zarr").write_text("")                      # a file, not a zarr group
        assert ExperimentID(tmp_path).current_id() == 0

    def test_plain_file_with_a_run_like_name_ignored(self, tmp_path):
        _run(tmp_path, "0001")
        (tmp_path / "0009").write_text("")
        assert ExperimentID(tmp_path).current_id() == 1

    def test_ids_above_9999_compare_numerically(self, tmp_path):
        for n in ("9999", "10000", "10001_x"):
            _run(tmp_path, n)
        eid = ExperimentID(tmp_path)
        assert eid.current_id() == 10001
        assert eid.next_dir() == tmp_path / "10002"

    def test_rolls_over_after_9999(self, tmp_path):
        _run(tmp_path, "9999")
        assert ExperimentID(tmp_path).next_dir() == tmp_path / "10000"

    def test_non_canonical_padding_ignored(self, tmp_path):
        _run(tmp_path, "0003"); _run(tmp_path, "00007")
        assert ExperimentID(tmp_path).current_id() == 3

    def test_get_dir_name_unchanged(self, tmp_path):
        eid = ExperimentID(tmp_path)
        assert eid.get_dir_name(2) == tmp_path / "0002"
        assert eid.get_dir_name("special_run") == tmp_path / "special_run"
        assert eid.get_dir_name(12345) == tmp_path / "12345"

    def test_get_folders_lists_all_subfolders(self, tmp_path):
        _run(tmp_path, "0001"); (tmp_path / "notes").mkdir(); (tmp_path / "f.txt").write_text("")
        assert sorted(p.name for p in ExperimentID(tmp_path).get_folders()) == ["0001", "notes"]

    def test_zarr_writer_round_trip_numbering(self, tmp_path):
        eid = ExperimentID(tmp_path)
        schema = MeasurementSchema(control=[AxisSpecs([ControlVar("a", np.arange(2.0))])],
                                   readout=[ReadoutSpecs("x", DataKind.SCALAR)])
        for expected in (1, 2, 3):
            assert eid.next_id() == expected
            ZarrWriter(root=eid.next_dir(), schema=schema, initialize_arrays=True)
