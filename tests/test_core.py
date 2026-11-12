"""
Tests for zarro/core2.py.
Covers all write patterns shown in tutorials/demo.ipynb plus error cases and
the fixes applied in core2 (error messages, shape validation, date field, etc.).

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
    ReadoutSpecs,
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
