"""
Tests for src/zarro/reader.py.

All vaults are written with real ZarrWriter / StreamingWriter instances into
pytest tmp_path directories.  No mocks are used.

Run with:  pytest tests/test_reader.py
"""

import numpy as np
import pytest

from zarro import (
    AxisSpecs,
    ControlVar,
    DataKind,
    MeasurementSchema,
    ReadoutSpecs,
    StreamingWriter,
    WriteType,
    ZarrWriter,
)
from zarro.reader import Axis, Readout, Vault, VaultError, open_vault

# ---------------------------------------------------------------------------
# Small but non-trivial dimensions so tests run fast
# ---------------------------------------------------------------------------
N_A = 3    # outer axis length
N_B = 4    # inner axis length
N_T = 6    # trace length
N_H = 3    # image height
N_W = 5    # image width


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def make_1d_schema(writetype=WriteType.POINTWISE):
    """Single axis with two simultaneous control variables."""
    c = ControlVar("c", np.linspace(0, 2, N_B, dtype=np.float32), unit="T")
    d = ControlVar("d", np.linspace(1, 3, N_B, dtype=np.float32))  # no unit
    return MeasurementSchema(
        control=[AxisSpecs([c, d])],
        readout=[
            ReadoutSpecs("scalar", DataKind.SCALAR),
            ReadoutSpecs("trace",  DataKind.TRACE, shape=(N_T,)),
        ],
        writetype=writetype,
    )


def make_2d_schema(writetype=WriteType.POINTWISE, with_image=False):
    a = ControlVar("a", np.linspace(0, 1, N_A, dtype=np.float32), unit="V")
    b = ControlVar("b", np.linspace(0, 1, N_B, dtype=np.float32), unit="Hz")
    readouts = [
        ReadoutSpecs("scalar", DataKind.SCALAR, unit="dBm"),
        ReadoutSpecs("trace",  DataKind.TRACE,  shape=(N_T,), unit="V", contains="S21"),
    ]
    if with_image:
        readouts.append(ReadoutSpecs("image", DataKind.IMAGE, shape=(N_H, N_W)))
    return MeasurementSchema(
        control=[AxisSpecs([a]), AxisSpecs([b])],
        readout=readouts,
        writetype=writetype,
    )


def write_all_2d(writer, scalar_data, trace_data):
    for i in range(N_A):
        for j in range(N_B):
            writer.write_point((i, j), {
                "scalar": float(scalar_data[i, j]),
                "trace":  trace_data[i, j],
            })


def finished_vault(tmp_path):
    """Full 2D sweep, all points written, metadata written."""
    schema = make_2d_schema()
    w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
    scalar = np.random.rand(N_A, N_B).astype(np.float32)
    trace  = np.random.rand(N_A, N_B, N_T).astype(np.float32)
    write_all_2d(w, scalar, trace)
    w.write_metadata()
    return scalar, trace


# ===========================================================================
# TestVaultStructure1D
# ===========================================================================

class TestVaultStructure1D:

    def test_kind_is_sweep(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.kind == "sweep"

    def test_shape(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.shape == (N_B,)

    def test_one_axis_two_controls(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert len(v.axes) == 1
        ax = v.axes[0]
        assert ax.index == 0
        assert ax.controls == ("c", "d")
        assert ax.length == N_B

    def test_units_on_multi_control_axis(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.axes[0].units["c"] == "T"
        assert v.axes[0].units["d"] is None

    def test_coords_shape_and_values(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert set(v.coords.keys()) == {"c", "d"}
        assert v.coords["c"].shape == (N_B,)
        np.testing.assert_array_almost_equal(
            v.coords["c"], np.linspace(0, 2, N_B, dtype=np.float32)
        )

    def test_coord_axis_mapping(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.coord_axis["c"] == 0
        assert v.coord_axis["d"] == 0

    def test_readout_kinds_and_shapes(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.readouts["scalar"].kind == "scalar"
        assert v.readouts["scalar"].shape == ()
        assert v.readouts["trace"].kind == "trace"
        assert v.readouts["trace"].shape == (N_T,)


# ===========================================================================
# TestVaultStructure2D
# ===========================================================================

class TestVaultStructure2D:

    def test_shape(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.shape == (N_A, N_B)

    def test_two_axes_one_control_each(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert len(v.axes) == 2
        assert v.axes[0].controls == ("a",)
        assert v.axes[1].controls == ("b",)
        assert v.axes[0].length == N_A
        assert v.axes[1].length == N_B

    def test_axis_units(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.axes[0].units["a"] == "V"
        assert v.axes[1].units["b"] == "Hz"

    def test_coord_axis_mapping(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.coord_axis["a"] == 0
        assert v.coord_axis["b"] == 1

    def test_scalar_readout_unit(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        r = v.readouts["scalar"]
        assert r.kind == "scalar"
        assert r.shape == ()
        assert r.unit == "dBm"

    def test_trace_readout_unit_and_contains(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        r = v.readouts["trace"]
        assert r.kind == "trace"
        assert r.shape == (N_T,)
        assert r.unit == "V"
        assert r.contains == "S21"

    def test_image_readout_kind_and_shape(self, tmp_path):
        schema = make_2d_schema(with_image=True)
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        r = v.readouts["image"]
        assert r.kind == "image"
        assert r.shape == (N_H, N_W)

    def test_tags_roundtrip(self, tmp_path):
        schema = make_2d_schema()
        w = ZarrWriter(
            root=tmp_path, schema=schema, overwrite=True,
            initialize_arrays=True, tags=["vna", "test"]
        )
        w.write_metadata()
        v = Vault(tmp_path)
        assert v.tags == ["vna", "test"]


# ===========================================================================
# TestVaultData — read() and values() roundtrip
# ===========================================================================

class TestVaultData:

    def test_read_full_scalar_roundtrip(self, tmp_path):
        scalar, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        np.testing.assert_array_almost_equal(v.read("scalar"), scalar)

    def test_read_row_selection(self, tmp_path):
        scalar, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        np.testing.assert_array_almost_equal(v.read("scalar", (0,)), scalar[0])

    def test_read_column_selection(self, tmp_path):
        scalar, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        np.testing.assert_array_almost_equal(
            v.read("scalar", (slice(None), 2)), scalar[:, 2]
        )

    def test_values_full_run_equals_data(self, tmp_path):
        scalar, trace = finished_vault(tmp_path)
        v = Vault(tmp_path)
        np.testing.assert_array_almost_equal(v.values("scalar"), scalar)
        np.testing.assert_array_almost_equal(v.values("trace"), trace)

    def test_values_full_run_no_nan(self, tmp_path):
        finished_vault(tmp_path)
        v = Vault(tmp_path)
        assert not np.isnan(v.values("scalar")).any()
        assert not np.isnan(v.values("trace")).any()

    def test_values_float64_dtype_roundtrip(self, tmp_path):
        """Non-default dtype float64 round-trips without precision loss."""
        a = ControlVar("a", np.arange(N_A, dtype=np.float64))
        b = ControlVar("b", np.arange(N_B, dtype=np.float64))
        schema = MeasurementSchema(
            control=[AxisSpecs([a]), AxisSpecs([b])],
            readout=[ReadoutSpecs("x", DataKind.SCALAR)],
            writetype=WriteType.POINTWISE,
        )
        w = ZarrWriter(root=tmp_path, schema=schema, dtype="float64",
                       overwrite=True, initialize_arrays=True)
        expected = np.random.rand(N_A, N_B)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"x": expected[i, j]})
        w.write_metadata()
        v = Vault(tmp_path)
        np.testing.assert_array_almost_equal(v.values("x"), expected)


# ===========================================================================
# TestWrittenMaskSweep — partial writes, chunk-file mask and shard heuristic
# ===========================================================================

class TestWrittenMaskSweep:

    def test_nothing_written_all_false(self, tmp_path):
        """Before any write_point call, every position is unwritten."""
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        v = Vault(tmp_path)
        assert not v.written_mask("scalar").any()
        assert not v.written_mask("trace").any()

    def test_mid_run_scalar_chunk_mask(self, tmp_path):
        """POINTWISE mid-run: scalar chunk files mark exactly written positions."""
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        for j in range(N_B):
            w.write_point((0, j), {"scalar": 1.0, "trace": np.ones(N_T)})
        v = Vault(tmp_path)
        mask = v.written_mask("scalar")
        assert mask[0, :].all()
        assert not mask[1:, :].any()

    def test_mid_run_values_nan_at_unwritten(self, tmp_path):
        """values() returns NaN exactly at unwritten positions."""
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        s = np.random.rand(N_A, N_B).astype(np.float32)
        for j in range(N_B):
            w.write_point((0, j), {"scalar": float(s[0, j]), "trace": np.ones(N_T)})
        v = Vault(tmp_path)
        vals = v.values("scalar")
        np.testing.assert_array_almost_equal(vals[0], s[0])
        assert np.isnan(vals[1:]).all()

    def test_mid_run_sharded_trace_nonzero_heuristic(self, tmp_path):
        """Sharded trace array: non-zero heuristic marks correct positions."""
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        for j in range(N_B):
            w.write_point((0, j), {"scalar": 1.0, "trace": np.ones(N_T)})
        v = Vault(tmp_path)
        mask = v.written_mask("trace")
        assert mask[0, :].all()
        assert not mask[1:, :].any()

    def test_zero_data_before_last_nonzero_recovered_by_propagation(self, tmp_path):
        """Genuine 0.0 trace at an earlier position is recovered by order propagation.

        Write point (0,0) with zero trace (may not create shard / chunk file),
        then (0,1) with non-zero trace.  The shard now exists.  The non-zero
        heuristic alone marks only (0,1), but propagation extends the mask to
        (0,0) because order[0,0]=0 <= order[0,1]=1.
        """
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_point((0, 0), {"scalar": 0.5, "trace": np.zeros(N_T)})
        w.write_point((0, 1), {"scalar": 1.0, "trace": np.ones(N_T)})
        v = Vault(tmp_path)
        mask = v.written_mask("trace")
        assert mask[0, 0], "genuine 0.0 trace at (0,0) must be recovered by propagation"
        assert mask[0, 1]
        assert not mask[0, 2:].any()
        assert not mask[1:].any()

    def test_mid_run_tracewise_one_trace_written(self, tmp_path):
        """TRACEWISE: one write_trace call marks the full inner row."""
        schema = make_2d_schema(writetype=WriteType.TRACEWISE)
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_trace((0,), {
            "scalar": np.ones(N_B, dtype=np.float32),
            "trace":  np.ones((N_B, N_T), dtype=np.float32),
        })
        v = Vault(tmp_path)
        mask = v.written_mask("scalar")
        assert mask[0, :].all()
        assert not mask[1:, :].any()

    def test_mid_run_imagewise_3d_control(self, tmp_path):
        """IMAGEWISE on a 3D control: one write_image call marks the right slice."""
        a = ControlVar("a", np.arange(2, dtype=np.float32))
        b = ControlVar("b", np.arange(N_A, dtype=np.float32))
        c = ControlVar("c", np.arange(N_B, dtype=np.float32))
        schema = MeasurementSchema(
            control=[AxisSpecs([a]), AxisSpecs([b]), AxisSpecs([c])],
            readout=[ReadoutSpecs("scalar", DataKind.SCALAR)],
            writetype=WriteType.IMAGEWISE,
        )
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_image((0,), {"scalar": np.ones((N_A, N_B), dtype=np.float32)})
        v = Vault(tmp_path)
        mask = v.written_mask("scalar")
        assert mask[0, :, :].all()
        assert not mask[1, :, :].any()


# ===========================================================================
# TestWrittenMaskOrder — custom visiting order
# ===========================================================================

class TestWrittenMaskOrder:

    def test_snake_order_prefix_rule(self, tmp_path):
        """Boustrophedon order: writing the first k steps marks exactly those k positions.

        Row 0 goes left-to-right, row 1 right-to-left, row 2 left-to-right, ...
        We stop after the first N_B+2 steps (all of row 0 and 2 steps into row 1).
        With a custom order array, written_mask must respect the prefix rule,
        marking all and only the visited positions.
        """
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)

        order = np.empty((N_A, N_B), dtype=int)
        step = 0
        for i in range(N_A):
            cols = range(N_B) if i % 2 == 0 else range(N_B - 1, -1, -1)
            for j in cols:
                order[i, j] = step
                step += 1

        positions = [None] * (N_A * N_B)
        for i in range(N_A):
            for j in range(N_B):
                positions[order[i, j]] = (i, j)

        n_written = N_B + 2
        for k in range(n_written):
            i, j = positions[k]
            w.write_point((i, j), {"scalar": 1.0, "trace": np.ones(N_T)})

        v = Vault(tmp_path)
        mask = v.written_mask("scalar", order=order)
        expected = order < n_written
        np.testing.assert_array_equal(mask, expected)


# ===========================================================================
# TestInterrupted — partial run with extra metadata
# ===========================================================================

class TestInterrupted:

    def test_extra_meta_stored_and_readable(self, tmp_path):
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_point((0, 0), {"scalar": 1.0, "trace": np.ones(N_T)})
        w.write_metadata(meta={"status": "error", "stopped_at": "row 0"})
        v = Vault(tmp_path)
        assert v.metadata["status"] == "error"
        assert v.metadata["stopped_at"] == "row 0"

    def test_partial_data_readable_with_nan_at_rest(self, tmp_path):
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        s = np.random.rand(N_A, N_B).astype(np.float32)
        w.write_point((0, 0), {"scalar": float(s[0, 0]), "trace": np.ones(N_T)})
        w.write_metadata(meta={"status": "error"})
        v = Vault(tmp_path)
        vals = v.values("scalar")
        assert float(vals[0, 0]) == pytest.approx(float(s[0, 0]))
        assert np.isnan(vals[0, 1:]).all()
        assert np.isnan(vals[1:]).all()


# ===========================================================================
# TestStreamingWriter (monitor vaults)
# ===========================================================================

class TestStreamingWriter:

    def _make_writer(self, tmp_path):
        return StreamingWriter(
            root=tmp_path,
            readouts={"x": (), "sig": (N_T,), "frame": (N_H, N_W)},
            chunk_size=4,
        )

    def test_kind_is_monitor(self, tmp_path):
        sw = self._make_writer(tmp_path)
        sw.append({"x": 1.0, "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))})
        sw.flush()
        assert Vault(tmp_path).kind == "monitor"

    def test_shape_and_time_after_flush(self, tmp_path):
        sw = self._make_writer(tmp_path)
        sw.append({"x": 1.0, "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))}, timestamp=1000.0)
        sw.append({"x": 2.0, "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))}, timestamp=2000.0)
        sw.flush()
        v = Vault(tmp_path)
        assert v.shape == (2,)
        np.testing.assert_array_almost_equal(v.time, [1000.0, 2000.0])

    def test_shape_grows_across_flushes(self, tmp_path):
        sw = self._make_writer(tmp_path)
        for i in range(3):
            sw.append({"x": float(i), "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))})
        sw.flush()
        assert Vault(tmp_path).shape == (3,)

        for i in range(2):
            sw.append({"x": float(i + 10), "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))})
        sw.flush()
        assert Vault(tmp_path).shape == (5,)

    def test_readout_shapes_inferred_correctly(self, tmp_path):
        sw = self._make_writer(tmp_path)
        sw.append({"x": 1.0, "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))})
        sw.flush()
        v = Vault(tmp_path)
        assert v.readouts["x"].kind == "scalar" and v.readouts["x"].shape == ()
        assert v.readouts["sig"].kind == "trace" and v.readouts["sig"].shape == (N_T,)
        assert v.readouts["frame"].kind == "image" and v.readouts["frame"].shape == (N_H, N_W)

    def test_written_mask_always_true_for_monitor(self, tmp_path):
        sw = self._make_writer(tmp_path)
        sw.append({"x": 1.0, "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))})
        sw.flush()
        assert Vault(tmp_path).written_mask("x").all()

    def test_values_match_appended_data(self, tmp_path):
        sw = self._make_writer(tmp_path)
        xs = [1.5, 2.5, 3.5]
        for x in xs:
            sw.append({
                "x": x,
                "sig":   np.ones(N_T) * x,
                "frame": np.ones((N_H, N_W)) * x,
            })
        sw.close()
        v = Vault(tmp_path)
        np.testing.assert_array_almost_equal(
            v.values("x"), np.array(xs, dtype=np.float32)
        )

    def test_opened_before_close_discovers_readouts(self, tmp_path):
        """Vault opened before close() (no zarr attrs / no yaml) can still be read."""
        sw = self._make_writer(tmp_path)
        sw.append({"x": 1.0, "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))})
        sw.flush()
        v = Vault(tmp_path)   # close() not yet called
        assert v.kind == "monitor"
        assert set(v.readouts.keys()) == {"x", "sig", "frame"}
        assert v.shape == (1,)

    def test_opened_after_close_has_full_metadata(self, tmp_path):
        sw = self._make_writer(tmp_path)
        sw.append({"x": 1.0, "sig": np.ones(N_T), "frame": np.ones((N_H, N_W))})
        sw.close(meta={"sensor": "A"})
        v = Vault(tmp_path)
        assert "date" in v.metadata
        assert v.metadata.get("sensor") == "A"
        assert v.tags == []


# ===========================================================================
# TestFallbacks — control/readouts keyword arguments
# ===========================================================================

class TestFallbacks:

    def test_fallback_args_supply_structure_when_yaml_missing(self, tmp_path):
        """When yaml is gone and zarr attrs never written, fallback kwargs work."""
        schema = make_2d_schema()
        # initialize_arrays=True writes yaml; do NOT call write_metadata (no zarr attrs)
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        # Corrupt the yaml
        (tmp_path / "metadata.yaml").write_text("[unclosed list")

        control = {
            "axis0": {"a": {"unit": "V"}},
            "axis1": {"b": {"unit": "Hz"}},
        }
        readouts = {
            "scalar": {"kind": "scalar", "unit": "dBm"},
            "trace":  {"kind": "trace",  "shape": [N_T]},
        }
        v = Vault(tmp_path, control=control, readouts=readouts)
        assert v.shape == (N_A, N_B)
        assert v.axes[0].controls == ("a",)
        assert v.axes[0].units["a"] == "V"
        assert v.readouts["scalar"].kind == "scalar"

    def test_no_metadata_no_fallback_raises_vault_error(self, tmp_path):
        """Sweep vault with no readable metadata and no fallback raises VaultError."""
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        (tmp_path / "metadata.yaml").write_text("[unclosed list")
        with pytest.raises(VaultError):
            Vault(tmp_path)

    def test_corrupt_yaml_ignored_when_zarr_attrs_present(self, tmp_path):
        """Half-written yaml is silently ignored; zarr attrs carry the structure."""
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_metadata()
        (tmp_path / "metadata.yaml").write_text("[unclosed list")
        v = Vault(tmp_path)
        assert v.shape == (N_A, N_B)


# ===========================================================================
# TestErrors
# ===========================================================================

class TestErrors:

    def test_no_vault_zarr_raises(self, tmp_path):
        with pytest.raises(VaultError):
            Vault(tmp_path)

    def test_open_vault_function_works(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(
            root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True
        ).write_metadata()
        v = open_vault(tmp_path)
        assert v.shape == (N_A, N_B)

    def test_vault_repr(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(
            root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True
        ).write_metadata()
        v = Vault(tmp_path)
        r = repr(v)
        assert "sweep" in r
        assert "3x4" in r
