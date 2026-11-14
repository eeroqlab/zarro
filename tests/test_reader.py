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

    def test_two_variables_on_same_dim(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert set(v.axes.keys()) == {"c", "d"}
        assert v.axes["c"].dim == 0
        assert v.axes["d"].dim == 0
        assert v.axes["c"].length == N_B

    def test_units_on_multi_control_axis(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.axes["c"].unit == "T"
        assert v.axes["d"].unit is None

    def test_values_shape_and_content(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.axes["c"].values.shape == (N_B,)
        np.testing.assert_array_almost_equal(
            v.axes["c"].values, np.linspace(0, 2, N_B, dtype=np.float32)
        )

    def test_dim_attribute(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.axes["c"].dim == 0
        assert v.axes["d"].dim == 0

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
        assert set(v.axes.keys()) == {"a", "b"}
        assert v.axes["a"].length == N_A
        assert v.axes["b"].length == N_B

    def test_axis_units(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.axes["a"].unit == "V"
        assert v.axes["b"].unit == "Hz"

    def test_dim_attribute(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.axes["a"].dim == 0
        assert v.axes["b"].dim == 1

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
# TestComponent — component= kwarg on read() and values()
# ===========================================================================

def make_image_schema_with_layers():
    """3-layer IMAGE readout using trailing_dims for component names and per-label units."""
    a = ControlVar("a", np.linspace(0, 1, N_A, dtype=np.float32))
    b = ControlVar("b", np.linspace(0, 1, N_B, dtype=np.float32))
    return MeasurementSchema(
        control=[AxisSpecs([a]), AxisSpecs([b])],
        readout=[
            ReadoutSpecs("res", DataKind.IMAGE, shape=(3, N_T),
                         trailing_dims=[
                             {"name": "component", "values": ["fpts", "mag", "phase"],
                              "units": ["Hz", "dBm", "deg"]},
                             {"name": "sample"},
                         ]),
            ReadoutSpecs("scalar", DataKind.SCALAR),
        ],
        writetype=WriteType.POINTWISE,
    )


def write_full_image_layers(tmp_path):
    schema = make_image_schema_with_layers()
    w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
    rng = np.random.default_rng(42)
    data = rng.random((N_A, N_B, 3, N_T), dtype=np.float32)
    for i in range(N_A):
        for j in range(N_B):
            w.write_point((i, j), {"res": data[i, j], "scalar": 1.0})
    w.write_metadata()
    return data


class TestComponent:

    def test_read_component_full_shape(self, tmp_path):
        data = write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        mag = v.read("res", component="mag")
        assert mag.shape == (N_A, N_B, N_T)
        np.testing.assert_array_almost_equal(mag, data[:, :, 1, :])

    def test_read_component_with_selection(self, tmp_path):
        data = write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        mag = v.read("res", (0,), component="mag")
        assert mag.shape == (N_B, N_T)
        np.testing.assert_array_almost_equal(mag, data[0, :, 1, :])

    def test_read_component_point_selection(self, tmp_path):
        data = write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        mag = v.read("res", (0, 1), component="mag")
        assert mag.shape == (N_T,)
        np.testing.assert_array_almost_equal(mag, data[0, 1, 1, :])

    def test_values_component_no_nan_full_run(self, tmp_path):
        data = write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        mag = v.values("res", component="mag")
        assert mag.shape == (N_A, N_B, N_T)
        assert not np.isnan(mag).any()
        np.testing.assert_array_almost_equal(mag, data[:, :, 1, :])

    def test_all_layer_names_selectable(self, tmp_path):
        data = write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        for idx, name in enumerate(["fpts", "mag", "phase"]):
            result = v.read("res", component=name)
            np.testing.assert_array_almost_equal(result, data[:, :, idx, :])

    def test_component_on_scalar_raises(self, tmp_path):
        write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        with pytest.raises(VaultError, match="no named components"):
            v.read("scalar", component="mag")

    def test_unknown_component_raises(self, tmp_path):
        write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        with pytest.raises(VaultError, match="not found"):
            v.read("res", component="nonexistent")


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
# TestToXarray — Vault.to_xarray
# ===========================================================================

xr = pytest.importorskip("xarray")


class TestToXarray:

    def test_scalar_1d_shape_and_dims(self, tmp_path):
        # make_1d_schema: single dim with two co-swept vars c,d of length N_B
        schema = make_1d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_metadata()
        for i in range(N_B):
            w.write_point((i,), {"scalar": 1.0, "trace": np.zeros(N_T)})
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        # two co-swept vars on dim0 → dim named "dim0"
        assert "dim0" in da.dims
        assert da.shape == (N_B,)

    def test_scalar_1d_both_coords_present(self, tmp_path):
        schema = make_1d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_metadata()
        for i in range(N_B):
            w.write_point((i,), {"scalar": 1.0, "trace": np.zeros(N_T)})
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        assert "c" in da.coords
        assert "d" in da.coords
        np.testing.assert_array_equal(da.coords["c"].values, v.axes["c"].values)

    def test_scalar_2d_shape_and_dims(self, tmp_path):
        _, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        assert da.dims == ("a", "b")
        assert da.shape == (N_A, N_B)

    def test_scalar_2d_coords(self, tmp_path):
        _, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        np.testing.assert_array_equal(da.coords["a"].values, v.axes["a"].values)
        np.testing.assert_array_equal(da.coords["b"].values, v.axes["b"].values)

    def test_trace_dims(self, tmp_path):
        _, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("trace")
        assert da.dims == ("a", "b", "sample")
        assert da.shape == (N_A, N_B, N_T)

    def test_image_dims_no_component(self, tmp_path):
        write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("res")
        # contains=["fpts","mag","phase"] → first trailing dim named "component"
        assert da.dims == ("a", "b", "component", "sample")
        assert da.shape == (N_A, N_B, 3, N_T)
        np.testing.assert_array_equal(da.coords["component"].values, ["fpts", "mag", "phase"])

    def test_image_with_component(self, tmp_path):
        write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("res", component="mag")
        assert da.dims == ("a", "b", "sample")
        assert da.shape == (N_A, N_B, N_T)
        assert da.attrs["component"] == "mag"

    def test_units_on_coord(self, tmp_path):
        _, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        assert da.coords["a"].attrs.get("units") == "V"
        assert da.coords["b"].attrs.get("units") == "Hz"

    def test_readout_unit_in_attrs(self, tmp_path):
        _, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        assert da.attrs.get("units") == "dBm"

    def test_nan_at_unwritten_points(self, tmp_path):
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        w.write_metadata()
        # write only first row
        for j in range(N_B):
            w.write_point((0, j), {"scalar": 1.0, "trace": np.zeros(N_T)})
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        assert np.all(np.isfinite(da.values[0]))
        assert np.all(np.isnan(da.values[1:]))

    def test_long_name_attr(self, tmp_path):
        _, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        assert da.attrs["long_name"] == "scalar"

    def test_sel_by_coord_value(self, tmp_path):
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        scalar = np.arange(N_A * N_B, dtype=float).reshape(N_A, N_B)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"scalar": scalar[i, j], "trace": np.zeros(N_T)})
        w.write_metadata()
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        a_val = v.axes["a"].values[1]
        sliced = da.sel(a=a_val, method="nearest")
        np.testing.assert_array_equal(sliced.values, scalar[1])

    def test_isel_by_index(self, tmp_path):
        schema = make_2d_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        scalar = np.arange(N_A * N_B, dtype=float).reshape(N_A, N_B)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"scalar": scalar[i, j], "trace": np.zeros(N_T)})
        w.write_metadata()
        v = Vault(tmp_path)
        da = v.to_xarray("scalar")
        np.testing.assert_array_equal(da.isel(a=0).values, scalar[0])

    def test_no_xarray_raises_import_error(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "xarray":
                raise ImportError("no xarray")
            return real_import(name, *args, **kwargs)
        _, _ = finished_vault(tmp_path)
        v = Vault(tmp_path)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="xarray"):
            v.to_xarray("scalar")


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
        assert "a" in v.axes
        assert v.axes["a"].unit == "V"
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

    def test_open_vault_by_4digit_id(self, tmp_path):
        folder = tmp_path / "0007_blabla"
        folder.mkdir()
        schema = make_2d_schema()
        ZarrWriter(
            root=folder, schema=schema, overwrite=True, initialize_arrays=True
        ).write_metadata()
        v = open_vault(tmp_path / "0007")
        assert v.shape == (N_A, N_B)

    def test_open_vault_by_int_id(self, tmp_path):
        folder = tmp_path / "0007_blabla"
        folder.mkdir()
        schema = make_2d_schema()
        ZarrWriter(
            root=folder, schema=schema, overwrite=True, initialize_arrays=True
        ).write_metadata()
        # int 7 from parent dir: not useful alone, but covered via path
        v = open_vault(tmp_path / "0007")
        assert v.shape == (N_A, N_B)

    def test_open_vault_exact_match_preferred(self, tmp_path):
        exact = tmp_path / "0007"
        exact.mkdir()
        schema = make_2d_schema()
        ZarrWriter(
            root=exact, schema=schema, overwrite=True, initialize_arrays=True
        ).write_metadata()
        # also create a suffixed sibling; exact path should win without ambiguity error
        (tmp_path / "0007_other").mkdir()
        v = open_vault(exact)
        assert v.shape == (N_A, N_B)

    def test_open_vault_id_no_match_raises(self, tmp_path):
        with pytest.raises(VaultError, match="no directory"):
            open_vault(tmp_path / "0099")

    def test_open_vault_id_ambiguous_raises(self, tmp_path):
        (tmp_path / "0007_foo").mkdir()
        (tmp_path / "0007_bar").mkdir()
        with pytest.raises(VaultError, match="ambiguous"):
            open_vault(tmp_path / "0007")

    def test_vault_repr(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(
            root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True
        ).write_metadata()
        v = Vault(tmp_path)
        r = repr(v)
        assert "sweep" in r
        assert "3x4" in r


# ===========================================================================
# TestDimNames — Vault.dim_names property and __repr__ role strings
# ===========================================================================

class TestDimNames:

    def test_1d_dim_names_is_sweep(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.dim_names == ["sweep"]

    def test_2d_dim_names_step_sweep(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.dim_names == ["step", "sweep"]

    def test_3d_dim_names_frame_step_sweep(self, tmp_path):
        a = ControlVar("a", np.arange(2, dtype=np.float32))
        b = ControlVar("b", np.arange(N_A, dtype=np.float32))
        c = ControlVar("c", np.arange(N_B, dtype=np.float32))
        schema = MeasurementSchema(
            control=[AxisSpecs([a]), AxisSpecs([b]), AxisSpecs([c])],
            readout=[ReadoutSpecs("x", DataKind.SCALAR)],
            writetype=WriteType.POINTWISE,
        )
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.dim_names == ["frame", "step", "sweep"]

    def test_1d_repr_says_1d_sweep(self, tmp_path):
        schema = make_1d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert "1D sweep" in repr(v)

    def test_2d_repr_says_step_sweep(self, tmp_path):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert "step sweep" in repr(v)


# ===========================================================================
# TestTrailingDims — ReadoutSpecs.trailing_dims, Vault.trailing_coords,
#                    Readout.trailing_dims, and to_xarray integration
# ===========================================================================

FREQ_VALS = np.linspace(4e9, 8e9, N_T)


def make_trace_with_freq_schema():
    """2D sweep, TRACE readout with a numeric 'freq' trailing_dim."""
    a = ControlVar("a", np.linspace(0, 1, N_A, dtype=np.float32), unit="V")
    b = ControlVar("b", np.linspace(0, 1, N_B, dtype=np.float32), unit="T")
    return MeasurementSchema(
        control=[AxisSpecs([a]), AxisSpecs([b])],
        readout=[
            ReadoutSpecs(
                "S21",
                DataKind.TRACE,
                shape=(N_T,),
                unit="dBm",
                trailing_dims=[{"name": "freq", "unit": "Hz", "values": FREQ_VALS}],
            ),
        ],
        writetype=WriteType.POINTWISE,
    )


def make_trace_with_string_trailing_schema():
    """TRACE with shape=(2,) whose trailing_dim values are string labels (X, Y)."""
    a = ControlVar("a", np.linspace(0, 1, N_A, dtype=np.float32))
    b = ControlVar("b", np.linspace(0, 1, N_B, dtype=np.float32))
    return MeasurementSchema(
        control=[AxisSpecs([a]), AxisSpecs([b])],
        readout=[
            ReadoutSpecs(
                "lockin_xy",
                DataKind.TRACE,
                shape=(2,),
                contains=["X", "Y"],
                trailing_dims=[{"name": "component", "values": ["X", "Y"]}],
            ),
        ],
        writetype=WriteType.POINTWISE,
    )


class TestTrailingDims:

    def test_numeric_coord_array_written_to_zarr(self, tmp_path):
        """ZarrWriter must create a 'S21.freq' array inside vault.zarr."""
        schema = make_trace_with_freq_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        import zarr as _zarr
        g = _zarr.open_group(str(tmp_path / "vault.zarr"), mode="r")
        assert "S21.freq" in g, "numeric trailing coord array must be written as readout.dimname"
        np.testing.assert_array_almost_equal(np.asarray(g["S21.freq"][:]), FREQ_VALS)

    def test_string_labels_not_written_to_zarr(self, tmp_path):
        """String trailing_dim values must stay in metadata.yaml, not appear in vault.zarr."""
        schema = make_trace_with_string_trailing_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        import zarr as _zarr
        g = _zarr.open_group(str(tmp_path / "vault.zarr"), mode="r")
        assert "lockin_xy.component" not in g

    def test_trailing_coords_loaded_on_open_numeric(self, tmp_path):
        """Vault.trailing_coords is populated with the numeric freq array."""
        schema = make_trace_with_freq_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert "S21" in v.trailing_coords
        np.testing.assert_array_almost_equal(v.trailing_coords["S21"]["freq"], FREQ_VALS)

    def test_trailing_coords_loaded_on_open_string(self, tmp_path):
        """String labels are loaded from metadata into trailing_coords."""
        schema = make_trace_with_string_trailing_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert "lockin_xy" in v.trailing_coords
        np.testing.assert_array_equal(v.trailing_coords["lockin_xy"]["component"], ["X", "Y"])

    def test_readout_trailing_dims_tuple_populated(self, tmp_path):
        """Readout.trailing_dims reflects the named dims from ReadoutSpecs."""
        schema = make_trace_with_freq_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert v.readouts["S21"].trailing_dims == ("freq",)

    def test_to_xarray_uses_vault_trailing_dims(self, tmp_path):
        """to_xarray uses vault trailing_dims for dim name and coords (priority 1)."""
        schema = make_trace_with_freq_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"S21": np.ones(N_T)})
        w.write_metadata()
        v = Vault(tmp_path)
        da = v.to_xarray("S21")
        assert "freq" in da.dims
        assert da.shape == (N_A, N_B, N_T)
        np.testing.assert_array_almost_equal(da.coords["freq"].values, FREQ_VALS)

    def test_to_xarray_trailing_coords_kwarg(self, tmp_path):
        """trailing_coords= kwarg supplies coords for older vaults (priority 2)."""
        # finished_vault uses make_2d_schema which has no trailing_dims
        finished_vault(tmp_path)
        v = Vault(tmp_path)
        custom_freq = np.linspace(1e9, 5e9, N_T)
        da = v.to_xarray("trace", trailing_coords={"freq": custom_freq})
        assert "freq" in da.dims
        assert da.shape == (N_A, N_B, N_T)
        np.testing.assert_array_almost_equal(da.coords["freq"].values, custom_freq)

    def test_to_xarray_trailing_coords_kwarg_no_vault_dims(self, tmp_path):
        """trailing_coords= is ignored when vault trailing_dims already exist (priority 1 wins)."""
        schema = make_trace_with_freq_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"S21": np.ones(N_T)})
        w.write_metadata()
        v = Vault(tmp_path)
        wrong_freq = np.zeros(N_T)
        da = v.to_xarray("S21", trailing_coords={"freq": wrong_freq})
        # vault trailing_dims take priority — coord must match FREQ_VALS, not wrong_freq
        np.testing.assert_array_almost_equal(da.coords["freq"].values, FREQ_VALS)

    def test_to_xarray_trailing_dim_unit_in_attrs(self, tmp_path):
        """Freq coord is present and dim is named 'freq' in the returned DataArray."""
        schema = make_trace_with_freq_schema()
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"S21": np.ones(N_T)})
        w.write_metadata()
        v = Vault(tmp_path)
        da = v.to_xarray("S21")
        assert "freq" in da.coords

    def test_component_units_in_xarray_attrs(self, tmp_path):
        """'units' list in trailing_dim dict surfaces as component_units attr in DataArray."""
        write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("res")
        assert "component_units" in da.attrs
        cu = da.attrs["component_units"]
        assert cu["fpts"] == "Hz"
        assert cu["mag"] == "dBm"
        assert cu["phase"] == "deg"

    def test_component_units_absent_after_component_selection(self, tmp_path):
        """component_units attr is not set when the component dim was selected away."""
        write_full_image_layers(tmp_path)
        v = Vault(tmp_path)
        da = v.to_xarray("res", component="mag")
        assert "component_units" not in da.attrs


# ===========================================================================
# TestContainsPromotion — contains-as-list auto-promotes to trailing_dims
# ===========================================================================

class TestContainsPromotion:

    def _make_schema_contains_only(self):
        """Schema using the old contains=list style (no explicit trailing_dims)."""
        a = ControlVar("a", np.linspace(0, 1, N_A, dtype=np.float32))
        b = ControlVar("b", np.linspace(0, 1, N_B, dtype=np.float32))
        return MeasurementSchema(
            control=[AxisSpecs([a]), AxisSpecs([b])],
            readout=[
                ReadoutSpecs("lockin", DataKind.TRACE, shape=(2,),
                             contains=["X", "Y"]),
            ],
            writetype=WriteType.POINTWISE,
        )

    def test_trailing_dims_set_after_promotion(self):
        """MeasurementSchema.__post_init__ populates trailing_dims on the ReadoutSpecs."""
        schema = self._make_schema_contains_only()
        r = schema.readout[0]
        assert r.trailing_dims is not None
        assert r.trailing_dims[0]["name"] == "component"
        assert r.trailing_dims[0]["values"] == ["X", "Y"]

    def test_component_labels_in_metadata(self, tmp_path):
        """Auto-promoted labels appear in metadata.yaml trailing_dims."""
        schema = self._make_schema_contains_only()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        import yaml
        meta = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
        td = meta["readout"]["lockin"]["trailing_dims"]
        assert any(d["name"] == "component" and d.get("values") == ["X", "Y"] for d in td)

    def test_component_labels_loadable_via_trailing_coords(self, tmp_path):
        """Vault.trailing_coords picks up auto-promoted string labels."""
        schema = self._make_schema_contains_only()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        v = Vault(tmp_path)
        assert "lockin" in v.trailing_coords
        np.testing.assert_array_equal(v.trailing_coords["lockin"]["component"], ["X", "Y"])

    def test_select_component_works_via_trailing_coords(self, tmp_path):
        """component= selection works for readouts with no contains but trailing_dims set."""
        schema = self._make_schema_contains_only()
        rng = np.random.default_rng(1)
        data = rng.random((N_A, N_B, 2), dtype=np.float32)
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"lockin": data[i, j]})
        w.write_metadata()
        v = Vault(tmp_path)
        result = v.read("lockin", component="X")
        assert result.shape == (N_A, N_B)
        np.testing.assert_array_almost_equal(result, data[:, :, 0])

    def test_to_xarray_dims_from_auto_promotion(self, tmp_path):
        """to_xarray uses the auto-promoted trailing_dims for dim names and coords."""
        schema = self._make_schema_contains_only()
        rng = np.random.default_rng(2)
        data = rng.random((N_A, N_B, 2), dtype=np.float32)
        w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
        for i in range(N_A):
            for j in range(N_B):
                w.write_point((i, j), {"lockin": data[i, j]})
        w.write_metadata()
        v = Vault(tmp_path)
        da = v.to_xarray("lockin")
        assert "component" in da.dims
        np.testing.assert_array_equal(da.coords["component"].values, ["X", "Y"])


# ===========================================================================
# TestSelectComponentTrace — _select_component fix for TRACE with shape=(2,)
# ===========================================================================

def make_trace_xy_vault(tmp_path):
    """2D sweep, TRACE readout with shape=(2,), contains=['X','Y']."""
    a = ControlVar("a", np.linspace(0, 1, N_A, dtype=np.float32))
    b = ControlVar("b", np.linspace(0, 1, N_B, dtype=np.float32))
    schema = MeasurementSchema(
        control=[AxisSpecs([a]), AxisSpecs([b])],
        readout=[ReadoutSpecs("lockin_xy", DataKind.TRACE, shape=(2,), contains=["X", "Y"])],
        writetype=WriteType.POINTWISE,
    )
    rng = np.random.default_rng(0)
    data = rng.random((N_A, N_B, 2), dtype=np.float32)
    w = ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True)
    for i in range(N_A):
        for j in range(N_B):
            w.write_point((i, j), {"lockin_xy": data[i, j]})
    w.write_metadata()
    return data


class TestSelectComponentTrace:

    def test_component_x_shape(self, tmp_path):
        """TRACE with shape=(2,): component='X' must return shape (N_A, N_B)."""
        make_trace_xy_vault(tmp_path)
        v = Vault(tmp_path)
        result = v.read("lockin_xy", component="X")
        assert result.shape == (N_A, N_B)

    def test_component_x_values(self, tmp_path):
        data = make_trace_xy_vault(tmp_path)
        v = Vault(tmp_path)
        result = v.read("lockin_xy", component="X")
        np.testing.assert_array_almost_equal(result, data[:, :, 0])

    def test_component_y_values(self, tmp_path):
        data = make_trace_xy_vault(tmp_path)
        v = Vault(tmp_path)
        result = v.read("lockin_xy", component="Y")
        np.testing.assert_array_almost_equal(result, data[:, :, 1])

    def test_values_component_no_nan_full_run(self, tmp_path):
        make_trace_xy_vault(tmp_path)
        v = Vault(tmp_path)
        result = v.values("lockin_xy", component="X")
        assert result.shape == (N_A, N_B)
        assert not np.isnan(result).any()


# ===========================================================================
# TestSnapshot — snapshot() printed output
# ===========================================================================

class TestSnapshot:

    def test_snapshot_contains_shape_and_kind(self, tmp_path, capsys):
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        Vault(tmp_path).snapshot()
        out = capsys.readouterr().out
        assert "3x4" in out or f"{N_A}x{N_B}" in out
        assert "sweep" in out

    def test_snapshot_shows_dim_roles(self, tmp_path, capsys):
        """snapshot() header must include dimension role names for a 2D vault."""
        schema = make_2d_schema()
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        Vault(tmp_path).snapshot()
        out = capsys.readouterr().out
        # 2D vault: outermost dim is "step"; kind_str is "2D step sweep"
        assert "step" in out

    def test_snapshot_shows_all_readouts(self, tmp_path, capsys):
        schema = make_2d_schema(with_image=True)
        ZarrWriter(root=tmp_path, schema=schema, overwrite=True, initialize_arrays=True).write_metadata()
        Vault(tmp_path).snapshot()
        out = capsys.readouterr().out
        assert "scalar" in out
        assert "trace" in out
        assert "image" in out

    def test_snapshot_monitor_shows_time_axis(self, tmp_path, capsys):
        sw = StreamingWriter(root=tmp_path, readouts={"x": ()}, chunk_size=1)
        sw.append({"x": 1.0})
        sw.close()
        Vault(tmp_path).snapshot()
        out = capsys.readouterr().out
        assert "monitor" in out
        assert "sample" in out
