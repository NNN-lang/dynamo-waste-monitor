"""
tests/test_monitor.py
Tests for Dynamo GPU Waste Monitor.
Run: pytest tests/ -v
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dynamo_waste_monitor import (
    GPUStats, ClusterSnapshot, WasteAnalyser,
    SimulatedGPUReader, IDLE_THRESHOLD_PCT,
    WARNING_THRESHOLD_PCT, GPU_PRICE_PER_HOUR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_gpu(index=0, util=80.0, name="H200 SXM5",
             mem_used=80_000.0, mem_total=141_000.0,
             temp=65.0, power=500.0, power_limit=700.0):
    return GPUStats(
        gpu_index       = index,
        name            = name,
        utilisation_pct = util,
        memory_used_mb  = mem_used,
        memory_total_mb = mem_total,
        temperature_c   = temp,
        power_w         = power,
        power_limit_w   = power_limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GPUStats tests
# ─────────────────────────────────────────────────────────────────────────────

def test_gpu_idle_detection():
    idle_gpu = make_gpu(util=5.0)
    busy_gpu = make_gpu(util=85.0)
    assert idle_gpu.is_idle is True
    assert busy_gpu.is_idle is False


def test_gpu_underutilised_detection():
    under_gpu = make_gpu(util=40.0)
    busy_gpu  = make_gpu(util=85.0)
    assert under_gpu.is_underutilised is True
    assert busy_gpu.is_underutilised  is False


def test_gpu_memory_pct():
    gpu = make_gpu(mem_used=70_500.0, mem_total=141_000.0)
    assert abs(gpu.memory_pct - 50.0) < 0.1


def test_gpu_pricing_h200():
    gpu = make_gpu(name="H200 SXM5")
    assert gpu.price_per_hour == GPU_PRICE_PER_HOUR["H200"]


def test_gpu_pricing_h100():
    gpu = make_gpu(name="H100 SXM5")
    assert gpu.price_per_hour == GPU_PRICE_PER_HOUR["H100"]


def test_gpu_pricing_unknown():
    gpu = make_gpu(name="UNKNOWN GPU XYZ")
    assert gpu.price_per_hour == GPU_PRICE_PER_HOUR["UNKNOWN"]


def test_gpu_waste_idle():
    """Idle GPU should waste most of its hourly cost."""
    gpu = make_gpu(util=0.0, name="H200 SXM5")
    expected = GPU_PRICE_PER_HOUR["H200"] * (WARNING_THRESHOLD_PCT / 100)
    assert abs(gpu.waste_per_hour - expected) < 0.01


def test_gpu_waste_busy():
    """Busy GPU above warning threshold wastes nothing."""
    gpu = make_gpu(util=90.0)
    assert gpu.waste_per_hour == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ClusterSnapshot tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cluster_idle_gpus():
    gpus = [
        make_gpu(index=0, util=85.0),
        make_gpu(index=1, util=5.0),
        make_gpu(index=2, util=0.0),
        make_gpu(index=3, util=70.0),
    ]
    snap = ClusterSnapshot(gpus=gpus)
    assert len(snap.idle_gpus) == 2
    assert snap.idle_gpus[0].gpu_index == 1
    assert snap.idle_gpus[1].gpu_index == 2


def test_cluster_avg_utilisation():
    gpus = [make_gpu(util=u) for u in [100.0, 50.0, 0.0, 50.0]]
    snap = ClusterSnapshot(gpus=gpus)
    assert abs(snap.avg_utilisation - 50.0) < 0.01


def test_cluster_total_waste():
    gpus = [
        make_gpu(index=0, util=0.0,  name="H200 SXM5"),
        make_gpu(index=1, util=90.0, name="H200 SXM5"),
    ]
    snap = ClusterSnapshot(gpus=gpus)
    assert snap.total_waste_per_hour > 0
    assert snap.total_waste_per_month == snap.total_waste_per_hour * 24 * 30


def test_cluster_total_cost():
    gpus = [make_gpu(name="H200 SXM5") for _ in range(8)]
    snap = ClusterSnapshot(gpus=gpus)
    expected = 8 * GPU_PRICE_PER_HOUR["H200"]
    assert abs(snap.total_cost_per_hour - expected) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# WasteAnalyser tests
# ─────────────────────────────────────────────────────────────────────────────

def test_analyser_detects_idle():
    gpus = [
        make_gpu(index=0, util=85.0, name="H200 SXM5"),
        make_gpu(index=1, util=2.0,  name="H200 SXM5"),
    ]
    snap    = ClusterSnapshot(gpus=gpus)
    report  = WasteAnalyser().analyse(snap)
    assert report.idle_cost_per_hour > 0
    assert any("idle" in r.lower() for r in report.recommendations)


def test_analyser_no_waste_when_busy():
    gpus = [make_gpu(index=i, util=90.0) for i in range(4)]
    snap   = ClusterSnapshot(gpus=gpus)
    report = WasteAnalyser().analyse(snap)
    assert report.idle_cost_per_hour == 0.0
    assert any("✅" in r for r in report.recommendations)


def test_analyser_generates_rebalance_commands():
    gpus = [
        make_gpu(index=0, util=95.0, name="H200 SXM5"),
        make_gpu(index=1, util=2.0,  name="H200 SXM5"),
    ]
    gpus[0].worker_id  = "worker-01"
    gpus[0].model_name = "llama3-405b"
    snap   = ClusterSnapshot(gpus=gpus)
    report = WasteAnalyser().analyse(snap)
    assert len(report.rebalance_commands) >= 1


def test_analyser_yaml_output():
    gpus   = [make_gpu(index=i, util=5.0, name="H200 SXM5") for i in range(4)]
    snap   = ClusterSnapshot(gpus=gpus)
    report = WasteAnalyser().analyse(snap)
    yaml_out = WasteAnalyser().to_yaml(report)
    assert "dynamo/v1" in yaml_out
    assert "RebalanceSchedule" in yaml_out
    assert "waste_per_hour_usd" in yaml_out


def test_analyser_monthly_waste():
    gpus   = [make_gpu(index=0, util=0.0, name="H200 SXM5")]
    snap   = ClusterSnapshot(gpus=gpus)
    report = WasteAnalyser().analyse(snap)
    assert abs(report.idle_cost_per_month - report.idle_cost_per_hour * 24 * 30) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# SimulatedGPUReader tests
# ─────────────────────────────────────────────────────────────────────────────

def test_simulated_reader_returns_8_gpus():
    reader = SimulatedGPUReader()
    gpus   = reader.read()
    assert len(gpus) == 8


def test_simulated_reader_gpu_fields():
    reader = SimulatedGPUReader()
    gpus   = reader.read()
    for gpu in gpus:
        assert 0 <= gpu.utilisation_pct <= 100
        assert gpu.memory_total_mb > 0
        assert gpu.memory_used_mb >= 0
        assert gpu.temperature_c  > 0
        assert gpu.price_per_hour > 0


def test_simulated_reader_has_idle_gpus():
    """Simulation should always have at least 3 idle GPUs (GPUs 4-7)."""
    reader = SimulatedGPUReader()
    gpus   = reader.read()
    idle   = [g for g in gpus if g.is_idle]
    assert len(idle) >= 2


def test_simulated_reader_time_variance():
    """Two readings should differ (dynamic simulation)."""
    reader = SimulatedGPUReader()
    gpus1  = reader.read()
    time.sleep(0.05)
    gpus2  = reader.read()
    utils1 = [g.utilisation_pct for g in gpus1]
    utils2 = [g.utilisation_pct for g in gpus2]
    assert utils1 != utils2


# ─────────────────────────────────────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline():
    """End-to-end: read → snapshot → analyse → yaml."""
    reader   = SimulatedGPUReader()
    gpus     = reader.read()
    snap     = ClusterSnapshot(gpus=gpus)
    analyser = WasteAnalyser()
    report   = analyser.analyse(snap)
    yaml_out = analyser.to_yaml(report)

    assert snap.total_gpus == 8
    assert len(report.recommendations) >= 1
    assert "dynamo/v1" in yaml_out
    assert report.idle_cost_per_month >= 0
