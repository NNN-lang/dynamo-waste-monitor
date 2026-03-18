# dynamo-waste-monitor

Real-time idle GPU detection and cost analysis for NVIDIA Dynamo 1.0

The first open-source cost-aware monitoring plugin for NVIDIA Dynamo 1.0.
Detects idle GPUs in real time and outputs Dynamo-compatible rebalancing commands.


The Problem
Companies running NVIDIA GPU clusters waste $1,500–$5,000/month per node
because inference workloads leave 40–60% of GPUs idle between requests.
NVIDIA Dynamo 1.0 (released March 2026) has no built-in cost monitoring.
No tool exists that tells you: "These 4 GPUs are idle. That's $159/hr wasted."
This tool fills that gap.

What It Does
========================================================================
  NVIDIA Dynamo GPU Waste Monitor  |  2026-03-18 09:35:18
========================================================================

  Cluster: 8 GPUs  |  Avg util: 41.2%  |  Idle: 4  |  Waste: $159.20/hr  ($114,624/mo)

  GPU  Model            Util%    Mem%    Temp   W       $/hr    Status
  ────────────────────────────────────────────────────────────────────
  0    H200 SXM5        99.4     80.3    79     669     39.80   OK
  1    H200 SXM5        81.1     76.1    72     559     39.80   OK
  2    H200 SXM5        64.8     58.0    69     442     39.80   OK
  3    H200 SXM5        52.9     49.9    67     354     39.80   LOW
  4    H200 SXM5         4.2     14.6    47     120     39.80   IDLE
  5    H200 SXM5         0.0     19.2    44      87     39.80   IDLE
  6    H200 SXM5        14.6     27.7    48     192     39.80   IDLE
  7    H200 SXM5        12.4     27.5    49     142     39.80   IDLE

  RECOMMENDATIONS:
    ⚡ 4 GPU(s) fully idle ($159.20/hr = $114,624/mo wasted).
    ⚠  1 GPU(s) underutilised (<60%). Consider consolidating requests.
    📉 Cluster avg below 50%. Consider right-sizing.

Quick Start
bash# Install
pip install -r requirements.txt

# Try immediately (no GPU required)
python dynamo_waste_monitor.py --simulate --once

# Live dashboard
python dynamo_waste_monitor.py --simulate --dashboard

# Real GPU mode
python dynamo_waste_monitor.py --once

# Connect to Dynamo 1.0
python dynamo_waste_monitor.py --dynamo-endpoint http://localhost:8080 --dashboard

Dynamo 1.0 Integration
The monitor reads NVIDIA Dynamo 1.0 worker API and outputs
rebalancing commands directly consumable by Dynamo:
yamlapi_version: dynamo/v1
kind: RebalanceSchedule
metadata:
  cluster_utilisation_pct: 41.2
  idle_gpus: 4
  waste_per_hour_usd: 159.20
  waste_per_month_usd: 114624.0
spec:
  commands:
    - action: migrate_worker
      worker_id: worker-llama405b-01
      from_gpu: 0
      to_gpu: 4
      model: llama3-405b
      estimated_gain: 39.80_usd_per_hour
Apply rebalancing schedule:
bashpython dynamo_waste_monitor.py --output-yaml rebalance.yaml
curl -X POST http://localhost:8080/v1/workers/rebalance \
     -H "Content-Type: application/json" \
     -d @rebalance.yaml

GPU Pricing
Built-in pricing table (March 2026, AWS / CoreWeave):
GPU$/hrH200 SXM5$39.80H100 SXM5$28.50B200$52.00A100$18.40L40S$6.50

Requirements

Python 3.10+
NVIDIA GPU with drivers (optional — simulation mode works without)
NVIDIA Dynamo 1.0 (optional — standalone mode works without)


License
Apache License 2.0 — see [LICENSE](LICENSE)
Patent pending. For commercial licensing open an issue. open an [issue](https://github.com/NNN-lang/dynamo-waste-monitor/issues)
