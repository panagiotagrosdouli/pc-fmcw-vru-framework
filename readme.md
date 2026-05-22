# PC-FMCW VRU Framework

Advanced Python framework for reproducing and extending the paper:

> *“Pedestrian-Cyclist FMCW Adaptive Illumination and Sensing Framework”*

using:

* FMCW sensing,
* VRU trajectory prediction,
* adaptive beam scheduling,
* uncertainty-aware illumination,
* and the Waymo Open Motion Dataset (WOMD).

---

# Overview

This project contains two major parts:

| Part   | Description                                                            |
| ------ | ---------------------------------------------------------------------- |
| Part A | PC-FMCW sensing / communication / tracking / illumination reproduction |
| Part B | Advanced VRU prediction and adaptive beam intelligence using WOMD      |

The framework is implemented entirely in Python.

---

# Main Features

## Part A — PC-FMCW Framework

Implemented modules:

* FMCW waveform simulation
* Beat spectrum analysis
* DPSK communication link
* Range-Doppler sensing
* CFAR target detection
* Multi-target tracking
* Adaptive beam illumination
* Performance evaluation

The framework was calibrated to reproduce paper-level metrics:

| Metric                | Achieved |
| --------------------- | -------- |
| Range resolution      | 1.5 cm   |
| Ranging RMSE @ 20 dB  | 3.82 cm  |
| Multi-target tracking | enabled  |
| Adaptive illumination | enabled  |

---

# Part B — WOMD-Based VRU Intelligence

The second part extends the original paper using real-world autonomous driving trajectories from:

## Waymo Open Motion Dataset (WOMD)

Implemented capabilities:

---

## 1. VRU Trajectory Prediction

Prediction horizons:

* 1 s
* 2 s
* 3 s
* 5 s
* 8 s

Models:

* Constant Velocity (CV)
* Kalman Filter (KF)
* Hybrid CV/KF predictor

Metrics:

* ADE
* FDE
* uncertainty growth

---

## 2. Predictive Adaptive Beam Scheduling

The framework predicts:

* when a VRU enters the beam zone,
* proactive beam scheduling,
* future shadow generation.

Metrics:

* beam timing error,
* proactive scheduling gain,
* prediction uncertainty.

---

## 3. Intention Prediction

Behavior-aware VRU classification:

* crossing
* turning
* straight
* stationary
* erratic

Motion features:

* heading change,
* lateral displacement,
* curvature,
* crossing detection.

---

## 4. Risk-Aware Beam Control

The system estimates:

* VRU risk score,
* uncertainty-aware beam expansion,
* dynamic illumination prioritization.

---

## 5. Near-Collision Forecasting

Forecasts:

* minimum future distance,
* time-to-nearest approach,
* collision risk.

This enables:

* proactive safety-oriented illumination.

---

## 6. Occlusion-Aware Prediction

Simulates:

* temporary VRU disappearance,
* sensor dropout,
* partial occlusion.

The framework evaluates:

* prediction drift,
* uncertainty explosion,
* adaptive beam widening.

---

## 7. Future Occupancy Heatmaps

Probabilistic future occupancy estimation:

* future VRU probability fields,
* occupancy-aware beam shaping,
* uncertainty-based illumination regions.

---

# Project Structure

```text
pc_fmcw_project_python/
│
├── part_a/
│   ├── waveform.py
│   ├── communication.py
│   ├── sensing.py
│   ├── tracking.py
│   ├── illumination.py
│   └── performance.py
│
├── part_b/
│   ├── womd_prediction.py
│   ├── womd_beam_eval.py
│   ├── womd_future_forecast.py
│   ├── womd_advanced_predictions.py
│   ├── risk_aware_beam.py
│   ├── intention_prediction.py
│   ├── occlusion_aware_prediction.py
│   ├── future_occupancy_heatmap.py
│   ├── near_collision_forecasting.py
│   └── hybrid_predictive_scheduler.py
│
├── womd/
│   ├── womd_loader.py
│   └── tfrecord_io.py
│
├── figures/
├── outputs/
├── data/
└── run_part_a.py
```

---

# Dataset

The framework uses:

## Waymo Open Motion Dataset (WOMD)

Validation subset downloaded from:

* Google Cloud Storage

Used data:

* pedestrian trajectories,
* cyclist trajectories,
* real-world urban interactions.

---

# Installation

## Ubuntu / WSL

Create environment:

```bash
python3 -m venv venv_womd
source venv_womd/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Part A

```bash
python run_part_a.py
```

Outputs:

* waveform figures,
* tracking figures,
* adaptive illumination figures,
* JSON metrics.

---

# Running Part B

## Basic WOMD experiments

```bash
python -m part_b.womd_experiments
```

---

## Future trajectory forecasting

```bash
python -m part_b.womd_future_forecast
```

---

## Advanced prediction analysis

```bash
python -m part_b.womd_advanced_predictions
```

---

## Hybrid predictive beam scheduling

```bash
python -m part_b.hybrid_predictive_scheduler
```

---

## Intention prediction

```bash
python -m part_b.intention_prediction
```

---

## Occlusion-aware prediction

```bash
python -m part_b.occlusion_aware_prediction
```

---

## Future occupancy heatmaps

```bash
python -m part_b.future_occupancy_heatmap
```

---

## Near-collision forecasting

```bash
python -m part_b.near_collision_forecasting
```

---

# Example Results

## Part A

| Metric                | Result     |
| --------------------- | ---------- |
| Range resolution      | 1.5 cm     |
| RMSE @ 20 dB          | 3.82 cm    |
| Multi-target tracking | successful |
| Adaptive beam shaping | successful |

---

## Part B

### 3-second forecasting

| Metric | Value  |
| ------ | ------ |
| CV ADE | 0.36 m |
| CV FDE | 0.80 m |
| KF ADE | 0.44 m |
| KF FDE | 0.90 m |

### 5-second forecasting

| Metric | Value  |
| ------ | ------ |
| CV ADE | 0.68 m |
| CV FDE | 1.52 m |
| KF ADE | 0.76 m |
| KF FDE | 1.61 m |

---

# Research Contributions

This project extends the original FMCW illumination framework with:

* trajectory-aware adaptive illumination,
* uncertainty-aware beam control,
* intention-aware beam scheduling,
* probabilistic occupancy prediction,
* occlusion-aware VRU forecasting,
* predictive safety-oriented illumination.

---

# Future Extensions

Potential future work:

* deep learning trajectory models,
* graph neural interaction prediction,
* multi-agent forecasting,
* reinforcement learning beam control,
* real-time GPU acceleration,
* CARLA simulator integration.

---

# Authors


Department of Electrical and Computer Engineering
Democritus University of Thrace

---

# License

Academic / research use only.

