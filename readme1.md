````markdown
# PC-FMCW VRU Framework
## Predictive Adaptive Illumination and VRU Trajectory Forecasting using FMCW Radar and WOMD

Research-oriented framework for:
- PC-FMCW radar sensing
- adaptive driving beam (ADB) control
- VRU trajectory prediction
- predictive illumination
- collision-risk forecasting
- uncertainty-aware beam shaping
- Waymo Open Motion Dataset (WOMD) experimentation

---

# Overview

This project extends the original PC-FMCW radar + adaptive headlamp paper with:

- real WOMD trajectory forecasting
- predictive VRU-aware beam steering
- probabilistic collision forecasting
- adaptive prediction horizons
- risk-aware illumination
- uncertainty-aware beam expansion
- social interaction analysis
- occlusion-aware prediction
- anomaly detection
- future safety envelopes
- multi-agent shadow allocation
- risk heatmap forecasting

The framework combines:
- radar sensing
- communication
- tracking
- prediction
- safety modeling
- intelligent beam control

into a unified research pipeline.

---

# Repository Structure

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
│   ├── prediction.py
│   ├── womd_prediction.py
│   ├── womd_beam_eval.py
│   ├── risk_aware_beam.py
│   ├── womd_future_forecast.py
│   ├── iape_metric.py
│   ├── vru_priority_queue.py
│   ├── energy_aware_beam.py
│   ├── multi_vru_shadow_allocation.py
│   ├── failure_case_analysis.py
│   ├── scenario_difficulty_score.py
│   ├── adaptive_prediction_horizon.py
│   ├── beam_smoothness.py
│   ├── safety_envelope_prediction.py
│   ├── collision_probability_forecasting.py
│   ├── hybrid_predictor_selector.py
│   ├── risk_heatmap_forecasting.py
│   ├── multi_agent_interaction.py
│   ├── dynamic_beam_width.py
│   ├── occlusion_aware_prediction.py
│   └── trajectory_anomaly_detection.py
│
├── womd/
│   └── womd_loader.py
│
├── outputs/
├── figures/
└── README.md
````

---

# Part A — PC-FMCW Radar Reproduction

## Implemented Modules

### FMCW Waveform and Ranging

* FMCW chirp generation
* beat signal extraction
* FFT-based range estimation
* phase analysis

### Communication Layer

* DPSK communication simulation
* BER evaluation
* communication-performance curves

### Radar Sensing

* range-Doppler processing
* CFAR detection
* multi-target detection

### Multi-Target Tracking

* clutter simulation
* clutter suppression
* MHT-inspired tracking pipeline

### Adaptive Illumination

* shadow zone generation
* adaptive beam shaping
* multi-object illumination suppression

### Performance Evaluation

* centimeter-level ranging analysis
* CRLB-inspired comparison
* paper-oriented calibration

---

# Part A Main Results

| Metric                   | Achieved |
| ------------------------ | -------- |
| Range resolution         | 1.5 cm   |
| Ranging RMSE @20 dB      | ~3.82 cm |
| Targets detected         | 2        |
| Tracking clutter removed | 159      |
| Mean tracking deviation  | ~1.64    |
| Adaptive beam shadows    | 2        |

---

# Part B — WOMD Predictive Framework

## Dataset

Waymo Open Motion Dataset (WOMD):

* real pedestrian trajectories
* cyclist motion
* urban driving scenarios
* multi-agent interactions

Used for:

* prediction
* collision forecasting
* beam steering
* adaptive illumination evaluation

---

# Implemented Research Extensions

---

## 1. WOMD Trajectory Forecasting

Implemented:

* Constant Velocity (CV)
* Kalman Filter (KF)

Metrics:

* ADE
* FDE

Forecast horizons:

* 3 seconds
* 5 seconds

---

## 2. Predictive Beam Evaluation

Prediction-aware adaptive illumination:

* reactive beam
* predictive beam
* angular prediction error
* glare reduction analysis

---

## 3. Risk-Aware Beam Steering

Introduced:

* VRU risk scoring
* uncertainty-aware illumination
* dynamic beam prioritization

Metrics:

* risk score
* uncertainty level
* top-risk VRUs

---

## 4. Future VRU Forecasting

Future trajectory prediction:

* 3-second prediction
* 5-second prediction

Metrics:

* ADE
* FDE
* prediction stability

---

## 5. Illumination-Aware Prediction Error (IAPE)

Novel metric introduced:

```text
IAPE =
trajectory error
+ angular beam error
+ uncertainty penalty
+ timing penalty
```

Purpose:

* evaluate prediction quality for adaptive illumination systems

---

## 6. VRU Priority Queue

Priority-based illumination scheduling:

* distance
* uncertainty
* crossing intent
* speed
* frontal angle

Outputs:

* priority scores
* ranked VRU queue

---

## 7. Energy-Aware Beam Allocation

Adaptive beam power control:

* limited beam budget
* risk-aware allocation
* safety-weighted optimization

Outputs:

* power allocation curves
* beam efficiency gains

---

## 8. Multi-VRU Shadow Allocation

Prediction-aware shadow generation:

* multiple VRUs
* merged shadow zones
* uncertainty-dependent shadow width

Outputs:

* shadow intervals
* merged beam zones

---

## 9. Failure Case Analysis

Analyzed:

* prediction failures
* turning maneuvers
* crossing failures
* uncertainty-driven failures

Outputs:

* worst-case scenarios
* failure distributions

---

## 10. Scenario Difficulty Scoring

Automatically ranks WOMD scenes by:

* trajectory complexity
* prediction error
* interaction density
* uncertainty
* crossing behavior

Outputs:

* difficult scenario ranking

---

## 11. Adaptive Prediction Horizon

Dynamic forecast horizon selection:

* stationary
* smooth motion
* crossing motion
* high-dynamics motion

Adaptive horizons:

* 2s
* 3s
* 4s
* 5s

---

## 12. Beam Smoothness Optimization

Anti-flicker adaptive illumination:

* beam steering smoothing
* jitter suppression
* actuator-friendly steering

Outputs:

* jitter reduction
* smoothed beam trajectories

---

## 13. Safety Envelope Prediction

Probabilistic future occupancy prediction:

* uncertainty ellipses
* future safety regions
* occupancy coverage estimation

Outputs:

* future safety envelopes

---

## 14. Collision Probability Forecasting

Probabilistic collision prediction:

* collision likelihood
* TTC estimation
* uncertainty-aware danger prediction

Outputs:

* collision probability curves
* TTC analysis

---

## 15. Hybrid Predictor Selection

Adaptive predictor routing:

* CV selection
* KF selection
* motion-aware switching

Outputs:

* hybrid prediction gains

---

## 16. Risk Heatmap Forecasting

Generated:

* future danger maps
* occupancy heatmaps
* high-risk zones

Outputs:

* probabilistic risk maps

---

## 17. Multi-Agent Interaction Modeling

Social interaction analysis:

* pairwise interactions
* motion coupling
* group behavior

Outputs:

* interaction scores
* proximity analysis

---

## 18. Dynamic Beam Width Optimization

Adaptive beam geometry:

* uncertainty-aware widening
* interaction-aware shaping
* risk-based beam expansion

Outputs:

* beam width evolution

---

## 19. Occlusion-Aware Prediction

Visibility-aware prediction:

* hidden VRUs
* occlusion probability
* uncertainty inflation

Outputs:

* occlusion-aware safety margins

---

## 20. Trajectory Anomaly Detection

Detected:

* erratic motion
* abnormal acceleration
* sudden turning
* prediction residual anomalies

Outputs:

* anomaly scores
* high-risk abnormal trajectories

---

# Figures

Generated figures include:

* FMCW waveform analysis
* beat spectrum
* BER curves
* range-Doppler maps
* MHT tracking
* adaptive beam patterns
* WOMD trajectory prediction
* risk heatmaps
* collision forecasting
* safety envelopes
* anomaly distributions
* interaction analysis
* dynamic beam control

All figures are automatically saved in:

```text
figures/
```

---

# Outputs

All metrics/results are exported to:

```text
outputs/
```

Formats:

* JSON
* CSV
* NumPy arrays

---

# Installation

## Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Dependencies

```bash
pip install -r requirements.txt
```

---

# WOMD Dataset Setup

## Install Google Cloud SDK

```bash
gcloud auth login
```

## Download WOMD Validation Set

```bash
gsutil -m cp -r \
gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/validation \
data/womd/
```

---

# Running Experiments

## Part A

```bash
python run_part_a.py
```

## WOMD Forecasting

```bash
python -m part_b.womd_future_forecast
```

## Risk Heatmap

```bash
python -m part_b.risk_heatmap_forecasting
```

## Collision Forecasting

```bash
python -m part_b.collision_probability_forecasting
```

## Occlusion-Aware Prediction

```bash
python -m part_b.occlusion_aware_prediction
```

---

# Research Contributions

This framework extends traditional PC-FMCW + adaptive beam systems with:

* predictive illumination
* uncertainty-aware forecasting
* VRU interaction analysis
* collision-risk forecasting
* adaptive prediction control
* intelligent beam scheduling
* safety-aware beam shaping
* probabilistic occupancy modeling

---

# Potential Future Work

* Graph neural trajectory prediction
* Transformer-based forecasting
* Real-time CUDA acceleration
* Multi-camera fusion
* Radar-camera fusion
* Reinforcement-learning beam control
* End-to-end autonomous illumination
* Real vehicle deployment

---

# Authors

Panagiota Grosdouli
Department of Electrical and Computer Engineering
Democritus University of Thrace

---

# License

MIT License

```
```
