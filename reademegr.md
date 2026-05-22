````markdown
# PC-FMCW VRU Framework
## Προγνωστικός Adaptive Φωτισμός και Πρόβλεψη Τροχιών VRUs με FMCW Radar και WOMD

---

# Περιγραφή Έργου

Το παρόν ερευνητικό framework υλοποιεί ένα ολοκληρωμένο σύστημα:

- FMCW radar sensing
- adaptive automotive illumination
- VRU trajectory prediction
- predictive beam steering
- collision-risk forecasting
- uncertainty-aware safety modeling
- adaptive beam shaping
- probabilistic future occupancy prediction

χρησιμοποιώντας:
- Python
- NumPy
- SciPy
- Matplotlib
- Waymo Open Motion Dataset (WOMD)

Το έργο επεκτείνει την αρχική ιδέα του paper σχετικά με:
- PC-FMCW radar
- adaptive headlights
- pedestrian/cyclist protection

και τη μετατρέπει σε ένα πλήρες predictive autonomous illumination research platform.

---

# Βασικός Στόχος

Ο βασικός στόχος του project είναι:

> η πρόβλεψη της μελλοντικής κίνησης VRUs
> και η δυναμική προσαρμογή του φωτισμού οχήματος
> ώστε να αυξάνεται η ασφάλεια και να μειώνεται το glare.

Το framework συνδυάζει:
- radar sensing
- trajectory prediction
- uncertainty estimation
- collision forecasting
- intelligent beam steering

σε ένα ενιαίο pipeline.

---

# Τι είναι VRU

VRU = Vulnerable Road User

Περιλαμβάνει:
- pedestrians
- cyclists
- scooters
- road users υψηλού κινδύνου

Οι VRUs είναι ιδιαίτερα σημαντικοί:
- τη νύχτα
- σε urban environments
- σε occlusions
- σε crossing scenarios

---

# Δομή Repository

```text
pc_fmcw_project_python/
│
├── part_a/
├── part_b/
├── womd/
├── figures/
├── outputs/
└── README.md
````

---

# PART A — FMCW Radar Reproduction

Το πρώτο μέρος αναπαράγει τις βασικές λειτουργίες του paper.

---

# 1. FMCW Waveform Simulation

Υλοποιήθηκε:

* FMCW chirp generation
* beat frequency extraction
* phase analysis
* FFT-based ranging

Παράγονται:

* waveform plots
* beat spectrum
* phase visualizations

---

# 2. Radar Communication Layer

Υλοποιήθηκε:

* DPSK modulation
* BER simulation
* communication robustness evaluation

Μετρικές:

* BER vs SNR
* communication stability

---

# 3. Radar Sensing

Υλοποιήθηκε:

* range-Doppler processing
* CFAR detection
* multi-target detection

Παράγονται:

* range-Doppler maps
* detection visualizations

---

# 4. Multi-Target Tracking

Υλοποιήθηκε:

* clutter generation
* clutter suppression
* MHT-inspired tracking

Το module:

* αφαιρεί false detections
* διατηρεί valid trajectories
* αξιολογεί tracking quality

---

# 5. Adaptive Illumination

Υλοποιήθηκε:

* adaptive beam shaping
* shadow-zone generation
* glare suppression

Το σύστημα:

* προστατεύει VRUs
* περιορίζει glare
* διαμορφώνει δυναμικά το beam

---

# 6. Radar Performance Evaluation

Υλοποιήθηκε:

* ranging accuracy evaluation
* CRLB-inspired comparison
* paper-oriented calibration

Τελικό αποτέλεσμα:

* ~3.8 cm ranging RMSE

κοντά στις τιμές του paper.

---

# PART A Αποτελέσματα

| Μετρική                 | Τιμή    |
| ----------------------- | ------- |
| Range Resolution        | 1.5 cm  |
| Ranging RMSE            | ~3.8 cm |
| Targets detected        | 2       |
| Clutter removed         | 159     |
| Mean tracking deviation | ~1.64   |

---

# PART B — WOMD Predictive Framework

Το δεύτερο μέρος επεκτείνει το framework χρησιμοποιώντας:

# Waymo Open Motion Dataset (WOMD)

Το WOMD περιέχει:

* πραγματικές trajectories
* pedestrians
* cyclists
* urban traffic scenarios
* multi-agent interactions

Χρησιμοποιήθηκε για:

* prediction
* forecasting
* collision analysis
* adaptive illumination evaluation

---

# 1. WOMD Trajectory Prediction

Υλοποιήθηκαν:

* Constant Velocity predictor
* Kalman Filter predictor

Μετρικές:

* ADE
* FDE

Forecast horizons:

* 3 seconds
* 5 seconds

---

# 2. Predictive Beam Steering

Το adaptive beam:

* προβλέπει future VRU positions
* προσαρμόζει δυναμικά τη γωνία φωτισμού
* μειώνει glare

Υπολογίζονται:

* angular prediction error
* glare metrics
* beam alignment quality

---

# 3. Risk-Aware Beam Steering

Εισήχθη:

* VRU risk scoring
* uncertainty-aware illumination
* adaptive beam prioritization

Το beam:

* δίνει προτεραιότητα σε high-risk VRUs
* αυξάνει illumination safety margins

---

# 4. Future Trajectory Forecasting

Πρόβλεψη:

* future pedestrian motion
* future cyclist motion

για:

* 3 seconds
* 5 seconds

Μετρικές:

* ADE
* FDE
* prediction stability

---

# 5. Illumination-Aware Prediction Error (IAPE)

Νέα custom metric.

Συνδυάζει:

* trajectory error
* angular beam error
* uncertainty
* timing penalties

Χρησιμοποιείται για:

* evaluation predictive illumination systems

---

# 6. VRU Priority Queue

Δημιουργήθηκε:

* adaptive VRU ranking system

Το priority score βασίζεται σε:

* distance
* uncertainty
* speed
* crossing behavior
* frontal position

---

# 7. Energy-Aware Beam Allocation

Υλοποιήθηκε:

* beam power optimization
* limited illumination budget
* adaptive power scheduling

Στόχος:

* μέγιστη safety efficiency
* χαμηλότερη energy consumption

---

# 8. Multi-VRU Shadow Allocation

Το σύστημα:

* δημιουργεί multiple shadow zones
* συγχωνεύει overlapping shadows
* υπολογίζει uncertainty-aware shadow widths

---

# 9. Failure Case Analysis

Αναλύθηκαν:

* prediction failures
* crossing failures
* turning trajectories
* high-uncertainty scenarios

Παράγονται:

* worst-case rankings
* failure distributions

---

# 10. Scenario Difficulty Scoring

Το framework:

* βαθμολογεί WOMD scenarios
* εντοπίζει difficult scenes

Βασίζεται σε:

* interaction density
* trajectory complexity
* prediction instability
* uncertainty

---

# 11. Adaptive Prediction Horizon

Υλοποιήθηκε:

* dynamic forecast horizon selection

Το horizon αλλάζει ανάλογα με:

* motion smoothness
* crossing behavior
* speed
* turning intensity

Horizons:

* 2s
* 3s
* 4s
* 5s

---

# 12. Beam Smoothness Optimization

Υλοποιήθηκε:

* anti-flicker beam steering
* beam smoothing
* jitter suppression

Στόχος:

* actuator-friendly illumination
* stable beam transitions

---

# 13. Safety Envelope Prediction

Προβλέπονται:

* future occupancy envelopes
* uncertainty ellipses
* safety zones

Το σύστημα:

* μοντελοποιεί probabilistic future occupancy
* δημιουργεί safety-aware beam regions

---

# 14. Collision Probability Forecasting

Υλοποιήθηκε:

* probabilistic collision prediction
* TTC estimation
* future danger estimation

Παράγονται:

* collision probability curves
* TTC-risk relations

---

# 15. Hybrid Predictor Selector

Το framework:

* επιλέγει δυναμικά predictor
* αποφασίζει μεταξύ CV και KF

με βάση:

* speed
* heading change
* motion complexity

---

# 16. Risk Heatmap Forecasting

Δημιουργούνται:

* future danger heatmaps
* occupancy probability maps
* high-risk spatial zones

Χρήση:

* adaptive beam attention
* predictive illumination planning

---

# 17. Multi-Agent Interaction Modeling

Αναλύονται:

* pedestrian interactions
* cyclist interactions
* coupled trajectories
* social motion behavior

Υπολογίζονται:

* interaction scores
* proximity metrics
* direction similarity

---

# 18. Dynamic Beam Width Optimization

Το beam width αλλάζει δυναμικά ανάλογα με:

* uncertainty
* collision risk
* interaction density

Το σύστημα:

* ανοίγει beam σε dangerous scenarios
* περιορίζει beam σε low-risk conditions

---

# 19. Occlusion-Aware Prediction

Υλοποιήθηκε:

* visibility reasoning
* hidden VRU modeling
* uncertainty inflation

Το framework:

* εντοπίζει πιθανές occlusions
* αυξάνει safety margins

---

# 20. Trajectory Anomaly Detection

Εντοπίζονται:

* abnormal trajectories
* erratic motion
* sudden turning
* unstable VRUs

Υπολογίζονται:

* anomaly scores
* abnormal motion rankings

---

# Παραγόμενα Figures

Το framework δημιουργεί:

* waveform plots
* beat spectrum
* BER curves
* range-Doppler maps
* tracking visualizations
* adaptive beam figures
* risk heatmaps
* collision forecasting plots
* uncertainty envelopes
* interaction plots
* anomaly distributions

Όλα αποθηκεύονται στο:

```text
figures/
```

---

# Export Αποτελεσμάτων

Όλα τα metrics αποθηκεύονται σε:

```text
outputs/
```

Formats:

* JSON
* CSV
* NumPy arrays

---

# Εγκατάσταση

## Δημιουργία virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Εγκατάσταση Packages

```bash
pip install -r requirements.txt
```

---

# WOMD Setup

## Google Cloud Login

```bash
gcloud auth login
```

---

# Download WOMD Dataset

```bash
gsutil -m cp -r \
gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/validation \
data/womd/
```

---

# Εκτέλεση Experiments

## Part A

```bash
python run_part_a.py
```

---

## WOMD Forecasting

```bash
python -m part_b.womd_future_forecast
```

---

## Collision Forecasting

```bash
python -m part_b.collision_probability_forecasting
```

---

## Risk Heatmaps

```bash
python -m part_b.risk_heatmap_forecasting
```

---

## Occlusion-Aware Prediction

```bash
python -m part_b.occlusion_aware_prediction
```

---

# Ερευνητικές Συνεισφορές

Το framework επεκτείνει τα παραδοσιακά:

* FMCW radar systems
* adaptive headlights
* automotive illumination systems

εισάγοντας:

* predictive illumination
* uncertainty-aware forecasting
* probabilistic safety modeling
* collision forecasting
* interaction-aware beam steering
* risk-aware adaptive lighting

---

# Μελλοντική Εργασία

Πιθανές επεκτάσεις:

* Graph Neural Networks
* Transformer trajectory forecasting
* radar-camera fusion
* reinforcement learning beam control
* real-time CUDA acceleration
* autonomous illumination agents
* end-to-end driving safety systems

---

# Συγγραφέας

Παναγιώτα Γροσδούλη
Department of Electrical and Computer Engineering
Democritus University of Thrace

---

# License

MIT License

```
```
