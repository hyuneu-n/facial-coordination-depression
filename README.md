# Facial Coordination: Interpretable, Lightweight Depression Recognition from Facial Dynamics

Research code for depression recognition from **facial expression coordination** — the covariance structure of how facial Action Units (or landmarks) move together over time, modeled on the Riemannian SPD manifold. Language-independent, on-device-scale (tens of thousands of parameters), and interpretable — designed as an alternative to black-box deep models for small-sample clinical and cross-corpus settings.

> **Status:** research / work-in-progress. This repository holds the experimental pipeline and analysis scripts. **No datasets are included** (see *Data* below).

## Motivation

Most video-based depression recognition treats a session as a single static snapshot and optimizes accuracy on one benchmark. This project asks three different questions instead:

1. **What structure carries the signal?** Not individual facial muscle activity, but the **coordination** (covariance) between muscles/landmarks — modeled on the SPD manifold rather than flattened to Euclidean features.
2. **Where in time does the signal live?** Given a coordination model, how early in a session is depression detectable, and does that hold across corpora — a step toward the long-term goal of detecting a **prodromal (pre-onset) trajectory** rather than a static label.
3. **Is the signal real, or is it confound?** When a benchmark's performance is audited (position/scale/pose, recording environment), does it survive — and does a lightweight, interpretable design stay robust where a heavier baseline collapses?

## Key findings (so far)

**Coordination + lightweight design (D-Vlog, official split)**
- SPD-manifold coordination encoder (BiMap → ReEig → LogEig) + audio branch, simple concat fusion, reaches **AUC 0.81 / F1 0.79** vs. the original D-Vlog paper's 0.63/0.63 — at **~50-68K parameters** (CPU ~2ms/sample), roughly 100x smaller than deep multimodal baselines.
- Cross-corpus: strong where facial signal is expressive/clean (CMDC AUC 0.89, D-Vlog 0.81), weak where it isn't (LMVD 0.63, E-DAIC 0.62) — signal quality, not model capacity, is the limiting factor.

**Time-axis earliness (confirmed, replicated)**
- Truncating a session to its **first 40%** reaches AUC 0.77 vs. 0.80 for the full session (96%) — replicated on **D-Vlog and LMVD**. Learned temporal attention independently confirms the model concentrates on the session start (depressed subjects more front-loaded).
- The same early-sufficiency pattern holds for PHQ severity regression (CMDC, CCC 0.49 at 40% vs. 0.48 full).
- Coordination adds information beyond individual-AU dynamics (mean/std/AR1): combined > individual in 3/4 corpora with bootstrap CI excluding 0.
- The discriminative coordination consistently localizes to the **mouth/chin region (AU15–AU17)** in both clinical (CMDC) and vlog (LMVD) data.

**Confound audit on an independent longitudinal benchmark (MUD3, ACM MM 2025 / CCAC2026)**
- Reproduced the MUD3 baseline and found its reported performance is **substantially explained by recording confounds**: face position/scale alone (11-feature logistic regression) reaches F1 0.6875, matching the 1.58M-parameter baseline (F1 0.6717); removing the confound collapses the baseline to **F1 0.334** (lr/epoch sweep rules out under-training as the cause).
- A lightweight coordination model transplanted from the D-Vlog pipeline, evaluated under the **same confound-free conditions**, holds **F1 0.68–0.75** at 1/50–1/89 the parameter count (Welch t=3.42, p=0.0039, d=1.53, 10 seeds).
- Coordination itself did **not** transfer to MUD3 (short, edited social-media clips) — confirmed invalid via a frame-count sweep, not merely under-sampled. This is a documented negative result, not silently dropped.
- The AU15–AU17 discriminative pair identified via landmarks (above) was independently validated against **ground-truth AU intensity** on DISFA (27 subjects, FACS-coded): the pair ranks **8th of 66** AU pairs by real co-activation (r=0.314, p<0.00001).

**Negative results (documented, not hidden)**
- Cross-lingual "anchor invariance", learned question-weighting (QDS), AU-to-image rendering, and Transformer backbones did not hold up under small-sample constraints (early anchor-based phase).
- Within-session temporal-dynamics markers (rigidity/complexity) and network-topology features were D-Vlog-specific, not robust cross-corpus.
- Within-user change-point detection on MUD3 (testing for a longitudinal "state shift") found no signal (all p>0.05) — the genuine longitudinal-prodrome question remains open.

## Repository layout

```
code/       # numbered exploratory experiments (exp1 … exp141+, prep_*, probe_*)
data/       # (git-ignored) clinical/social-media datasets — not committed
features/   # (git-ignored) extracted features / caches
results/    # (git-ignored) experiment outputs, figures
```

Representative scripts, by phase:
- `code/exp19_shrinkage_spd.py`, `exp20_confirm.py` — early anchor + Riemannian SPD phase (CMDC/DAIC)
- `code/exp80_dvlog_official.py`, `exp110_light.py` — coordination model, D-Vlog official benchmark, lightweight design
- `code/exp104_early.py`, `exp106_attn.py`, `exp109_region.py` — time-axis earliness, attention, discriminative region
- `code/prep_mud3.py`, `exp121_mud3_coord.py` — MUD3 acquisition, model transplant
- `code/exp122_leakage.py`, `exp127_confoundfree.py`, `exp124_lrsweep.py` — confound audit + robustness verification
- `code/exp136_changepoint.py` — within-user change-point detection (negative result)
- `code/exp138_disfa_au1517.py`, `exp141_affectnet.py` — external AU/region validation (DISFA, AffectNet)

## Data

Publicly available / access-controlled corpora. **Datasets are NOT redistributed here** — obtain them from the original providers under their licenses.

| Dataset | Modality | Label | Role |
|---|---|---|---|
| **DAIC-WOZ / E-DAIC** | Audio, transcript, OpenFace/CLNF AUs | PHQ-8 | Clinical interview, anchor phase |
| **CMDC** | Text, audio, OpenFace 2.2 AUs | PHQ-9/HAMD | Clinical, strongest coordination signal |
| **D-Vlog** | Landmarks + acoustic | Binary | Main coordination benchmark (official split) |
| **LMVD** | OpenFace AUs + VGGish | Binary | Cross-corpus replication (large-scale vlog) |
| **MUD3** | Landmarks + acoustic (no raw video) | Binary, self-reported | Longitudinal confound audit (CCAC2026) |
| **DISFA** | Landmarks + FACS-coded AU intensity | AU intensity 0–5 | External AU-coactivation ground truth (no depression label; validation only) |
| **AffectNet** | Static face images | 8-way emotion | Static-image region check (no depression label; validation only) |

Facial features are landmark coordinates or OpenFace Action Units. No raw video is used or stored for any dataset.

## Method (current pipeline)

1. **Representation** — per-window/per-video covariance of AU or landmark coordinates (Ledoit–Wolf shrinkage), i.e. the *coordination* structure, not raw activity.
2. **Encoder** — SPDNet-style manifold layers (BiMap → ReEig → LogEig) followed by a GRU over the window/video sequence for temporal modeling.
3. **Fusion** — concatenation with a lightweight 1D-CNN audio branch (concat outperformed attention/gated/bilinear fusion in ablations).
4. **Evaluation** — AUC/F1 (classification), CCC/MAE (severity regression), multi-seed with permutation/bootstrap CI; confound-controlled variants (position/scale/pose-normalized, recording-environment-standardized) where applicable.

## Requirements

```
numpy scipy scikit-learn pyriemann torch pandas ruptures openpyxl
```

## Ethics & privacy

All datasets contain sensitive personal/clinical information and are used under their respective agreements. Only de-identified, pre-extracted facial/acoustic features are processed; no raw video is stored or shared for any dataset in this repository.

## Acknowledgements

Emotion & Memory Interaction Lab, Seokyeong University. Built on prior multimodal depression work (KIICE), the OpenFace / pyRiemann / dlib toolkits, and the D-Vlog, LMVD, MUD3, and DISFA dataset releases.
