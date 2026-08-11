# Facial Prodrome: Context-Anchored Facial Analysis for Depression Recognition

Research code for depression recognition from **facial Action Unit (AU) time series**, with a focus on **which interview context (question) the face is observed in**. Language-independent (facial signals), lightweight, and interpretable — designed for small-scale clinical settings where large deep models overfit.

> **Status:** research / work-in-progress. This repository holds the experimental pipeline and analysis scripts. **No datasets are included** (see *Data* below).

## Motivation

Most video-based depression recognition treats the whole interview as one blob. We investigate a different question: **the diagnostic value of facial behaviour depends heavily on *what the person was asked at that moment* (the "anchor").** Rather than proposing a heavier model, we study *where to look* and *how to represent* facial dynamics under small-sample clinical constraints.

## Key findings (so far)

- **Anchor effect (statistically significant).** Selecting AU segments aligned to specific interview questions improves depression discrimination over whole-interview aggregation — DAIC-WOZ ΔAUC +0.23 (p=0.002), CMDC ΔAUC +0.08 (p=0.006), 10-seed paired test.
- **Question diagnosticity varies widely.** Per-question AUC ranges 0.36–0.79 (DAIC) and 0.49–0.78 (CMDC); symptom-related questions (sleep, fatigue) tend to be most diagnostic.
- **Facial covariance geometry + anchor (best result).** On CMDC, representing anchored AU segments as covariance matrices (Ledoit–Wolf shrinkage) on the Riemannian SPD manifold reaches **AUC 0.885** (permutation p=0.005), beating Euclidean stats, whole-interview, and random-window baselines (4-way control). Data-quality dependent: reproduces on CMDC (OpenFace 2.2) but not DAIC (older CLNF features).
- **Negative results (documented).** Cross-lingual "anchor invariance", learned question-weighting (QDS), AU-to-image rendering, and Transformer backbones did **not** hold up under small-sample constraints.

## Repository layout

```
code/       # numbered exploratory experiments (exp1 … exp20, probe_*)
data/       # (git-ignored) clinical datasets — not committed
features/   # (git-ignored) extracted features / caches
```

Representative scripts:
- `code/exp15_anchor_proof.py` — anchor vs whole-interview, statistical test
- `code/exp16_daic_adi.py` — per-question diagnosticity (DAIC)
- `code/exp19_shrinkage_spd.py` — Riemannian SPD + anchor, 4-way control
- `code/exp20_confirm.py` — permutation test + AU-pair interpretation

## Data

This project uses publicly available / access-controlled clinical corpora. **Datasets are NOT redistributed here** — obtain them from the original providers under their licenses:

| Dataset | Modality | Label | Access |
|---------|----------|-------|--------|
| **DAIC-WOZ** | Audio, transcript, facial features (OpenFace/CLNF AUs) | PHQ-8 | https://dcapswoz.ict.usc.edu/ (EULA) |
| **E-DAIC** (Extended DAIC-WOZ) | Audio, transcript, OpenFace 2.0 features | PHQ-8 | https://dcapswoz.ict.usc.edu/ (EULA) |
| **CMDC** (Chinese Multimodal Depression Corpus) | Text, audio, OpenFace 2.2 AUs | PHQ-9 / HAMD | https://ieee-dataport.org/open-access/chinese-multimodal-depression-corpus (EULA) |

Facial features are OpenFace Action Units (`AU*_r` intensities). No raw video is used or stored.

## Method (current pipeline)

1. **Anchor selection** — locate question segments from transcripts (e.g. sleep/fatigue/positive-memory prompts).
2. **Representation** — per-segment AU statistics, or covariance (SPD) on the Riemannian manifold.
3. **Classifier/regressor** — lightweight (logistic regression / Ridge / tangent-space logistic), chosen for small-sample robustness over deep models.
4. **Evaluation** — AUC (classification) and CCC/MAE (PHQ severity regression), 10-seed cross-validation with permutation / Wilcoxon tests.

## Requirements

```
numpy scipy scikit-learn pyriemann openpyxl
```

## Ethics & privacy

All datasets contain sensitive clinical information and are used under their respective agreements. Only de-identified, pre-extracted facial features (AUs) are processed; no raw face video is stored or shared.

## Acknowledgements

Emotion & Memory Interaction Lab, Seokyeong University. Built on prior multimodal depression work (KIICE) and the OpenFace / pyRiemann toolkits.
