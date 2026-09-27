# Facial Coordination: A Plug-in Encoder for Multimodal Depression Recognition

Research code for depression recognition from **facial expression coordination** — the covariance structure of how facial landmarks / Action Units move together over time. The current focus is a **plug-in encoder**: a small branch (+32,000 parameters) that attaches to an existing multimodal model without modifying it, so any performance change is attributable to the branch alone.

> **Status:** research / work-in-progress. This repository holds the experimental pipeline and analysis scripts. **No datasets are included** (see *Data*).

---

## What this repository actually establishes

This section is written to be checkable. Claims that did not survive testing are listed in *Retracted claims* below, not quietly removed.

### 1. A unified benchmark harness (`code/bench/`)

Four published models for the same task, re-implemented and run under one fixed protocol across five corpora:

| Model | Source | Architecture |
|---|---|---|
| BLSTM | 2019 baseline | per-modality BiLSTM, temporal mean, concat |
| TFN | Zadeh et al., EMNLP 2017 | outer-product tensor fusion |
| **DepDetector** | **Yoon et al., AAAI 2022** | Transformer encoders + cross-modal attention — *the model from the D-Vlog dataset paper* |
| TAMFN | Zhou et al., IEEE TNSRE 2023 | global-information TCN + time-aware attention fusion |

**Protocol is fixed in `code/bench/README.md`** (T=256 resample, per-sample z-score, validation-AUC model selection, threshold fixed at 0.5, both seed-mean and ensemble reported).

**Reproduction check (D-Vlog official split, weighted-average F1 — the metric the original papers report):**

| Model | Published | Reproduced | Δ |
|---|---|---|---|
| DepDetector | 0.6482 | 0.6477 | −0.0005 |
| TAMFN | 0.661 | 0.657 | −0.004 |

Depression-specific models reproduce their published numbers, so the harness is comparable to published conditions. Generic baselines (BLSTM, TFN) score 7–11%p *higher* here than in the papers that used them as baselines — reported as-is, which is the conservative direction for our own claims.

### 2. A plug-in encoder that attaches to any host at constant cost

`Graft` wraps any host exposing `features()` / `feat_dim`. The host is left untouched; only the coordination vector is concatenated before a new classifier.

| host | base | grafted | added |
|---|---|---|---|
| DepDetector | 202,305 | 234,305 | **+32,000** |
| BLSTM | 166,529 | 198,529 | **+32,000** |
| TFN | 31,233 | 63,233 | **+32,000** |
| TAMFN | 134,852 | 166,852 | **+32,000** |

### 3. Where the graft helps — and where it does not

Seed-mean change from grafting (`fig22_graft_grid.png`). Both metrics shown because they disagree in places:

| corpus | BLSTM | TFN | DepDetector | TAMFN | cells with a drop |
|---|---|---|---|---|---|
| **LMVD** (n=1556) | +1.7 / +1.4 | +1.7 / +5.6 | +0.7 / ±0.0 | +1.4 / +5.4 | **0 / 8** |
| D-Vlog (n=952) | +1.3 / +0.4 | −6.4 / −2.9 | +7.2 / +0.8 | +10.7 / +2.7 | 2 / 8 |
| MUD3 raw (n=650) | +4.0 / +0.7 | −6.6 / +2.2 | ±0.0 / −0.2 | −7.3 / +3.5 | 3 / 8 |
| MUD3 confound-removed | +1.8 / −2.0 | +5.3 / +1.8 | −9.9 / +0.2 | +1.1 / +3.8 | 3 / 8 |
| CMDC (n=45) | +20.2 / −3.0 | +3.4 / −6.9 | +4.1 / +2.8 | −4.0 / +0.3 | 3 / 8 |

*(each cell: ΔF1 %p / ΔAUC %p)*

**LMVD is the only corpus where no host×metric cell degrades.** It is also the largest corpus, with the smallest seed variance. Effects there are small (+0.7–1.7%p F1) but consistent. Significance testing under repeated cross-validation is in progress.

On MUD3 raw with repeated cross-validation (all 650 subjects, 5-fold × 3 repeats), DepDetector + graft gains **F1 +4.8%p and AUC +3.4%p with bootstrap CIs excluding zero** — the first significant result in this line of work.

### 4. Methodological findings (these are robust, and they are warnings)

These came out of trying to validate the encoder and are reported because they affect how anyone should read benchmark tables in this area.

- **Ensembling silently favors high-variance models.** Averaging 5 seeds before scoring reverses the *sign* of the graft effect in several cells (e.g. D-Vlog/TFN: +5.7%p by ensemble, −6.4%p by seed-mean). See `fig24_ensemble_artifact.png`. Comparisons that report only ensemble numbers will overstate any model with higher variance.
- **TAMFN is not reproducible run-to-run.** Identical code and seeds give different results across executions (F1 0.3448 → 0.3417, ensemble 0.5029 → 0.4824). Its predictions cluster near the 0.5 threshold, so tiny floating-point differences flip decisions. AUC is stable (0.666–0.668); **TAMFN should be reported by AUC only.**
- **The SPD-manifold layer is not carrying the benefit.** Removing it (`nospd`) matches or beats the full design in 5 of 6 host × corpus comparisons under repeated CV. However `nospd` is itself unstable on the D-Vlog official split — 3 of 5 seeds collapse to near-zero recall (F1 0.18/0.27/0.69/0.71/0.18), a bimodal training failure that fold-averaging hides.
- **Small test splits make significance unreachable.** With the official MUD3 test set (66 subjects) every bootstrap CI includes zero. Repeated cross-validation over all subjects is required before any significance claim.

### 5. Confound audit (MUD3, ACM MM 2025 / CCAC2026)

The MUD3 authors' own README states the collection method: depressed users were gathered with depression keywords, non-depressed with `grwm` / daily-vlog keywords — **two different filming genres**. Measured face-centre coordinates and scale differ systematically between groups.

Face position/scale alone (4 features, logistic regression) reaches F1 0.6875, matching the 1.58M-parameter published baseline (0.6717). Removing position/scale/rotation collapses that baseline to **F1 0.334**; a learning-rate sweep (100×) and doubled epochs rule out under-training.

This is why every MUD3 result here is reported under **both** raw and confound-removed conditions.

---

## Retracted claims

Listed because earlier versions of this README asserted them.

| Claim | Why it was withdrawn |
|---|---|
| "Session earliness: first 40% reaches 96% of full performance — a step toward prodrome detection" | A segment control (first 0–40% / middle 30–70% / last 60–100%) was run for the first time. **All four models score highest on the middle segment.** The proposed model too (front AUC 0.7708 < middle 0.7741). Retention at 40% is 97–101% for *every* model, so it is a property of the data, not of this method. |
| "Temporal attention concentrates on the session start" | Contradicted by the segment control above; needs re-examination. |
| "CPU ~2 ms/sample" | That figure excluded the coordination computation (PCA + Ledoit-Wolf, 2.54 ms). Honest total is **3.93 ms**. |
| "Face position/scale, 11 features" | The leakage probe used **4** features (cx, cy, scale, scale_std). |
| "Coordination did not transfer to MUD3" | Superseded: with the graft formulation it does transfer on MUD3 raw (significant under repeated CV). |

---

## Repository layout

```
code/
├── bench/                  # unified benchmark harness
│   ├── README.md           # fixed protocol — do not change mid-study
│   ├── data.py             # D-Vlog loader + npz cache + registry
│   ├── data_extra.py       # LMVD / CMDC / E-DAIC loaders (AU and landmark variants)
│   ├── data_mud3.py        # MUD3 loader (raw / Procrustes-aligned)
│   ├── models.py           # 4 hosts + Graft wrapper + CoordBranch(full/nospd/notime/nocov)
│   ├── run.py              # single entry point; early-truncation and segment-control paths
│   ├── repeat_cv.py        # repeated stratified CV + paired tests + bootstrap CIs
│   ├── bootci.py           # bootstrap CI on saved predictions
│   ├── metrics.py          # Acc/P/R/F1/AUC, parameter count
│   ├── make_table.py       # master comparison table
│   └── figs*.py            # figures
└── exp*.py                 # earlier single-question experiments (historical)
results/bench/              # per-corpus CSVs, master table, figures
```

### Model names

Hosts: `blstm` `tfn` `depdetector` `tamfn`
Grafted: `<host>+full` `<host>+nospd` `<host>+notime` `<host>+nocov`
Standalone proposed encoder: `ours`

### Corpora

`dvlog` `lmvd` `lmvd_lmk` `cmdc` `cmdc_lmk` `edaic` `mud3` `mud3_aligned`

E-DAIC is excluded from graft evaluation: **all five models score F1 0.17–0.41**, so the corpus cannot resolve a module effect. The exclusion is decided by baseline performance, independent of our own results.

---

## Running

```bash
python -m bench.run --corpus dvlog --model 'depdetector+full' --seeds 5
python -m bench.repeat_cv --corpus mud3 --repeats 3 --models 'depdetector,depdetector+full,depdetector+nospd'
python -m bench.make_table
```

Useful flags: `--lr`, `--ep`, `--pca`, `--coord_lr_mult`, `--early 0.2,0.4,...`, `--seg 0:0.4,0.3:0.7,0.6:1.0`

---

## Data

No datasets are included. D-Vlog, LMVD, CMDC, E-DAIC and MUD3 each require their own access agreement from the original providers. Feature extraction follows each dataset's published specification (OpenFace landmarks/AUs, OpenSMILE or VGGish acoustic features).

---

## Honest summary

A coordination branch can be attached to four different multimodal depression models at a constant +32,000 parameters. It produces a **significant gain on MUD3 raw** and a **small but uniformly non-negative effect on LMVD**. On other corpora the effect depends on the host and on the metric, and one design component originally assumed essential (the SPD manifold layer) does not appear to contribute.

The most transferable results here may be the methodological ones in §4: ensembling artifacts, an irreproducible published baseline, and the inadequacy of small official test splits for significance testing.
