# DESIGN — PartialSpoof Detection & Localization with DSP

> One line: detect and localize partially-spoofed speech with **DSP features
> (magnitude / phase / discontinuity) + LightGBM**. Evaluated with the official
> **Utterance EER** (detection) and **Range-EER** (localization). The model was
> upgraded from simple logistic regression to **LightGBM**.

---

## 1. Task
- **Partial Spoof (PS)**: a short TTS/VC-generated segment is inserted/substituted
  into otherwise genuine speech — one word/syllable can flip the meaning.
- **Goal**: use Digital Signal Processing (DSP) to (a) **detect** (does an
  utterance contain spoof?) and (b) **localize** (which time spans are spoof?).
- **Data**: PartialSpoof database v1.2 (built on ASVspoof2019 LA).

## 2. Dataset
- **Verified on disk**: con_wav / protocols / segment_labels.
  - segment labels are **.npy (per-frame 0/1 at 6+1 resolutions 0.01–0.64 s)**,
    convention **1 = bonafide, 0 = spoof** (NOT the text format the docs implied).
  - dev **24,844 = 2,548 bonafide (LA_D) + 22,296 spoof (CON_D)**; all four
    sources agree.
- **Key property**: spoof segments are spliced from the **same speaker** (VAD-cut,
  joined with cross-correlation + overlap-add, loudness-normalized to −26 dBov).
  → speaker / loudness cues are useless; **seam + vocoder artifacts** are the cues.

## 3. Pipeline (7 steps)
```
[1] audio 16 kHz
[2] framing 25 ms window / 10 ms hop (n_fft=512, Hann)
[3] DSP features (per frame)          -> features.py
[4] window pooling 0.16 s, mean+std+max  (1:1 with segment labels) -> pipeline.py
[5] classifier  LogReg -> LightGBM    -> model.py
[6] window P(spoof): localization = sequence, detection = max pooling
[7] median smoothing + evaluation (Utt-EER / Range-EER) -> model.py / evaluate.py
```
Run: `python3 src/run.py [--backend logreg|lgbm] [--smooth 5]`

## 4. DSP features (what, why)
| group | content | targeted artifact |
|---|---|---|
| **magnitude** | STFT log-magnitude (257), LFCC+Δ+ΔΔ (60) | vocoder spectral texture (linear filterbank keeps high-freq/formants) |
| **phase** | instantaneous-frequency deviation + phase flux (16 bands) | vocoder phase incoherence |
| **disc** | spectral flux + log-energy d1/d2 | concatenation-seam discontinuity (robust to loudness norm) |
| **seam** | F0 jump, voicing change, spectral novelty | the seam (boundary), directly |

Pooling uses mean+std+**max** — the seam is sparse, so max preserves the in-window peak.

## 5. Model — from simple regression to LightGBM (upgraded)
- **Baseline**: `LogisticRegression` (linear) = simple regression, for feasibility.
- **Main model**: **LightGBM** (gradient boosting, non-linear). `model.py` holds
  several tabular backends; LightGBM is the chosen method.
  ```python
  LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                 class_weight="balanced",
                 subsample=0.8, colsample_bytree=0.8)
  ```
- **How it is used**: train on window-level DSP features (e.g. full = 1101-dim) →
  `predict_proba` gives per-window P(spoof). Detection = max pooling, post-proc =
  median smoothing (pipeline unchanged, classifier swapped).
- **Extra**: ±1 neighbor-window temporal context further improves localization.
- Window labels are ~balanced (spoof ≈ 0.5), so class weighting matters little;
  the win is the non-linear decision boundary.

## 6. Evaluation metrics (official)
- **Utterance EER** (detection): standard EER on utterance scores.
- **Range-EER** (localization): reproduces the official `metric/RangeEER.py` using
  the **same library/logic (pyannote DetectionCostFunction, threshold sweep)**;
  only the input is our window scores instead of their model pkl. Reference uses
  the finer 0.02 s labels.

## 7. Results
### 7.1 dev 1,500-sample, 70/30 utterance split — *optimistic baseline*
| feature | model | Utt-EER% | Range-EER% |
|---|---|---:|---:|
| stft | LogReg | 9.56 | 13.96 |
| full | LogReg | 11.07 | 11.05 |
| stft | LightGBM | 2.69 | 9.96 |
| stft+phase+disc | LightGBM | 1.85 | 8.27 |
| **full** | **LightGBM** | **0.84** | **7.69** |
| full + context k=1 | LightGBM | 1.51 | **6.96** |

> LogReg → LightGBM alone: detection 11% → 0.8%, localization 11% → 7.7%.

### 7.2 Official protocol — train(25,380) → dev(24,844) full (LightGBM) ⭐
| feature | win-EER% | Utt-EER% | Range-EER% |
|---|---:|---:|---:|
| lfcc | 14.91 | 5.91 | 12.61 |
| stft | 10.77 | 2.91 | 9.78 |
| stft+phase+disc | 8.69 | 2.43 | 8.10 |
| **full** | 7.67 | **2.22** | **7.93** |

### 7.3 Official protocol — train → eval(71,237) held-out (LightGBM) ⭐⭐ *final trusted numbers*
| feature | win-EER% | Utt-EER% | Range-EER% |
|---|---:|---:|---:|
| lfcc | 24.46 | 19.27 | 25.47 |
| stft | 22.99 | 15.61 | 25.22 |
| stft+phase+disc | 22.96 | 15.83 | 26.02 |
| **full** | 21.57 | **13.88** | **24.60** |

### 7.4 Optimistic → official → held-out (key framing)
| full feature | dev internal (optimistic) | train→dev | **train→eval (final)** |
|---|---:|---:|---:|
| Utt-EER (detection) | 0.84 | 2.22 | **13.88** |
| Range-EER (localization) | 7.69 | 7.93 | **24.60** |

> Three-stage reality check: optimistic 0.84% → in-domain dev 2.22% →
> **unseen-attack eval 13.88%**. eval contains **unseen TTS/VC attacks**
> (ASVspoof2019 LA A07–A19 vs. train/dev A01–A06), so the generalization gap is
> expected (deep SOTA also degrades on eval). **13.88% / 24.60% are the trusted
> final numbers.** Reporting all three is honesty + realism.

## 8. Reliability checks
### 8.1 Speaker leakage (checked — none)
| split | speaker overlap | Utt-EER | Range-EER |
|---|---|---:|---:|
| utterance split (current) | 20/20 (all) | 0.84 | 7.69 |
| speaker-disjoint split | 0 | **0.59** | 8.61 |

Fully disjoint speakers do not hurt detection (even 0.59) → the model is **not
memorizing speakers**. Confirms PS's same-speaker design neutralizes speaker cues.
(dev has only 20 speakers; the real test is train→eval.)

### 8.2 Is the gap overfitting?
Not sample-overfitting (handcrafted features + regularized GBM, and dev is fine at
2.22%). It is **specialization to the seen attack distribution** (A01–A06) that
does not transfer to **unseen attacks** (A07–A19) — a known ASVspoof challenge,
i.e. a domain/attack-generalization gap, not high variance.

## 9. Key findings (presentation story)
1. **STFT is the strongest detector**; phase + discontinuity add on top
   (official Range-EER 13.96 → 11.05 on the optimistic split).
2. **Counter-intuitive: F0 (pitch) does not work.** The construction picks the
   smoothest join via overlap-add, so the **attacker already removed the pitch
   discontinuity**; **phase** survives best for boundary detection
   (F1 0.355 vs F0 0.127).
3. **LogReg → LightGBM** is the single biggest improvement.
4. **Official Range-EER** integrated for proper localization measurement.

## 10. Limitations & next
- Closing the unseen-attack gap is a fundamental limit of handcrafted DSP (would
  need SSL features / augmentation / domain generalization — out of scope for a
  "simple model").
- The gap is a **finding to explain, not a defect to hide**.
- Remaining: feature-importance analysis, report/README refresh, 15-min talk.

## 11. Code layout (step-by-step pipeline)
```
step1_data/      ps_data.py                 [1] data + label verification
step2_features/  features.py                [2][3] framing + DSP features
step3_dataset/   pipeline.py, build_full.py [4] window pooling + dataset build
step4_model/     model.py                   [5][6] classifiers + pooling/smoothing
step5_evaluate/  evaluate.py                [7] Utterance EER + Range-EER
step6_run/       run.py, run_full.py        end-to-end orchestration
step7_analysis/  analyze_ratio, compare_methods, seam_detect, + exploratory scripts
tools/           make_report.py             PDF report generator (non-pipeline tool)
figures/ results/ report/   artifacts
```
Each script bootstraps `sys.path` with the sibling `step*/`+`tools/` dirs, so
cross-step imports work from any working directory.
