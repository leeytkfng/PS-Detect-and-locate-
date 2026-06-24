# PS-Detect-and-locate

Partial-spoof (PS) **detection & localization** on the
[PartialSpoof](https://github.com/nii-yamagishilab/partialspoof) database (v1.2).

A partial spoof embeds short synthetic/edited speech segments into otherwise
genuine human audio. The goal here is to use **DSP features + a simple model**
to (a) decide whether an utterance contains spoofed speech and (b) **localize
which time spans are spoofed**.

## Dataset (dev subset used here)

Downloaded from Zenodo record `5766198` via the upstream `01_download_database.sh`.
We use the **dev** split only (train/eval are ~5–6 GB each and skipped for now).

| source | content |
|---|---|
| `dev/con_wav/<id>.wav` | 24,844 utterances, 16 kHz mono |
| `protocols/.../PartialSpoof.LA.cm.dev.trl.txt` | utterance label (`bonafide`/`spoof`) |
| `segment_labels/dev_seglab_<res>.npy` | per-frame `0/1` labels at 7 resolutions (0.01–0.64 s) |

**Verified facts**
- Frame label convention: **`1` = bonafide, `0` = spoof**.
- IDs: `LA_D_*` = original genuine ASVspoof2019 utterance (all `1`);
  `CON_D_*` = concatenated (may be partially spoofed).
- dev counts agree across all four sources: **24,844 total = 2,548 bonafide + 22,296 spoof**
  (of the 22,296 spoof: 9,900 fully spoofed, 12,396 *partial* spoof).
- Segment-label `.npy` is a 0-d object array wrapping `defaultdict{utt_id: array}`;
  `#frames == ceil(duration / resolution)` at every resolution.

> Note: the dataset itself is **not** committed (see `.gitignore`); it lives in the
> local `partialspoof/` clone. Set `PS_DATA` to point the code at another copy.

## Layout

```
src/
  ps_data.py   data paths, protocol + segment-label loaders, frames->intervals
  ps_plot.py   waveform + spectrogram with spoof spans shaded
figures/       audio<->label verification plots
```

## Quick start

```bash
# inspect one utterance (labels, intervals, multi-resolution sanity check)
python3 src/ps_data.py CON_D_0000000 0.16

# render waveform + spectrogram with spoof regions shaded
python3 src/ps_plot.py CON_D_0000000 0.02
```

## Method (in progress)

Features: **LFCC**, **delta / delta-delta**, **STFT spectrogram**. Detection +
localization model and evaluation results to follow.
