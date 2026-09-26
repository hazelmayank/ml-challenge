# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 business, find the Source 2 / Source 3 records that describe the same
real-world business. Scored by macro F0.5 per S1 entity.

## Approach

1. **Normalise** (`normalize.py`, `prep.py`): lowercase, transliterate (Indic words via a
   dictionary learned from training pairs, `learn_indic.py`), strip junk and legal suffixes,
   canonicalise address abbreviations, states and ordinals.
2. **Block** (`block.py`): IDF-weighted token blocking within country (name words, address
   words, word pairs, name word × house number); keep the top 10 S1 per S2/S3 record.
3. **Features** (`features.py`): RapidFuzz similarities on names/addresses, house-number and
   legal-form difference features, blocking score/rank.
4. **Model** (`train.py`): LightGBM, 2-fold CV grouped by S1 on a 25% S1 sample.
5. **Decide** (`predict.py`): every S2/S3 record is assigned to its highest-scoring S1 if the
   probability exceeds a threshold tuned on out-of-fold macro F0.5.

## Setup

```bash
pip install -r requirements.txt
```

Paths are set by environment variables (defaults are relative to the repo root):

| Variable | Meaning | Default |
|---|---|---|
| `BER_DATA` | folder containing `train/` and `test/` TSVs | `student_resource/dataset` |
| `BER_WORK` | intermediate files | `work/` |
| `BER_OUTPUT` | submission files | `output/` |
| `BER_LGB_DEVICE` | `cpu` (default), `gpu` or `cuda` for LightGBM | `cpu` |

## Run order (from `src/`)

```bash
cp ../artifacts/indic_dict.json "$BER_WORK"/   # or: python prep.py train && python learn_indic.py
python prep.py train test
python block.py train --sample 0.25
python features.py train_cand_sample
python train.py 0.25          # prints CV macro F0.5 and the chosen threshold
python block.py test
python features.py test_cand
python predict.py             # -> matching_results.tsv, candidate_pairs.tsv
python ../../../student_resource/utils/validate_submission.py \
  --matching "$BER_OUTPUT/matching_results.tsv" \
  --candidate "$BER_OUTPUT/candidate_pairs.tsv" \
  --test-dir "$BER_DATA/test"
```

On Kaggle, use `kaggle/kaggle_runner.ipynb` from the repo root.
