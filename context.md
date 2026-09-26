# Amazon ML Challenge 2026: Project Context

_Last updated: 26 Sep 2026, ~22:50 IST_

This file is the single source of truth for the team: what the problem is, what has been built, what we learned from the data, current scores, and what comes next.

---

## 1. Competition at a glance

| Item | Detail |
|---|---|
| Task | **Business entity resolution.** For every Source 1 (S1) business, find all Source 2 / Source 3 (S2/S3) records that are the same real-world business. |
| Metric | **Macro F0.5**, computed per S1 entity and then averaged. Singletons: an empty prediction scores 1.0, any predicted match scores 0.0. |
| Scored file | `output/matching_results.tsv` (tab-separated, one row per test S1, comma-joined IDs, empty when there are no matches) |
| Also required | `output/candidate_pairs.tsv`: the exact candidate set the model scored. Final matches must be a subset of it. |
| Validator | `student_resource/utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test` |
| Hackathon window | 72 hours, **25–27 Sep 2026**. Portal countdown at 19:50 on 25 Sep showed about 2 days 4 hours left, so it ends around **28 Sep, 00:00 IST**. |
| Submission limit | **5 per day** |
| Leaderboard | Refreshes every 15 min. Public board uses a subset of test; final rank uses the private board. |
| Day-1 leaderboard | Ranks 4–11 at **0.984–0.985** (as of 25 Sep evening) |
| Prizes | Top 50 get PPIs; top 500 at the 48-hour mark get +$100 AWS credits |
| Rules | No external data, lookups or geocoding APIs. Models must be MIT/Apache-2.0 and ≤8B params. Country is an open set (France appears in test only). |
| Final package | zip containing `output/` (2 TSVs), `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`), filled-in `Documentation_template.md` |

## 2. Compute setup

- **All development runs on the laptop:** 16 threads, 13 GB RAM, no NVIDIA GPU.
- **Run heavy steps one at a time.** Running two in parallel caused out-of-memory crashes (segfault, "bad allocation").
- Memory lessons: tokenise S1 in 300k chunks in `block.py`. In `features.py`, load only candidate records, use 1M-pair chunks spilled to disk, and avoid polars list set-ops (they cost ~1 GB per 500k pairs, so they were removed).
- **Disk:** C: had 0 bytes free at 00:30 on 26 Sep. Feature parts are no longer merged into one file (train/predict read `*_feat_parts/` directly). About 5 GB free after cleanup.
- Test timings: blocking ~15 min (resumable, auto-retry after a crash); features 98M pairs ~17 min; **predict 68 min** (a 1500-tree, 127-leaf model is slow; use fewer or smaller trees next time); validator 5 min.
- Timings on the laptop: prep 7 min; blocking of the train 25% sample 5 min; train features (25M pairs) 9 min; LightGBM 2-fold ~35 min (1500 rounds, still improving).
- AWS: account `hazel.mayank`, bucket `s3://amlc26-hazel.mayank` (us-east-1), dataset zip uploaded. Every SageMaker quota on this new account is 0 (domains, `ml.m5.2xlarge` notebook), so SageMaker is blocked. Quota requests are pending. `ml.t3.medium` notebook (free tier) is the only likely option and is weaker than the laptop.
- GPU options if needed later: Kaggle/Colab free T4, or a teammate's older AWS account.
- Python env: `.venv/` in the workspace root. Pinned in `code/business_entity_resolution/requirements.txt`.

### 2a. Kaggle (from 26 Sep evening): the main compute
- SageMaker dropped. Code lives in GitHub **hazelmayank/ml-challenge** (public, branch `main`), pushed from this folder. An allow-list `.gitignore` keeps data, work, output, `.venv` and the tarball out.
- Runner: `kaggle/kaggle_runner.ipynb` (clone/pull → deps + LightGBM GPU check → 2% smoke test in `work_smoke/` → full train → test → validator → artifacts tagged with the commit).
- Kaggle gives ~30 GB RAM but only ~4 CPU cores, and `/kaggle/working` holds ~20 GB. The pipeline is CPU-bound; a GPU only helps LightGBM, and only if the runner's check passes.
- v2 train OOM fixed: features are cast to float32 in polars before `to_numpy`. v2 has not been trained yet.
- Feature speed is ~12 s per 250k pairs on the laptop. The 8 h in `feat_train_v2.log` was a stall, not compute.
- `block.py` / `features.py` resume from existing parts, so after changing code, delete the parts folders (`fresh()` in the runner).

## 3. Data facts (from `reports/step1_inspection.json`)

| File | Rows |
|---|---:|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 (7,638,365 true pairs) |
| test_source1 | 1,732,544 (US 663k, India 810k, **France 259k**) |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

**Key findings the pipeline relies on:**
1. **Each S2/S3 record belongs to at most one S1** (max = 1 over all 7.6M links). So we solve it in reverse: for each S2/S3 record, pick its best S1 and accept only if confident.
2. **Matched pairs always share the same country** (100%), so we block within country. France is handled automatically because country is just a string.
3. About **26% of S2/S3 records match no S1** (distractors), so the model must be able to say "no match".
4. Singletons are 5.6% of S1. Matches per S1 range from 0 to 11 (mean 3.67 when there is at least one).
5. S1 is clean Latin text with no empty fields. S2/S3 contain about 470k Indic-script names (Devanagari/Tamil/Telugu/Kannada…), about 300k accented names, about 3.4% empty addresses, and `null`/`<NULL>` inside addresses.
6. The data is **synthetic** with templated noise:
   - typos (`Power`→`Ponr`)
   - junk (`>>`, `[Inc]`, `***`, `[[LLC]]`, phone numbers)
   - filler words (`Center`, `Services`, `Partners`)
   - shuffled word order
   - `name.com` domains
   - random aliases sharing the same address (`Solkeloquo`, `X D.B.A. Y`)
   - address variants (`MO`/`Missouri`, `Ave`/`Avenue`, `10th`/`Tenth`, `45ND`, uppercase, reordered parts)

7. **Unlinked S2/S3 records are deliberate decoys:** near-copies of a real S1 with one detail changed: the house/unit number (`2405` vs `12405`, `T-24/309` vs `T-24/302`), the legal form (`Private` → `Public Limited`, `Pvt Ltd` → `LLP`), a credential (`MD` vs `DMD`), or a name word (`Sanchez` → `Shah`, `Services` → `Value`). True matches carry their own benign noise (`1479` for `14793`, `1056c` for `1056`, dropped suffixes). Exact-difference features are needed to tell the two apart.
8. **v1 OOF error breakdown** (1.9M linked sample records + 670k unlinked):
   - blocking misses 61.5k (23k Indic names, 21k missing address, 18k other)
   - low confidence 22k
   - wrong top S1 15k (12k missing address)
   - **decoys wrongly matched 29.8k (4.4% of unlinked)**
9. **Indic dictionary** (`learn_indic.py`): 1,347 word mappings learned from 551k aligned train pairs, covering 97.8% of train and **96.4% of test** Indic name words (`एसएस फूड प्राइवेट लिमिटेड` → `ss food private limited`).

## 4. Pipeline (`code/business_entity_resolution/src/`)

| Step | Script | What it does | Status |
|---|---|---|---|
| 0 | `step1_inspect.py` | Data inspection report | Done |
| 1 | `normalize.py` | Lowercase, unidecode, junk removal, legal-suffix stripping (`name_core`), `name_compact` (for domains), address abbreviation/state/ordinal canonicalisation, `addr_nums` | Done |
| 2 | `prep.py` | TSV → cleaned parquet in `work/` (22M records in about 7 min) | Done |
| 3 | `block.py` | IDF-weighted token blocking within country. Tokens: name words, address words, compact name, **name-word pairs, address-word pairs, name-word × address-number**. Tokens with S1 frequency >150 are dropped. Keeps the top 10 S1 per S2/S3 record. | Done; test run in progress |
| 4 | `features.py` | About 25 features: rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on name_norm, name_core, name_compact, addr_norm, addr_nums; house-number equality; word overlap; lengths; flags (indic, domain, addr_missing, src); blocking score, rank, gap | In progress |
| 5 | `train.py` | LightGBM, 2-fold CV grouped by S1, on a **25% S1 sample** (all of those businesses' S2/S3 records plus 25% of unlinked ones). Assign each S2/S3 record to its argmax S1 if p > t, with t tuned on OOF macro F0.5. | In progress |
| 6 | `predict.py` | Score test pairs, write both output TSVs | Pending |
| — | `metric.py` | Exact official macro-F0.5 scorer | Done |

## 5. Results log

| Date / time | Experiment | Metric |
|---|---|---|
| 25 Sep 21:00 | Blocking v1: single tokens, df ≤ 150 | recall@10 = **0.769** |
| 25 Sep 21:15 | Blocking v2: + word pairs + name × number tokens | recall@10 = **0.968**, recall@1 = 0.923, 9.9 candidates per record |
| 25 Sep ~23:15 | Blocking ceiling (perfect matcher on v2 candidates) | **0.98785** |
| 25 Sep ~23:15 | LightGBM v1: 23 features, 2-fold OOF, argmax + threshold t=0.30 | **CV 0.97638** (pair P 0.994, R 0.948, singletons 0.977) |
| 26 Sep ~01:00 | Test run v1: 98.4M candidate pairs → 5.91M matches; validator **PASS**. Per country: France 4.5% empty, 3.56 matches per S1; India 5.8% / 3.35; US 5.4% / 3.44 | ready to submit |
| 26 Sep 22:31 | **Submission #1 (v1) public LB** (validator re-run: PASS) | **0.945857** vs CV 0.97638: gap 0.030 |
| 26 Sep ~19:30 | **Kaggle smoke test** (v2, 2% sample, commit a04a46e): recall@10 0.9758, ceiling 0.99192 | **CV 0.98626** at t=0.20 (grid edge → grid now starts at 0.05) |
| 26 Sep ~20:00 | Kaggle v2 25% blocking: recall@10 **0.9768**; features ~6 s per 250k pairs (2× laptop) | full train running |
| 26 Sep ~20:30 | **Blocking-miss analysis** (v2, 25%): 36.0k of 1.90M links missed (1.9%). Address missing 14.4k (**18.6% miss rate** for no-address records), both fields similar but typo'd 7.5k, Indic not in dictionary 5.7k (4.2%), address ok + domain/alias name 4.5k, name ok + address variant 2.6k, both differ 1.4k. India 3.1% vs US 1.1%. | → `embed.py` |
| 26 Sep ~20:15 | **v2 on Kaggle, 25% sample** (commit 2019045): blocking recall@10 0.9768 | **CV 0.98386** at t=0.30 (final refit interrupted, so no v2 model saved) |
| 26 Sep ~20:30 | **Embedding probe** (first 300k sampled S2/S3, T4×2 at ~6k texts/s): recall block 0.9794 → **union 0.9904** (misses −53%). Indic 0.960 → 0.991, has address 0.989 → 0.998, **no address 0.764 → 0.829** (still weakest), India 0.973 → 0.989, US 0.984 → 0.992. Pairs per record ~9.9 → ~18 | full train with embeddings next |
| 26 Sep ~22:45 | **CV–LB gap diagnosed:** the 25% sample kept all S2/S3 records of sampled S1s but only 25% of unlinked decoys (4× fewer decoys per S1 than test) and none of the other S1s' records (test has blocking-missed records attaching to wrong S1s). CV was optimistic and t chosen too low. Fix: train/evaluate on **all** train S2/S3 records (`train.py 1.0`), made affordable by `prune.py`. France (15% of test) is still unmeasurable. | — |
| 26 Sep ~22:45 | **Rule update from organisers:** a smaller candidate set per S1 ranks higher in the final evaluation, and `candidate_pairs.tsv` = exactly what the final model scores. v1 ≈ 57 candidates per S1. → `prune.py` (stage-1 model, top-K per record + p_min), `model.py` (XGBoost CUDA / LightGBM) | — |
| 26 Sep ~21:00 | **`embed.py` (GPU)**: multilingual-e5-small name+address embeddings, per-country top-10 kNN added as candidates + `emb_sim`/`emb_gap`/`erank` features. Tested locally on CPU (mini data); Kaggle probe pending | _probe pending_ |
| 26 Sep ~01:45 | **v2 started.** Changes: Indic dictionary; `legal`/`addr_ids`/`hnum` fields; 9 difference features (legal_rel, hnum_rel, hnum_lev, ids extra counts, unmatched name words, first-word similarity) + addr_ids similarities; empty fields read as ""; French street words; LightGBM 63 leaves, ≤800 rounds. v1 model archived in `work/v1/`. | _running_ |

## 6. Plan / roadmap

**Phase 1 (tonight): first valid submission**
- [x] Inspect data, clean, block
- [x] Features, LightGBM CV score, tuned threshold (CV 0.9764, t=0.30)
- [x] Test predictions, run the validator (PASS) → `output/matching_results.tsv` ready as **Submission #1**

**Phase 2 (26 Sep): climb to 0.97+**
- [ ] Analyse blocking misses (3.2%) and raise recall above 99%: TF-IDF char n-gram kNN, larger top-K, tuned df cap
- [ ] **Indic → Latin word dictionary learned from training pairs** (e.g. `प्राइवेट`→`private`); the current unidecode output is poor (`eses phuudd`)
- [ ] S1-side context features computed on the full candidate graph (competition between S2/S3 records for the same S1)
- [ ] Error analysis loop on OOF false positives and false negatives
- [ ] Be in the top 500 at the 48-hour mark (about 27 Sep evening)

**Phase 3 (27 Sep): top-50 push (0.98+)**
- [ ] France-specific normalisation (accents, `rue`, `bis`, `SARL/SAS`)
- [ ] Threshold and singleton tuning, possibly per-source thresholds
- [ ] Optional: multilingual embeddings (`intfloat/multilingual-e5-small`, MIT) if a GPU is available

**Phase 4 (last ~6 h): package**
- [ ] Final run, validator PASS, `README.md`, fill in `Documentation_template.md`, zip

## 6a. Detailed implementation plan (how each step is and will be built)

### Working rules (apply to every step)
1. **One change → one measurement.** Every idea is judged by the OOF macro F0.5 from `train.py` on the fixed 25% S1 sample. It is kept only if the score goes up.
2. **Blocking recall is tracked separately** (`block.py` prints recall@k). The "blocking ceiling" line in `train.py` is the best score possible with the current candidates.
3. **Heavy steps run sequentially** on the laptop (RAM limit). Development happens on the 25% train sample; test always runs in full.
4. **Submissions (5 per day):** submit only when the CV score has improved by at least ~0.002, or to check that CV matches the leaderboard. The CV-vs-leaderboard gap is logged in §5 every time.
5. `context.md` is updated after every milestone.

### Step A: Submission #1 (tonight, in progress)
- **How:** `features.py` computes ~25 similarity features per candidate pair → `train.py` trains LightGBM with 2 folds grouped by S1 (a business never appears in both folds) → OOF probabilities → each S2/S3 record goes to its highest-probability S1 if p > t → t is searched over 0.20–0.95 to maximise macro F0.5 → the final model is retrained on the whole sample → `predict.py` scores ~100M test pairs in 5M chunks and writes both TSVs → validator → upload.
- **Expected:** CV ≈ 0.90–0.95 (limited by 96.8% blocking recall and weak Indic handling).

### Step B: Fix blocking misses, 96.8% → 99%+ recall (26 Sep morning)
- **How:**
  1. Dump a sample of missed true pairs and group them by cause (Indic name, alias name, missing address, common words).
  2. **Raise the df cap only for pair tokens** (pairs are rarer, so a cap of 150 → ~500 is affordable), and increase TOP_K from 10 to 15–20.
  3. Add **character 3-gram tokens of `name_compact`**, restricted to rare n-grams, so heavy typos still share tokens (`etrepndiels` ~ `enterprises`).
  4. Add a **phonetic/consonant skeleton** token for names (drop vowels: `payne enterprises` → `pyn ntrprss`), which survives most typos.
- **Check:** recall@TOP_K on the sample goes up while candidates per record stay ≤ 20 (to keep runtime/memory OK).

### Step C: Indic-script names (26 Sep)
- **Problem:** ~5% of S2/S3 names are in Devanagari/Tamil/Telugu/Kannada etc. unidecode gives `eses phuudd praaivett`, which doesn't match `ss food private`.
- **How:** learn a **word translation table from the training pairs**. For every true pair where the S2/S3 name is Indic and the S1 name is Latin with the same number of words, align words by position and count the (Indic word → Latin word) co-occurrences. Keep mappings with high agreement (e.g. `प्राइवेट`→`private`, `लिमिटेड`→`limited`, `राज`→`raj`). In `normalize.py`, translate Indic names word by word with this table and fall back to unidecode for unknown words. Addresses (Indic state names such as `उत्तर प्रदेश`) are handled the same way.
- **Rule check:** this uses only the provided training data, so it's allowed.
- **Check:** recall and F0.5 on the `indic` slice of OOF before and after.

### Step D: Better features (26 Sep afternoon)
- **Context features on the full candidate graph:** for each S1, how many S2/S3 records have it as their top choice, and this record's rank among them. They'll be computed from the full-train run (or test) so train and test distributions match.
- **Rare-word agreement:** IDF-weighted share of the S1 name's rarest word that also appears in the candidate.
- **Address components:** zip/PIN match, street-name match without the number, city match.
- **Alias flags:** `aka` / `d.b.a.` present, `name_compact` found inside the other name.
- **Check:** feature importance plus OOF delta; drop features that don't help.

### Step E: Error analysis loop (26 Sep evening → 27 Sep)
- **How:** sort OOF errors into false positives (wrong merges, costly under F0.5) and false negatives. Read 50 of each and write a rule or feature for the biggest group. Repeat 3–4 times.
- Also check the score by country (US vs India) and by source (S2 vs S3) to find weak slices.

### Step F: Decision tuning (27 Sep)
- Separate thresholds per source (S2/S3) and per script (Latin/Indic), tuned on OOF.
- **Second-best rule:** if the top two S1 candidates are almost tied, abstain (protects precision).
- **Singleton protection:** an S1 whose best incoming probability is low gets no matches.

### Step G: France readiness (27 Sep)
- Training has no French data, so this is rule-based: accents are stripped by unidecode; add French street words (`rue`, `avenue`, `boulevard`/`bd`, `chemin`, `allee`, `place`, `bis`/`ter`), legal forms (`sarl`, `sas`, `sasu`, `eurl`, `sa`, `sci`) and region names to `normalize.py`.
- **Check:** print ~50 French test candidates with their scores and confirm by eye that matches look right. No labels exist for France, so a manual look is the only check.

### Step H: Optional GPU boost (only if time and GPU allow)
- Multilingual sentence embeddings (`intfloat/multilingual-e5-small`, MIT) for names, used as one extra cosine-similarity feature and for extra kNN candidates on Indic names. Only on Kaggle/Colab or AWS, and only if Steps B–F are done.

### Step I: Final packaging (last ~6 h before 28 Sep 00:00)
- Freeze the best configuration → full test run → validator PASS → final upload.
- Write `code/business_entity_resolution/README.md` with the exact run order (§7).
- Fill in `Documentation_template.md` (method, blocking, features, model, results).
- Build the zip in the required layout. **No new ideas in the last 6 hours.**

### Timeline (IST)
| When | Target |
|---|---|
| 25 Sep night | Step A → Submission #1 |
| 26 Sep morning | Step B (recall 99%+) → Submission #2 |
| 26 Sep afternoon | Steps C + D → Submission #3 |
| 26 Sep night | Step E, first round |
| 27 Sep morning–afternoon | Steps E + F + G → Submissions #4–6; in the top 500 by the 48-hour mark |
| 27 Sep evening | Final improvements, then freeze by ~18:00 |
| 27 Sep 18:00–24:00 | Step I (package and final submission) |

## 7. How to run (from `code/business_entity_resolution/src`, using the `.venv` Python)

```bash
python prep.py train test          # clean -> work/*.parquet
python block.py train --sample 0.25
python features.py train_cand_sample
python train.py 0.25               # prints CV macro F0.5 + best threshold
python block.py test
python features.py test_cand
python predict.py                  # -> output/matching_results.tsv, candidate_pairs.tsv
python ../../../student_resource/utils/validate_submission.py \
  --matching ../../../output/matching_results.tsv \
  --candidate ../../../output/candidate_pairs.tsv \
  --test-dir ../../../student_resource/dataset/test
```
