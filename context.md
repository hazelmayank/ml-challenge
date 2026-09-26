# Amazon ML Challenge 2026: Project Context & Handoff

_Last updated: 27 Sep 2026, ~00:40 IST_

This file is the single source of truth for the team: the problem, what has been built, what we learned, every score so far, what is running right now, and the plan to the deadline. It is written so that someone on a **new machine** (or a fresh Claude session) can continue without the chat history.

---

## 0. Quick start on a new machine

1. **Get the code:** `git clone https://github.com/hazelmayank/ml-challenge.git` (public, branch `main`). Everything in this repo is the current pipeline. Commit history = experiment history.
2. **Get the data** (not in Git, 1+ GB): either the Kaggle dataset **`mayankagarwal007/amazon-ml-2026-resources`** (mounted in Kaggle notebooks under `/kaggle/input/...`, the runner auto-detects it), or the original `student_resource.tar.gz` (on the old laptop in `Downloads/6ab10eb3b23ba_student_resource/`). Locally the pipeline expects `student_resource/dataset/{train,test}/*.tsv` next to `code/`, or set `BER_DATA`.
3. **Heavy runs happen on Kaggle** (notebook `notebook9cc53584ee` on account `mayankagarwal007`, GPU T4 ×2, Internet on). See §2 for what is running and §9 for the exact cells.
4. **Local Python env** (only for small tests / analysis): Python 3.11+, `pip install -r code/business_entity_resolution/requirements.txt` (torch/transformers/xgboost are only needed for `embed.py` / XGBoost; CPU builds work for tiny tests).
5. Things that exist **only on the old laptop** (not in Git): `output/` (v1 submission files, validated PASS), `work/` (old v1/v2 intermediate parquet), `test.md` (old chat notes), the problem-statement PDF, `.venv/`. None of these are needed to continue. A stale second checkout `OneDrive/Documents/ChatGPT/ml-challenge` (only `KAGGLE_HANDOFF.md` + `tmp/`) can be ignored.

## 1. Competition at a glance

| Item | Detail |
|---|---|
| Task | **Business entity resolution.** For every Source 1 (S1) business, find all Source 2 / Source 3 (S2/S3) records that are the same real-world business. |
| Metric | **Macro F0.5**, computed per S1 entity and then averaged. Singletons: an empty prediction scores 1.0, any predicted match scores 0.0. Precision weighs 2× recall. |
| Scored file | `matching_results.tsv` (tab-separated, one row per test S1, comma-joined IDs, empty when no matches). Uploaded on the portal. |
| Also required | `candidate_pairs.tsv`: **exactly** the candidate set the final model scored (the last filtering stage). Final matches must be a subset. |
| **Rule update (26 Sep)** | Candidate generation counts toward the final ranking: **a smaller candidate set per S1 ranks higher**, reviewed together with the code, beyond the leaderboard. |
| Validator | `student_resource/utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test` (stdlib only, ~5 min). |
| Deadline | **28 Sep 2026, 00:00 IST** (72-hour hackathon, 25–27 Sep). |
| Submission limit | **5 per day** (resets at midnight). |
| Leaderboard | Public = subset of test, refreshes ~15 min; final rank = private board. Day-1 top ranks were ~0.984–0.985. |
| Prizes | Top 50 get PPIs; top 500 at the 48-hour mark get +$100 AWS credits. |
| Rules | No external data / lookups / geocoding APIs. Models must be **MIT/Apache-2.0 and ≤ 8B params** (we use multilingual-e5-small = MIT, XGBoost = Apache-2.0, LightGBM = MIT). Country is an open set (France appears in test only; don't hard-code {US, India}). |
| Final package | zip: `output/` (2 TSVs), `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`), filled-in `Documentation_template.md`. |
| Team | "Linear_depression" (Mohammad Hifzaan Ansari + 3). |

## 2. Status right now (27 Sep ~00:40 IST)

- **v4 is running on Kaggle** in the interactive session of `notebook9cc53584ee`, one cell, started ~00:10, expected done **~04:15**. It runs the whole pipeline with the new normalisation: prep → block → embed → prune → features → train (XGBoost GPU) → test block → embed → prune → features → predict → validator.
  - Work dir `/tmp/v4` (lost when the session stops), outputs `/kaggle/working/output_v4/{matching_results,candidate_pairs}.tsv`, logs `/tmp/v4/logs/*.log`.
  - **The Kaggle session hard-stops ~07:00 IST** (12 h max; it started ~19:00). Download the outputs before then.
  - The laptop that drives it must stay awake with the tab open (sleep disabled). If the browser display freezes, the kernel keeps running; a new cell simply queues.
- **Check-ins:** after `prune.py train` (~01:30) → `print(open('/tmp/v4/logs/prune_train.log').read()[-3000:])`; after train → `.../train_1.0.log`; at the end → `validate.log`.
- **Next action when it finishes:** if validator = PASS → download `output_v4/matching_results.tsv` + `candidate_pairs.tsv` → upload `matching_results.tsv` to the portal = **submission #2** → log LB vs CV in §6 → stop the session (⏻) to save GPU quota.
- **If it crashed:** read the traceback; steps are resumable (see §5 "Resumability"), fix the code locally, push, `exec(CELLS[0], globals())` (git pull) in the notebook, rerun from the failed step.
- **GPU quota:** 30 h/week free. ~5.3 h used before v4, ~9.5 h after v4 → ~20 h left for 27 Sep (enough for 2–3 more full runs).

## 3. Compute & tooling

### Kaggle (main compute since 26 Sep evening)
- T4 ×2 (15 GB each), ~30 GB RAM, **only 4 CPU cores**, `/kaggle/working` ≈ 20 GB (persisted only in saved versions), `/tmp` ≈ 1 TB free (lost at session end). **Put work dirs in `/tmp`**, only final outputs in `/kaggle/working`.
- Session max **12 h**; the GPU quota counts the whole time a GPU session is open, even idle.
- The notebook bootstraps from the repo: `kaggle/kaggle_runner.ipynb` code cells are loaded and exec'd:
  ```python
  RUNNER = json.load(open('/kaggle/working/ml-challenge/kaggle/kaggle_runner.ipynb'))
  CELLS = [c['source'] for c in RUNNER['cells'] if c['cell_type'] == 'code']
  for src in CELLS[:4]: exec(src, globals())   # 0 clone/pull, 1 paths (auto-detect data), 2 pip + GPU check, 3 helpers
  ```
  Helpers: `step(script, *args)` (subprocess, streams + logs to `$BER_WORK/logs/`, records the commit), `fresh(*folders)` (delete), `set_paths(work, output)`, `disk()`, `seed_indic()` (copies the learned Indic dictionary into the work dir). `exec(CELLS[0], globals())` alone = `git pull`.
- Pinned pip install shows dependency-conflict warnings for unrelated preinstalled packages (jax, opencv…): harmless, steps run in subprocesses.
- GPU checks done: LightGBM `gpu` (OpenCL) works but was slower on small data; `cuda` build absent. **XGBoost `device=cuda` works and is ~8× faster than LightGBM on 4 CPU cores** (5 min vs ~40 min per fold on 46.6M pairs). e5 encoding ≈ **6,000–6,500 texts/s** on 2 T4s (DataParallel, fp16).

### GitHub
- Repo `hazelmayank/ml-challenge` (public). The laptop folder `Downloads/6ab10eb3b23ba_student_resource/` **is** the working tree; an allow-list `.gitignore` keeps data, work, output, `.venv`, the tarball and the PDF out. Only `.gitignore`, `context.md`, `kaggle/`, `code/`, `student_resource/utils/`, `student_resource/Documentation_template.md` are tracked.
- Commit messages end with a `Co-Authored-By: Claude` line.

### Old laptop (still usable for analysis)
- 16 threads, 13 GB RAM, no NVIDIA GPU, C: nearly full (~3 GB free). `.venv` has CPU torch + transformers + xgboost for local tests.
- Local mini test set for plumbing tests lived in the Claude scratchpad (not needed).

### Gotchas we hit (avoid repeating)
- **Laptop sleep** froze the Kaggle display (kernel kept running). Power setting "sleep when plugged in = Never" is now set.
- **Pasting huge cell output** (HF "Loading weights" progress bars) truncated at 50k chars → read logs with `open(log).read()[-N:]` instead. HF progress bars are now silenced in `embed.py`.
- **Kaggle runs one cell at a time.** A new cell shows a spinner while the previous one is still running; don't press stop on it (stop interrupts whatever is running). Cancelling a cell does **not** kill a `step()` subprocess → `pkill -f "python -u train.py"`.
- **Stale resumable parts** would silently be reused: now fingerprinted (see §5).
- In Git Bash heredocs on the laptop, `\\` collapses to `\` (turned `\b` into a backspace once). Build backslashes with `chr(92)` or use the Edit tool.
- AWS SageMaker quotas were 0 on the new account → abandoned (bucket `s3://amlc26-hazel.mayank` has the dataset zip; not used).

## 4. Data facts

| File | Rows |
|---|---:|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 (7,638,365 true pairs) |
| test_source1 | 1,732,544 (US 663k, India 810k, **France 259k**) |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 (test S2+S3 France: 1.43M) |

**Key findings the pipeline relies on:**
1. **Each S2/S3 record belongs to at most one S1** (max = 1 over all 7.6M links). So we solve it in reverse: for each S2/S3 record, pick its best S1 and accept it only if confident.
2. **Matched pairs always share the same country** (100%), so everything is done within country (country is just a string, so France works automatically).
3. ~**26% of S2/S3 records match no S1**. They are deliberate **decoys**: near-copies of a real S1 with one detail changed (house/unit number `2405`→`12405`, legal form `Pvt Ltd`→`LLP`, credential `MD`→`DMD`, one name word). True matches carry benign noise (`1479` for `14793`, dropped suffixes). Exact-difference features tell them apart.
4. Singletons are 5.6% of S1. Matches per S1: 0–11 (mean 3.67 when ≥ 1).
5. S1 is clean Latin text. S2/S3 have ~470k Indic-script names, accents, ~3.4% empty addresses, `null`/`<NULL>`.
6. **Synthetic, templated noise:** typos, junk (`>>`, `[Inc]`, phone numbers, `(ID: 34016)`), injected filler words, shuffled word order, `name.com` domains, aliases (`X D.B.A. Y`, `formerly`, `aka`), address variants (abbreviations, state names/codes, ordinals, reordered parts).
7. **Injected-noise discovery method (26 Sep night):** words far more frequent in S2/S3 than in S1 (per country) are injected noise or unhandled formats. Findings:
   - **India:** state names in Indic script inside addresses (`महाराष्ट्र` 457k, `दिल्ली` 288k, … 16 states, ~2M fields), garbled by transliteration to `mhaaraassttr`. Name prefixes `M/s`, `Smt`, `Shri`, `Dr`, fillers `Overseas`, `Infratech`.
   - **US:** qualifiers `Midtown/Northside/Eastgate/Greater/Southside/Riverside/Westgate/Lakeside`, `formerly`, `(ID: n)`, `PMB`/`CDP` in addresses.
   - **France (test only):** `R.` for rue in **25%** of S2/S3 addresses, departments (`Gironde`, `Nord`, `Loire-Atlantique`, `Pas-de-Calais`) instead of S1's regions in ~30%, `N°`, `Ch.`, `Crs`, `Q.`, `All.`; injected name words `Participations`, `Holding`, `SNC`, `Distribution`, `International`, `Associés`, `Développement`, `Groupe`, `Et`. French cities: Lille/Roubaix/Tourcoing/Dunkerque/Calais, Bordeaux/Mérignac/Pessac/La Teste/Lège-Cap-Ferret, Nantes/Saint-Nazaire/Pornic/La Baule.
8. **Indic name dictionary** (`learn_indic.py`, artifact `code/business_entity_resolution/artifacts/indic_dict.json`): 1,347 word mappings from 551k aligned train pairs, covering 96.4% of test Indic name words.

## 5. Pipeline v4 (`code/business_entity_resolution/src/`)

```text
prep → block → embed (GPU) → prune (GPU) → features → train (GPU) │ test: prep → block → embed → prune → features → predict → validator
```

| Step | Script | What it does |
|---|---|---|
| 1 | `normalize.py` + `prep.py` | TSV → cleaned parquet. Lowercase, transliterate (Indic names via dictionary, Indic state names in addresses via a fixed map), junk removal, legal suffixes → `legal`, generic + **per-country filler words** removed → `name_core`, `name_compact`, address abbreviations/states/ordinals (+ France-only street abbreviations and department→region) → `addr_norm`, `addr_nums`, `addr_ids`, `hnum`. ~7 min for train+test on Kaggle. |
| 2 | `block.py` | IDF-weighted token blocking within country (name words, address words, compact name, name-word pairs, address-word pairs, name word × house number; S1 df cap 150). Top 10 S1 per S2/S3 record. `--sample f` = legacy 25% sample (biased, don't use for CV). |
| 3 | `embed.py` (GPU) | multilingual-e5-small (MIT) embeddings of raw `"query: name \| address"`; per-country top-10 kNN S1 per S2/S3 record added to the candidates; `emb_sim`, `emb_gap`, `erank` for every pair. Embeddings cached in `$BER_WORK/emb_cache/` (raw text doesn't change with normalisation). `--probe N` = recall report only. |
| 4 | `prune.py` (GPU via XGBoost) | **Stage-1 model** on cheap scores (bscore, n_shared, brank, bscore_rel/gap, emb_sim/gap, erank), out-of-fold on train (two halves). Keeps each record's top-K candidates with p1 ≥ p_min; K/p_min auto-chosen as the smallest set within **0.2%** recall of the unpruned candidates (`BER_PRUNE="K,p_min"` overrides). Saves `prune.json` + `prune.xgb.ubj` for test. `p1`, `p1rank` become stage-2 features. **This pruned set is what `candidate_pairs.tsv` reports.** |
| 5 | `features.py` | ~47 features: RapidFuzz ratio/token_set/token_sort/partial/JW on names/addresses, house-number & legal-form difference detectors, unmatched name words, blocking + embedding + stage-1 scores, flags. Process pool sized to the CPUs. |
| 6 | `train.py` + `model.py` | `BER_MODEL=xgb` (XGBoost `device=cuda`, lossguide 63 leaves, lr 0.1, ≤ 2000 rounds, early stop 50) or `lgb` (LightGBM CPU, ≤ 800). 2-fold CV grouped by true S1. `train.py 1.0` = **all train records** (honest decoy density). Decision: each S2/S3 → its argmax S1 if p > t; t grid 0.05–0.95 tuned on OOF macro F0.5. Saves `model.*`, `decision.json` (threshold, cv, backend, feature list), `oof.parquet`. |
| 7 | `predict.py` | Loads the backend named in `decision.json`, scores test feature parts, writes `matching_results.tsv` and `candidate_pairs.tsv` (+ `test_scored.parquet`). |
| — | `metric.py` | Exact official macro F0.5 (per S1, singletons included). |
| — | `learn_indic.py`, `step1_inspect.py` | Indic dictionary learning; data inspection report. |

**Environment variables:** `BER_DATA` (folder with `train/`, `test/`), `BER_WORK`, `BER_OUTPUT`, `BER_MODEL` (`xgb`/`lgb`), `BER_EMB` (`0` = run without embeddings/pruning features), `BER_EMB_K`, `BER_EMB_MODEL`, `BER_PRUNE`, `BER_LGB_DEVICE`.

**Resumability:** block/embed/features write parts per chunk and skip existing parts on rerun. Parts folders carry an `input_fingerprint.txt`; if the input changed they are wiped automatically (`config.parts_dir`). `embed.py` keeps the blocking-only file as `*_block.parquet`, `prune.py` keeps the unpruned one as `*_union.parquet`, so each step can be rerun alone.

## 6. Results log (chronological)

| When (IST) | Experiment | Result |
|---|---|---|
| 25 Sep 21:00 | Blocking v1: single tokens, df ≤ 150 | recall@10 0.769 |
| 25 Sep 21:15 | Blocking v2: + word pairs + name × number tokens | recall@10 0.968, 9.9 candidates/record |
| 25 Sep ~23:15 | LightGBM v1: 23 features, 25% sample, t=0.30 | CV 0.97638 (biased sample) |
| 26 Sep ~01:00 | v1 test run: 98.4M candidate pairs (~57 per S1) → 5.91M matches, validator PASS | — |
| 26 Sep ~19:30 | Kaggle smoke test (v2, 2% sample) | CV 0.98626 (noisy) |
| 26 Sep ~20:15 | v2 on Kaggle (Indic dictionary, difference features), 25% sample, LightGBM | recall@10 0.9768, **CV 0.98386** at t=0.30 |
| 26 Sep ~20:30 | Blocking-miss analysis (v2): 36.0k of 1.90M links missed (1.9%): address missing 14.4k (18.6% miss rate), typos in both fields 7.5k, Indic not in dictionary 5.7k, domain/alias names 4.5k, address variants 2.6k, both differ 1.4k. India 3.1% vs US 1.1% | → embeddings |
| 26 Sep ~20:30 | Embedding probe (300k records): recall 0.9794 → **0.9904**; Indic 0.960 → 0.991, has address 0.989 → 0.998, no address 0.764 → 0.829 | kept |
| 26 Sep ~22:10 | Embeddings on the full 25% sample: recall@10 0.9768 → **0.9898**, 18.07 candidates/record | — |
| 26 Sep 22:31 | **Submission #1 (v1) → public LB 0.945857** vs CV 0.97638: **gap 0.030** | — |
| 26 Sep ~22:45 | **Gap diagnosed:** the 25% sample kept only 25% of decoys (4× fewer per S1 than test) and none of the other S1s' records, so CV was optimistic and t too low. Fix: `train.py 1.0` on all records + `prune.py`. France (15% of test) still unmeasurable | → v3/v4 |
| 26 Sep ~23:10 | Embeddings + **XGBoost CUDA**, 25% sample (46.6M pairs, 42 features): folds 5 min each; ceiling 0.9969; still improving at 800 rounds | **CV 0.98785** at t=0.35 (+0.004 vs 0.98386, same sample) |
| 26 Sep ~23:50 | **Normalisation v4** (data-driven, §4.7). A/B on 300k labelled train pairs: India address similarity 92.3 → 94.1, 10.7% of true pairs improved > 5 points, 0% worse; India exact name_core 68.5 → 69.6%; US small gain | in v4 run |
| 27 Sep ~00:10 | **v4 full run started** (commit 1d7f68c): new normalisation + all train records + embeddings + pruning + XGBoost | _running, ~04:15_ |

Top features (XGBoost, 25% sample): `bscore_gap`, `emb_gap`, `brank`, `ids_a_extra_fuzzy`, `bscore_rel`, `addr_ids_tset`, `addr_norm_tset`, `name_norm_tsort`, `name_a_extra`, `hnum_rel`.

**CV comparability warning:** scores from `train.py 0.25` (old sample) and `train.py 1.0` (all records) are **not** comparable; the full-data CV will be lower and closer to the leaderboard. Compare runs only within the same mode.

## 7. What was built when (commits, oldest → newest)

`3fc961d` pipeline + Kaggle runner (float32 OOM fix) → `2019045` threshold grid from 0.05 → `2fa0e37` `embed.py` + fingerprinted parts → `3054739` quiet HF logs → `8e9c8ce` `model.py` (XGBoost CUDA backend) → `5d2177d` `prune.py` + full-train mode → `daa5e2e` xgb ≤ 2000 rounds → `1d7f68c` normalisation v4 + embedding cache. (`git log` has details.)

## 8. Plan to the deadline (27 Sep)

### Working rules
1. **One change → one measurement**, always in full-data mode (`train.py 1.0`) so CV tracks the leaderboard.
2. Log every run in §6 with the commit; log CV **and** LB for every submission to track the gap.
3. **Submissions (5 today):** submit when full-data CV improves by ≥ ~0.002, or to measure the CV–LB gap.
4. Keep the candidate set small (the organisers' rule): report candidates per S1 for every test run.
5. **Freeze at 18:00 IST. No new ideas in the last 6 hours.**

### Timeline (IST)

| When | What | Output |
|---|---|---|
| ~04:15 | v4 finishes → validator → **submission #2** → stop session | LB v4, CV v4, candidates/S1 |
| Morning (~09:00–10:00) | Read v4 logs: prune table (recall kept vs size), full-data CV per country/source, CV–LB gap. **France check:** sample ~50 French matches and ~50 French rejected top candidates from `test_scored.parquet` and eyeball them (no labels) | decide next steps |
| 10:00–14:00 | **Improvement round 1** (one Kaggle run, ~2.5 h thanks to the embedding cache if run in one session; ~4 h from scratch) — candidates in priority order below | submission #3 |
| 14:00–18:00 | Improvement round 2 / ensemble | submissions #4–#5 |
| 18:00 | **Freeze** the best configuration by full-data CV (and LB) | — |
| 18:00–22:00 | Final full run (if the best run's outputs aren't already the final ones), validator, final upload | final `output/` |
| 22:00–23:30 | Package: README (done, check), fill `Documentation_template.md`, pinned requirements, zip, upload | zip |

### Improvement candidates (ranked by expected gain / cost)
1. **Decision tuning on full-data OOF** (cheap, no rerun of features): per-country and per-source thresholds; **near-tie abstain** (if the top-2 S1 probabilities of a record are close, don't match: protects precision); **singleton protection** (an S1 whose best incoming probability is low gets nothing). Implement in `train.py`/`predict.py` as extra fields in `decision.json`.
2. **LightGBM vs XGBoost vs average** on the same pruned features (pruned data is small, so LightGBM on CPU is affordable now). The winner goes into `decision.json`.
3. **S1-side context features** on the full candidate graph (now valid because train uses all records): how many S2/S3 records choose this S1 as top-1, this pair's rank among them, S1's candidate count. Helps decoys that compete with the true records for the same S1.
4. **No-address records** (recall 0.83, the weakest slice): more embedding neighbours on name-only text for `addr_missing` records, or name-only kNN.
5. **Tuning:** XGBoost depth/leaves/learning rate; `BER_EMB_K`; pruning `MAX_LOSS`.
6. **France:** after eyeballing, add rules if systematic errors show up (e.g. more department/city synonyms, `Sainte`/`Saint`, legal forms).

### Final package checklist
- [ ] `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the **same** final run (validator PASS)
- [ ] `code/business_entity_resolution/src/` = the exact commit that produced them (record the hash)
- [ ] `README.md` run order (already written for v4; re-check)
- [ ] `requirements.txt`: pin torch / transformers / xgboost to the Kaggle versions used (`pip freeze | grep -E "torch|transformers|xgboost"` in the notebook)
- [ ] `Documentation_template.md` filled: methodology, blocking + embeddings + pruning, features, model, decision rule, candidate-set size, results table from §6, compute used, model licenses
- [ ] Zip layout exactly as in §1; team name in the file name

## 9. How to run

### Kaggle, full pipeline in one cell (what v4 runs)
Needs: dataset attached, GPU T4 ×2, Internet on. In a fresh session, first bootstrap (`git clone` the repo to `/kaggle/working/ml-challenge`, load `CELLS`, `exec` `CELLS[:4]`, see §3), then:
```python
exec(CELLS[0], globals())                            # git pull
set_paths('/tmp/v4', '/kaggle/working/output_v4')
seed_indic()
os.environ['BER_MODEL'] = 'xgb'
step('prep.py', 'train', 'test')
step('block.py', 'train');  fresh('train_cand_parts')
step('embed.py', 'train');  fresh('train_cand_emb_parts')
step('prune.py', 'train')
step('features.py', 'train_cand')
step('train.py', 1.0)
step('block.py', 'test');   fresh('test_cand_parts')
step('embed.py', 'test');   fresh('test_cand_emb_parts')
step('prune.py', 'test')
step('features.py', 'test_cand')
step('predict.py')
step(VALIDATOR, '--matching', '/kaggle/working/output_v4/matching_results.tsv',
     '--candidate', '/kaggle/working/output_v4/candidate_pairs.tsv', '--test-dir', f'{DATA}/test',
     log='validate.log')
```
Approximate times: prep 10 min, block train 25, embed train 35 (cached: ~2), prune 10, features 10, train 15, block test 25, embed test 35 (cached: ~2), prune/features/predict ~25, validator 5 → **~4 h from scratch**. For an unattended run, put the bootstrap + this into a single-cell notebook and use **Save Version → Save & Run All**, then download from the version's Output tab.

### Reading logs without huge output
```python
print(open('/tmp/v4/logs/train_1.0.log').read()[-3000:])
```

### Local (small tests only)
From `code/business_entity_resolution/src/` with `BER_WORK`/`BER_DATA` set; see `code/business_entity_resolution/README.md` for the full order.
