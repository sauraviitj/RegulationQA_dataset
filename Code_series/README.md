# ADSIR / RegulationQA - clean code series

One numbered series that covers the whole project. Run the steps in order; every step says what it reads, what it writes and which
part of the paper it feeds. Shared settings and helper functions live in ONE file, `adsir_common.py`.

## Files
| File | Purpose | Paper link | Where to run |
|---|---|---|---|
| `adsir_common.py` | paths, constants (k=60, m=20, tau=0, delta=2 ...), token F1, inner decay, RRF, label normalisation, bootstrap | all | imported by every step |
| `01_build_indexes` | (optional) rebuild RegulationQA index; build ObliQA index | Setup | Kaggle GPU, internet ON, ~10 min |
| `02_rqa_cache` | run retrieval + cross-encoder once for the 1,002 questions, cache | prerequisite | Kaggle GPU, ~30 min |
| `03_rqa_span_extraction` | top-1 / query-guided / oracle best-of-3, F1, R@0.3, R@0.5, per source, single vs cross, comparison with the saved run | main RegulationQA table, per-source | CPU, 1 min |
| `04_rqa_ablations_sensitivity` | window ablation, delta sensitivity, sentence vs contiguous span | ablation + sensitivity tables | CPU, 2 min |
| `05_bertscore_and_length_controls` | BERTScore (roberta-large), length-matched controls, bootstrap CIs | BERTScore table + length control | Kaggle GPU, ~10 min |
| `06_obliqa_cache` | ObliQA retrieval + CE scores for 2,786 questions | prerequisite | Kaggle GPU, 40-70 min |
| `07_obliqa_results_ablation` | ObliQA Table 4, pool-size ablation, CIs, LaTeX rows | ObliQA table + ablation | CPU, 2 min |
| `08_llm_judges` | Qwen3-4B + Mistral-7B judges, kappa | LLM-judge table | Kaggle GPU, 1-3 h |
| `09_hallucination_nli` | full chunk vs span generation, DeBERTa NLI | hallucination table (re-run) | Kaggle GPU, ~25 min |
| `10_human_evaluation` | annotator % and Cohen's kappa from the annotation file | human-eval table | CPU |
| `11_corpus_analysis` | per-instrument F1, document similarity, DPDP Act->Rules confusion | analysis section | Kaggle GPU (encoding), 3 min |
| `make_notebooks.py` | turns the `.py` files into `.ipynb` (already done: see `notebooks/`) | - | anywhere |

Each `.py` is written as `# %%` cells, so it works as a plain script, in VS Code / Spyder as a notebook, and as the ready-made
`.ipynb` in `notebooks/`.

## Setup on Kaggle (once)
1. New notebook -> Accelerator **GPU T4**, Internet **ON**.
2. Add as input datasets: the released repo folders `index/` (chunks.pkl, faiss_index.bin, bm25.pkl), `dataset/` (regulationqa_v1.csv),
   `evaluation/` (extraction_results.csv, extraction_log.json, bge_extraction_log.json, bm25_extraction_log.json), plus `adsir_common.py`
   (upload it as a tiny dataset). `find()` searches `/kaggle/input` recursively, so folder names do not matter.
3. Everything the steps create is written to `/kaggle/working` (locally: `./outputs`). Download it before the session ends.

Order: `01B -> 02 -> 03 -> 04 -> 05`, `06 -> 07`, and independently `08`, `09` (after 03), `10`, `11` (after 03).

## Which numbers come from where (read this before editing the paper)
| Paper number | Produced by | Comes from |
|---|---|---|
| RegulationQA top-1 / query-guided / best-of-3 F1, BERTScore | submitted run (saved in `evaluation/extraction_results.csv`); step 03 re-derives them | re-implementation reproduces ~88% of saved top-1 spans |
| Window ablation, delta sensitivity, sentence vs contiguous | step 04 | re-implementation on the step-02 cache (m = 20) |
| BERTScore + length controls | step 05 | saved spans + baselines |
| ObliQA Table 4, ablation, CIs | steps 06-07 | fresh run on all 2,786 questions |
| LLM judges | step 08 | saved best-of-3 spans, fresh judge run |
| Hallucination NLI | step 09 | fresh re-run (original generations were not recovered) |
| Human evaluation | step 10 | needs the original annotation CSV (not in the repo) |

## Honest notes (keep these consistent with the paper)
1. **The extraction rule.** The saved RegulationQA spans are single sentences: for each candidate (ranked by the best cross-encoder
   score of its own passage) the best-scoring sentence of its +-1 window. Best-of-3 picks, WITH the reference, the best of the three.
   `src/pipeline.py` instead joins every sentence kept by inner decay. Step 03 computes both and labels them `saved_run_rule` / `repo_rule`.
2. **delta.** The repo default is delta = 2.0 (the old reconstruction notebook says 5.0). delta only changes outputs that keep
   several sentences (query-guided pool, multi-sentence variant, step 04 sensitivity); it does not change the single-sentence top-1.
3. **About 12% of saved top-1 spans cannot be reproduced** exactly; they trace to small retrieval-pool differences of the original run.
   Say "about 88% of the saved top-1 spans are reproduced exactly", not "fully reproducible".
4. **m = 20 vs 50.** The paper uses a pool of 20; the repo `evaluate.py` default is `top_n = 50`; the ObliQA notebooks use 50. Step 07 reports both.
5. **Oracle numbers use the reference answer** (best-of-3, and the judged `best_span`). They are upper bounds, not deployable accuracy.
6. **Steps 09 and 10** cannot recover the original files; they either re-run (09) or need the original CSV (10). Do not mix re-run numbers
   and original numbers in one table.
7. **Label normalisation** (`SOURCE_CANON` in `adsir_common.py`) maps the 40 raw `document_source` strings to 13 groups
   (12 single-source instruments + cross-document). The assert in `load_regulationqa()` checks 607 single-source / 395 cross-document.
   Check the ISO/IEC 27001 label against the paper before submission (the raw labels use "ISO 27001:2022" and "ISO/IEC 27001:2022").

## Common problems
* `FileNotFoundError: ... not found` -> add the dataset that contains that file as notebook input, or run the step that creates it.
* `cache has N questions - run step 02 first` -> step 02 was interrupted; re-run it (finished shards are skipped).
* `Unmapped document_source label` -> a new label appeared; add one line to `SOURCE_CANON`.
* Out of GPU memory in step 08 -> `SINGLE_GPU` is automatic; restart the kernel between the Qwen and Mistral parts if needed.
