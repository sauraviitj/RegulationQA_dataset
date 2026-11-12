# %% [markdown]
# # STEP 03 - RegulationQA main results (paper Table 2 / span-extraction results)
# **What this does:** from the cache of step 02 it forms the three reported span modes, scores them with token F1,
# R@0.3 and R@0.5, splits single-source vs cross-document, and compares with the SAVED submitted run (`extraction_results.csv`).
#
# **The three modes**
# * **top-1**: the span of the highest-ranked candidate (no use of the reference answer).
# * **query-guided**: the highest cross-encoder sentence among the top-3 candidates (no use of the reference).
# * **oracle best-of-3**: the sentence with the best token F1 among the top-3 candidates. This USES the reference answer; it is an
#   upper bound, not a deployable system. State it as such everywhere.
#
# **Two extraction rules are computed, and the difference matters (see README "honest note")**
# * `saved_run_rule`: ONE best-scoring sentence of the +-1 window per candidate. This reproduces ~88% of the saved top-1 spans.
# * `repo_rule`: every sentence kept by inner decay (tau = 0, delta = 2) joined into one span. This is what `src/pipeline.py` returns.
#
# **Run on:** CPU, offline, ~1 min (needs the cache from step 02). **Writes:** `rqa_per_question.csv`, `rqa_summary.json`.

# %%
from adsir_common import *
cache = load_cache("rqa_cache"); rqa = load_regulationqa()
assert len(cache) == 1002, f"cache has {len(cache)} questions - run step 02 first"
SAVED = find("extraction_results.csv", required=False)          # the submitted run's outputs (in the repo's evaluation/ folder)
PAPER = dict(top1=0.2994, qg=0.3083, orc=0.3737)                # numbers printed in the paper, for the side-by-side check

# %% [markdown]
# ### Score every question under both rules

# %%
rows = []
for q in cache:
    r = rqa.iloc[q["row"]]; gt = str(r["ground_truth_answer"])
    saved_rule = span_modes_saved_run(q, gt)
    repo_rule  = span_modes_repo(results_for(q), gt)
    d = dict(row=q["row"], question=r["question"], source=r["source"], is_cross=r["is_cross"], gt=gt)
    for tag, m in [("sv", saved_rule), ("rp", repo_rule)]:
        for k in ("top1", "qg", "orc"):
            d[f"{tag}_{k}"] = m[k]; d[f"{tag}_{k}_f1"] = token_f1(m[k], gt)
    rows.append(d)
df = pd.DataFrame(rows); df.to_csv(WORK / "rqa_per_question.csv", index=False)

def report(tag, title):
    print(f"\n=== {title} ===")
    out = {}
    for k, name in [("top1", "top-1"), ("qg", "query-guided"), ("orc", "oracle best-of-3")]:
        f = df[f"{tag}_{k}_f1"]; w = df[f"{tag}_{k}"].str.split().str.len().mean()
        out[k] = dict(F1=f.mean(), R03=(f >= .3).mean(), R05=(f >= .5).mean(), words=w,
                      F1_single=df[df.is_cross == 0][f"{tag}_{k}_f1"].mean(), F1_cross=df[df.is_cross == 1][f"{tag}_{k}_f1"].mean())
        print(f"{name:18s} F1 {f.mean():.4f} (paper {PAPER[k]})  R@0.3 {(f>=.3).mean():.4f}  R@0.5 {(f>=.5).mean():.4f}  words {w:5.1f}"
              f"  | single {out[k]['F1_single']:.4f}  cross {out[k]['F1_cross']:.4f}")
    return out
summary = dict(saved_run_rule=report("sv", "saved_run_rule: best sentence of the window per candidate"),
               repo_rule=report("rp", "repo_rule: all inner-decay sentences joined (tau=0, delta=2)"))

# %% [markdown]
# ### Compare with the saved submitted run
# Expect: top-1 spans identical for about 88% of questions under `saved_run_rule`. The remaining ~12% cannot be traced to a
# rule; they come from small retrieval differences (models/index state at the time). Report it exactly like that.

# %%
if SAVED:
    ex = pd.read_csv(SAVED)
    same_order = (ex["question"].values == df["question"].values).mean()
    eq = lambda a, b: set(str(a).lower().split()) == set(str(b).lower().split())
    same_top1 = np.mean([eq(a, b) for a, b in zip(ex["top1_span"], df["sv_top1"])])
    same_orc  = np.mean([abs(float(f) - g) < 1e-3 for f, g in zip(ex["f1"], df["sv_orc_f1"])])
    print(f"\nsame question order {same_order:.3f} | identical top-1 spans {same_top1:.3f} | per-question best-of-3 F1 equal {same_orc:.3f}")
    print(f"saved file: mean best-of-3 F1 = {ex['f1'].mean():.4f}")
    summary["match_with_saved"] = dict(top1_identical=same_top1, best3_f1_equal=same_orc, saved_best3_f1=float(ex["f1"].mean()))
else:
    print("extraction_results.csv not found - skipping the comparison")

# %% [markdown]
# ### Per-instrument table (paper per-source analysis) - oracle best-of-3 and top-1

# %%
per_src = df.groupby("source").agg(n=("row", "count"), top1_F1=("sv_top1_f1", "mean"), best3_F1=("sv_orc_f1", "mean")).sort_values("best3_F1", ascending=False)
print(per_src.round(4).to_string()); per_src.to_csv(WORK / "rqa_per_source.csv")
save_json(summary, "rqa_summary.json")
