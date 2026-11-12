# %% [markdown]
# # STEP 10 - Human evaluation statistics (paper: annotator agreement, Cohen's kappa)
# **What this does:** two annotators scored the same spans as 0 (wrong), 0.5 (partial) or 1 (correct). This step reads the annotation
# file and computes, per condition (best-of-3, top-1): % correct-or-partial for each annotator, the combined mean score,
# unweighted Cohen's kappa and linearly-weighted kappa.
#
# **Input file:** `human_annotations.csv` (also accepted: `human_eval.csv`, `annotations.csv`) with columns
# `condition`, `annotator`, `score`, `question_id`  (one row per annotator x condition x question).
# **Paper values:** best-of-3 kappa 0.711 (weighted 0.798), top-1 kappa 0.648 (weighted 0.739);
# annotator correct-or-partial 65.0% and 48.4% (best-of-3), 65.0% and 44.8% (top-1).
#
# **Honest note:** the historical annotation file is NOT in the released repo. If you do not have it, the cell below stops with a clear
# message instead of inventing numbers; Part B writes a blank annotation sheet if you want to annotate again.
#
# **Run on:** CPU, seconds. **Writes:** `human_eval_summary.csv`.

# %%
from adsir_common import *
from sklearn.metrics import cohen_kappa_score
path = None
for n in ["human_annotations.csv", "human_eval.csv", "annotations.csv"]:
    path = find(n, required=False)
    if path: break

def summary(df):
    need = {"condition", "annotator", "score", "question_id"}
    if need - set(df.columns): raise ValueError(f"missing columns: {sorted(need - set(df.columns))}")
    ann = sorted(df["annotator"].unique()); assert len(ann) == 2, f"expected 2 annotators, found {len(ann)}"
    a1, a2 = ann; enc = {0.0: 0, 0.5: 1, 1.0: 2}; rows = []
    for cond in sorted(df["condition"].unique()):
        piv = df[df.condition == cond].pivot_table(index="question_id", columns="annotator", values="score", aggfunc="first").dropna()
        x, y = piv[a1].astype(float).to_numpy(), piv[a2].astype(float).to_numpy()
        xi, yi = np.array([enc[v] for v in x]), np.array([enc[v] for v in y])
        rows.append(dict(condition=cond, n_pairs=len(piv), ann1_correct_or_partial_pct=100 * (x >= .5).mean(), ann2_correct_or_partial_pct=100 * (y >= .5).mean(),
                         combined_mean=float((x.mean() + y.mean()) / 2), cohen_kappa=cohen_kappa_score(xi, yi), weighted_kappa=cohen_kappa_score(xi, yi, weights="linear")))
    return pd.DataFrame(rows)

if path:
    out = summary(pd.read_csv(path)); print(out.round(3).to_string()); out.to_csv(WORK / "human_eval_summary.csv", index=False)
else:
    print("No annotation file found - nothing to compute. The paper values above come from the original annotation sheet.")

# %% [markdown]
# ### Part B (optional) - write a blank annotation sheet from the saved run
# 100 questions (seed 42); for each, the best-of-3 span and the top-1 span, ready for two annotators to fill the `score_*` columns.

# %%
MAKE_SHEET = False
if MAKE_SHEET:
    ex = pd.read_csv(find("extraction_results.csv")); idx = np.random.default_rng(42).choice(len(ex), 100, replace=False)
    sheet = pd.DataFrame(dict(question_id=idx, question=ex.question.values[idx], reference=ex.ground_truth.values[idx],
                              span_best3=ex.best_span.values[idx], span_top1=ex.top1_span.values[idx], score_best3="", score_top1=""))
    sheet.to_csv(WORK / "annotation_sheet_blank.csv", index=False); print("wrote annotation_sheet_blank.csv")
