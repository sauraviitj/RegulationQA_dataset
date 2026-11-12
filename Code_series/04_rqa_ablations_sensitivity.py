# %% [markdown]
# # STEP 04 - RegulationQA ablations and sensitivity (paper Tables: window ablation, delta sensitivity, extraction strategy)
# **What this does** (all offline from the step-02 cache):
# 1. **Window ablation**: oracle best-of-3 F1 / R@0.3 / R@0.5 with the +-1 window vs. without (own passage only).
# 2. **Delta sensitivity** (tau = 0): how many words the MULTI-sentence span has for delta = 1, 2, 3, 5, 10 and what F1 it gets.
#    delta changes nothing for single-sentence outputs - it only matters when several sentences are kept (see README).
# 3. **Sentence selection vs contiguous span**: best-of-3 under the repo rule, with the kept sentences joined as-is
#    vs. the whole stretch from the first to the last kept sentence in the window.
#
# **Important:** these ablations use the repo rule (`results_for`, kept sentences) because the question they ask is about the
# repo's rule. They are re-implementations on the step-02 cache, so the values can differ slightly from the numbers in the paper
# that came from the original run. Write down which one you quote.
#
# **Run on:** CPU, ~2 min. **Writes:** `rqa_ablations.json`.

# %%
from adsir_common import *
cache = load_cache("rqa_cache"); rqa = load_regulationqa()
assert len(cache) == 1002, "run step 02 first"
gts = [str(g) for g in rqa["ground_truth_answer"]]
PAPER_ABL = {"with window": (0.3623, 0.4062, 0.2764), "without window": (0.3551, 0.3962, 0.2685),
             "sentence selection": 0.3696, "contiguous span": 0.3925}

def best3_f1(q, gt, **kw):
    """Oracle best-of-3 over the kept-sentence text of the top-3 candidates (candidate level, as in the repo ablation notebooks)."""
    res = results_for(q, **kw)[:3]
    return max([token_f1(r["text"], gt) for r in res], default=0.0)

# %% [markdown]
# ### 1. Window ablation

# %%
out = {}
for name, uw in [("with window", True), ("without window", False)]:
    f = np.array([best3_f1(q, gts[q["row"]], use_window=uw) for q in cache])
    out[name] = dict(F1=f.mean(), R03=(f >= .3).mean(), R05=(f >= .5).mean())
    print(f"{name:15s} F1 {f.mean():.4f}  R@0.3 {(f>=.3).mean():.4f}  R@0.5 {(f>=.5).mean():.4f}   | paper {PAPER_ABL[name]}")

# %% [markdown]
# ### 2. Delta sensitivity (tau = 0, window on)

# %%
sens = {}
for d in [1.0, 2.0, 3.0, 5.0, 10.0]:
    f, w = [], []
    for q in cache:
        res = results_for(q, tau=0.0, delta=d)[:3]
        best = max(res, key=lambda r: token_f1(r["text"], gts[q["row"]])) if res else None
        f.append(token_f1(best["text"], gts[q["row"]]) if best else 0.0); w.append(len(best["text"].split()) if best else 0)
    sens[d] = dict(F1=float(np.mean(f)), avg_words=float(np.mean(w)))
    print(f"delta={d:4.1f}  best-of-3 F1 {np.mean(f):.4f}  avg words {np.mean(w):6.1f}")

# %% [markdown]
# ### 3. Sentence selection vs contiguous span

# %%
def contiguous_text(q, cand_key):
    """Whole stretch of the window from the first kept sentence to the last kept sentence (fills the gaps between them)."""
    c = next(c for c in q["cands"] if (c["doc"], c["chunk_idx"]) == cand_key)
    S, C = q["sents"], q["ce"]
    kept = {t for _, t in inner_decay([(float(C[j]), S[j]) for j in c["win"]], TAU_IN, DELTA_IN)}
    pos = [i for i, j in enumerate(c["win"]) if S[j] in kept]
    return " ".join(S[j] for j in c["win"][min(pos):max(pos) + 1]) if pos else ""

f_sent, f_cont = [], []
for q in cache:
    gt = gts[q["row"]]; res = results_for(q)[:3]
    if not res: f_sent.append(0.0); f_cont.append(0.0); continue
    best = max(res, key=lambda r: token_f1(r["text"], gt))
    f_sent.append(token_f1(best["text"], gt)); f_cont.append(token_f1(contiguous_text(q, (best["doc"], best["chunk_idx"])), gt))
out["sentence_selection_F1"], out["contiguous_F1"] = float(np.mean(f_sent)), float(np.mean(f_cont))
print(f"sentence selection F1 {np.mean(f_sent):.4f} (paper 0.3696) | contiguous span F1 {np.mean(f_cont):.4f} (paper 0.3925)")
print("BERTScore for these two variants is in step 05 (needs the GPU).")
save_json(dict(window=out, delta_sensitivity=sens), "rqa_ablations.json")
