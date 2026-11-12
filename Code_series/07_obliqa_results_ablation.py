# %% [markdown]
# # STEP 07 - ObliQA results (paper Table 4) and ablations, with confidence intervals
# **What this does** (offline, from the step-06 cache, ~2 min):
# * **Table 4**: BGE only, BM25 only, BGE + CE, full pipeline (pool m = 20 and m = 50): R@1, R@5, R@10, MAP@10 on all 2,786 questions.
# * **Ablation**: pool size m in {1, 5, 10, 20, 30, 50, 100}; ranking by RRF instead of CE; the cross-encoder cost per pool size.
# * **Window-max** ablation on the first 200 questions (the only ones with neighbours scored).
# * **Paired bootstrap 95% CIs** (B = 2000, over questions) for the full pipeline minus each baseline.
# * **LaTeX rows** ready to paste.
#
# **Metric definitions:** a retrieved passage is correct if its (DocumentID, PassageID) is one of the question's gold passages;
# duplicates from split passages are collapsed before the top-k cut. R@k = at least one gold passage in the top k;
# MAP@10 = average precision over the top 10 divided by the number of gold passages.
# **Honest note:** several differences between the full pipeline and BM25 are small; read the CIs, not only the means.

# %%
from adsir_common import *
cache_ob = load_cache("ob_cache"); print("cached questions:", len(cache_ob))
assert [c["q"] for c in cache_ob] == list(range(len(cache_ob))), "shards missing or out of order - rerun step 06"
op = pickle.load(open(find("clean_passages_obliqua.pkl"), "rb"))
ob_pid = [(p["DocumentID"], p["PassageID"]) for p in op]
ob_pid2idx = {}
for i, pid in enumerate(ob_pid): ob_pid2idx.setdefault(pid, i)
ob_docs = {}
for i, p in enumerate(op): ob_docs.setdefault(p["DocumentID"], []).append(i)
ob_pos = {i: (d, k) for d, ids in ob_docs.items() for k, i in enumerate(ids)}
def neighbours(i):
    d, k = ob_pos[i]; ids = ob_docs[d]; return [ids[j] for j in (k - 1, k + 1) if 0 <= j < len(ids)]

# %% [markdown]
# ### Rankers and metrics

# %%
def dedup(lst):                                   # indices -> unique (doc, passage) ids, order kept
    seen, out = set(), []
    for i in lst:
        pid = ob_pid[i]
        if pid not in seen: seen.add(pid); out.append(pid)
    return out
def rrf_order(c): return rrf_fuse([c["bge"], c["bm"]], key_fn=lambda i: ob_pid[i])      # [(pid, score)] best first
def score_map(c):                                  # pid -> best own-passage CE sentence score
    m = {}
    for i, v in c["ce_max"].items():
        pid = ob_pid[i]; m[pid] = max(m.get(pid, -1e9), v)
    return m
def sort_desc(pids, key): return sorted(pids, key=lambda p: -key[p])

def metrics(ranked, gt):
    g = set(gt); hit = lambda k: float(any(p in g for p in ranked[:k]))
    ap, h = 0.0, 0
    for k, p in enumerate(ranked[:10], 1):
        if p in g: h += 1; ap += h / k
    return [hit(1), hit(5), hit(10), ap / max(len(g), 1)]
def run(ranker, qs=None):
    return np.array([metrics(ranker(c), c["gt"]) for c in (cache_ob if qs is None else [cache_ob[i] for i in qs])])

def bge_only(c): return dedup(c["bge"])[:10]
def bm_only(c):  return dedup(c["bm"])[:10]
def bge_ce(c):
    sm = score_map(c); return sort_desc([p for p in dedup(c["bge"]) if p in sm], sm)
def full(m, by="ce"):
    def f(c):
        sm = score_map(c); pool = [p for p, _ in rrf_order(c)[:m] if p in sm]
        return sort_desc(pool, sm) if by == "ce" else pool
    return f
def win_max(m):                                    # rank by the max over the passage and its two neighbours
    def f(c):
        sm = score_map(c); pool = [p for p, _ in rrf_order(c)[:m] if p in sm]
        def w(p): i = ob_pid2idx[p]; return max(c["ce_max"][j] for j in [i] + neighbours(i) if j in c["ce_max"])
        return sort_desc(pool, {p: w(p) for p in pool})
    return f
def cost(m):                                       # mean passages / sentences the CE must score for a pool of m
    P, S = [], []
    for c in cache_ob:
        pool = [ob_pid2idx[p] for p, _ in rrf_order(c)[:m] if p in ob_pid2idx]
        P.append(len(pool)); S.append(sum(c["n_sents"].get(i, 0) for i in pool))
    return np.mean(P), np.mean(S)

# %% [markdown]
# ### Table 4 (all questions)

# %%
paper4 = {"BGE only": (.4986, .7146, .7778, .5400), "BM25 only": (.5696, .7545, .8008, .5808),
          "BGE + CE": (.5754, .7373, .7778, .5841), "Full pipeline, pool m=50": (.5750, .7624, .8029, .5915)}
T4 = {"BGE only": run(bge_only), "BM25 only": run(bm_only), "BGE + CE": run(bge_ce),
      "Full pipeline, pool m=50": run(full(50)), "Full pipeline, pool m=20": run(full(20))}
print(f"{'system':28s} R@1 R@5 R@10 MAP@10   (ours | paper)")
for k, a in T4.items():
    p = paper4.get(k); print(f"{k:28s} " + " ".join(f"{x:.4f}" for x in a.mean(0)) + (" | " + " ".join(f"{x:.4f}" for x in p) if p else " | -"))

# %% [markdown]
# ### Ablation: pool size and ranking signal

# %%
ABL = {}
for m in [1, 5, 10, 20, 30, 50, 100]:
    a = run(full(m)); ABL[f"RRF top-{m} -> CE"] = a; P, S = cost(m)
    print(f"m={m:3d}  " + " ".join(f"{x:.4f}" for x in a.mean(0)) + f"   passages {P:5.1f}  sentences {S:7.0f}")
ABL["RRF order (no CE), m=50"] = run(full(50, by="rrf"))
print("no CE (RRF order, m=50) " + " ".join(f"{x:.4f}" for x in ABL["RRF order (no CE), m=50"].mean(0)))
q200 = list(range(min(200, len(cache_ob)))); a_own, a_win = run(full(50), q200), run(win_max(50), q200)
print("first 200: own-passage ranking", " ".join(f"{x:.4f}" for x in a_own.mean(0)), "| window-max ranking", " ".join(f"{x:.4f}" for x in a_win.mean(0)))

# %% [markdown]
# ### Confidence intervals (paired bootstrap over questions)

# %%
names = ["R@1", "R@5", "R@10", "MAP@10"]
for pool_name, F in [("m=20", T4["Full pipeline, pool m=20"]), ("m=50", T4["Full pipeline, pool m=50"])]:
    print(f"\n--- Full pipeline ({pool_name}) minus ... ---")
    for nm in ["BM25 only", "BGE only", "BGE + CE"]:
        for j in range(4):
            d, lo, hi = paired_bootstrap(F[:, j], T4[nm][:, j])
            print(f"vs {nm:10s} {names[j]:7s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("   <- CI includes 0" if lo <= 0 <= hi else ""))

# %% [markdown]
# ### LaTeX rows for the paper tables

# %%
def row(label, a, bold=None):
    cells = [f"\\textbf{{{x:.4f}}}" if bold is not None and abs(x - bold[j]) < 1e-9 else f"{x:.4f}" for j, x in enumerate(a.mean(0))]
    return f"{label:<38s} & " + " & ".join(cells) + " \\\\"
tab4 = {"BGE only": T4["BGE only"], "BM25 only": T4["BM25 only"], "BGE + CE": T4["BGE + CE"], "Full pipeline ($m$=20)": T4["Full pipeline, pool m=20"]}
best = np.max([a.mean(0) for a in tab4.values()], axis=0)
for k, a in tab4.items(): print(row(k, a, best))
print()
for k, a in [("Full pipeline ($m$=50)", T4["Full pipeline, pool m=50"]), ("Rank by RRF, not CE ($m$=50)", ABL["RRF order (no CE), m=50"]),
             ("Fixed top-$k$=1", ABL["RRF top-1 -> CE"]), ("Fixed top-$k$=5", ABL["RRF top-5 -> CE"]), ("Fixed top-$k$=10", ABL["RRF top-10 -> CE"])]: print(row(k, a))
pd.DataFrame({f"{k} | {n}": v[:, j] for k, v in {**T4, **ABL}.items() for j, n in enumerate(names)}).to_csv(WORK / "obliqa_per_question.csv", index=False)
