# %% [markdown]
# # STEP 05 - BERTScore and length controls (paper Table: BERTScore, and the "does BERTScore just reward length?" control)
# **What this does:** scores ADSIR spans and several baselines with BERTScore (roberta-large, raw F1, no rescaling) AND token F1,
# then repeats the baselines cut to the SAME length as the ADSIR top-1 span (first-N words and a random N-word window).
# If the length-matched baselines score as high as ADSIR, BERTScore would only be measuring length. Paired bootstrap 95% CIs
# (B = 2000) give the difference ADSIR minus each control.
#
# **Which ADSIR spans?** By default the SAVED submitted-run spans (`extraction_results.csv`: `top1_span`, `best_span`), because
# those are the spans the paper's numbers came from. If that file is absent it falls back to the spans re-computed in step 03.
#
# **Run on:** Kaggle GPU T4 (roberta-large), Internet ON. **Time:** ~10 min. **Writes:** `bertscore_controls.csv`, `bertscore_per_question.csv`.

# %%
#!pip install -q bert-score
from adsir_common import *
from bert_score import score as bscore
import torch
cache = load_cache("rqa_cache"); rqa = load_regulationqa()
R = load_rqa_resources(load_models=False); chunks = R["chunks"]
gts = [str(g) for g in rqa["ground_truth_answer"]]
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = random.Random(0)

SAVED = find("extraction_results.csv", required=False)
if SAVED:
    ex = pd.read_csv(SAVED); top1 = [str(s) for s in ex["top1_span"]]; best3 = [str(s) for s in ex["best_span"]]
    print("using the SAVED submitted-run spans")
else:
    d = pd.read_csv(WORK / "rqa_per_question.csv"); top1, best3 = d["sv_top1"].astype(str).tolist(), d["sv_orc"].astype(str).tolist()
    print("using spans recomputed in step 03")
assert len(top1) == len(best3) == len(cache) == len(gts)

# %% [markdown]
# ### Build the systems. Baselines: top-1 chunk of BGE alone / BM25 alone (about 500 words each), plus length-matched cuts.

# %%
N = [max(5, len(s.split())) for s in top1]                       # per-question length of the ADSIR top-1 span
bge_chunk = [chunks[q["bge10"][0]]["text"] for q in cache]
bm_chunk  = [chunks[q["bm10"][0]]["text"] for q in cache]
first_n = lambda t, n: " ".join(t.split()[:n])
def rand_n(t, n):
    w = t.split()
    if len(w) <= n: return t
    s = rng.randrange(0, len(w) - n + 1); return " ".join(w[s:s + n])
systems = {
    "ADSIR top-1 span": top1, "ADSIR best-of-3 span": best3,
    "BGE full top-1 chunk": bge_chunk, "BM25 full top-1 chunk": bm_chunk,
    "BGE first-N words (len-matched)": [first_n(t, n) for t, n in zip(bge_chunk, N)],
    "BM25 first-N words (len-matched)": [first_n(t, n) for t, n in zip(bm_chunk, N)],
    "BGE random N-word window": [rand_n(t, n) for t, n in zip(bge_chunk, N)],
}

# %% [markdown]
# ### Score (roberta-large downloads on first use)

# %%
res, per_q = [], {}
for name, cands_ in systems.items():
    cands_ = [c if c.strip() else " " for c in cands_]
    P, Rc, F = bscore(cands_, gts, model_type=BERT_MODEL, lang="en", batch_size=32, device=dev, verbose=False)
    bf = F.numpy(); tf = np.array([token_f1(c, g) for c, g in zip(cands_, gts)])
    per_q[name + " | BERTScoreF1"] = bf; per_q[name + " | tokenF1"] = tf
    res.append(dict(system=name, avg_words=np.mean([len(c.split()) for c in cands_]), BERTScore_F1=bf.mean(), BERTScore_P=P.mean().item(), BERTScore_R=Rc.mean().item(), token_F1=tf.mean()))
    print(f"{name:36s} words {res[-1]['avg_words']:6.1f}  BERTScoreF1 {bf.mean():.4f}  tokenF1 {tf.mean():.4f}")
pd.DataFrame(res).to_csv(WORK / "bertscore_controls.csv", index=False); pd.DataFrame(per_q).to_csv(WORK / "bertscore_per_question.csv", index=False)

# %% [markdown]
# ### Paired bootstrap: ADSIR top-1 minus each control

# %%
for name in systems:
    if name == "ADSIR top-1 span": continue
    for metric in ["BERTScoreF1", "tokenF1"]:
        d, lo, hi = paired_bootstrap(per_q["ADSIR top-1 span | " + metric], per_q[name + " | " + metric])
        print(f"{name:36s} {metric:12s} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

# %% [markdown]
# ### BERTScore for the extraction-strategy table (sentence selection vs contiguous span)
# Needs the per-question spans of both variants; they come from step 04's logic. Left as a short optional cell.

# %%
RUN_STRATEGY_BERT = False
if RUN_STRATEGY_BERT:
    from importlib import import_module
    sent_spans, cont_spans = [], []
    for q in cache:
        gt = gts[q["row"]]; res_q = results_for(q)[:3]
        if not res_q: sent_spans.append(" "); cont_spans.append(" "); continue
        b = max(res_q, key=lambda r: token_f1(r["text"], gt)); sent_spans.append(b["text"])
        c = next(c for c in q["cands"] if (c["doc"], c["chunk_idx"]) == (b["doc"], b["chunk_idx"]))
        kept = {t for _, t in b["kept"]}; pos = [i for i, j in enumerate(c["win"]) if q["sents"][j] in kept]
        cont_spans.append(" ".join(q["sents"][j] for j in c["win"][min(pos):max(pos) + 1]) if pos else " ")
    for nm, sp in [("Sentence selection", sent_spans), ("Contiguous span", cont_spans)]:
        _, _, F = bscore(sp, gts, model_type=BERT_MODEL, lang="en", batch_size=32, device=dev, verbose=False)
        print(f"{nm:20s} BERTScore F1 {F.mean().item():.4f}   (paper: 0.8939 / 0.8540)")
