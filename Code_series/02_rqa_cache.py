# %% [markdown]
# # STEP 02 - RegulationQA: run retrieval ONCE and cache everything
# **What this does:** for each of the 1,002 questions it runs the retrieval pipeline up to cross-encoder scoring and stores the
# result. Every later RegulationQA step (03, 04, 05, 08, 09, 11) works from this cache, so the slow GPU part is done once.
#
# **Pipeline per question** (paper Section "Method"):
# 1. BGE (FAISS inner product on normalised vectors) top-100 and BM25 top-100.
# 2. Reciprocal Rank Fusion, k = 60, accumulated per passage (doc, chunk_idx); keep the top m = 20 fused passages.
# 3. For each passage: sentences of the passage itself ("own") and of its +-1 neighbour window ("win") via NLTK.
# 4. Cross-encoder (ms-marco-MiniLM-L-6-v2) scores every unique sentence against the question.
#
# **Cache content per question:** `sents` (unique sentences), `ce` (their scores), `cands` (per passage: own / window sentence ids),
# `bge10`, `bm10` (top-10 chunk ids of each single retriever, used for the full-chunk baselines).
#
# **Run on:** Kaggle, GPU T4, Internet ON. **Time:** ~25-35 min. Resumable (existing shards are skipped).
# **Writes:** `rqa_cache_0000.pkl ... rqa_cache_1000.pkl` (11 shards, in the output folder)

# %%
#!pip install -q faiss-cpu rank-bm25 sentence-transformers nltk
from adsir_common import *
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)

CACHE_PREFIX = "rqa_cache"
M = M_RQA                      # candidate pool size; change to 50 to test the repo default (writes under another prefix!)
if M != 20: CACHE_PREFIX = f"rqa_cache_m{M}"

R = load_rqa_resources(load_models=True)
chunks, index, bm25, bge, ce = R["chunks"], R["index"], R["bm25"], R["bge"], R["ce"]
pid_to_idx, window_text = R["pid_to_idx"], R["window_text"]
rqa = load_regulationqa()
print("questions:", len(rqa))

# %% [markdown]
# ### Retrieval + scoring of one question

# %%
def rrf_top(query):
    qe = bge.encode([query], normalize_embeddings=True).astype(np.float32)
    _, bi = index.search(qe, PRE_K); bi = bi[0]                              # dense top-100
    bm_i = np.argsort(bm25.get_scores(tokenize(query)))[::-1][:PRE_K]        # BM25 top-100
    fused = rrf_fuse([list(bi), list(bm_i)], key_fn=lambda i: (chunks[i]["doc_name"], chunks[i]["chunk_idx"]))
    return fused[:M], [int(x) for x in bi[:10]], [int(x) for x in bm_i[:10]]

def process(query):
    top, bge10, bm10 = rrf_top(query)
    sents, uniq, cands, seen = [], {}, [], set()
    def sid(s):                                    # give every distinct sentence one id, so the CE scores it only once
        if s not in uniq: uniq[s] = len(sents); sents.append(s)
        return uniq[s]
    for pid, rrf_score in top:
        if pid in seen: continue
        seen.add(pid)
        idx = pid_to_idx.get(pid)
        if idx is None: continue
        c = chunks[idx]
        own = sent_tokenize(c["text"])
        if not own: continue
        win = sent_tokenize(window_text(idx))
        cands.append(dict(doc=c["doc_name"], chunk_idx=c["chunk_idx"], idx=idx, rrf=float(rrf_score),
                          own=[sid(s) for s in own], win=[sid(s) for s in win]))
    scores = ce.predict([[query, s] for s in sents], batch_size=128, show_progress_bar=False) if sents else []
    return dict(sents=sents, ce=np.asarray(scores, dtype=np.float32), cands=cands, bge10=bge10, bm10=bm10)

# %% [markdown]
# ### Run all 1,002 questions in shards of 100 (safe to stop and re-run)

# %%
SHARD, t0 = 100, time.time()
for start in range(0, len(rqa), SHARD):
    path = WORK / f"{CACHE_PREFIX}_{start:04d}.pkl"
    if path.exists(): print("skip", path.name); continue
    out = []
    for i in range(start, min(start + SHARD, len(rqa))):
        r = process(str(rqa.iloc[i]["question"])); r["row"] = i; out.append(r)
    pickle.dump(out, open(path, "wb"))
    print(f"shard {start:4d}-{start+len(out)-1:4d} saved | {time.time()-t0:6.0f}s | avg unique sentences/question {np.mean([len(o['sents']) for o in out]):.0f}")
print("shards on disk:", len(glob.glob(str(WORK / f'{CACHE_PREFIX}_*.pkl'))), "(expect 11)")
