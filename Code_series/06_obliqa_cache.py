# %% [markdown]
# # STEP 06 - ObliQA: run retrieval ONCE and cache it
# **What this does:** for all 2,786 ObliQA test questions it stores the BGE top-100, BM25 top-100 and, for every candidate passage,
# the best cross-encoder sentence score of that passage. Step 07 then derives Table 4 and all ablations from this cache.
# The first 200 questions additionally get their +-1 neighbours scored (needed only for the "window-max ranking" ablation).
#
# **Needs:** step 01 Part B (ObliQA index in `ob_artifacts/`).  **Run on:** Kaggle GPU T4, Internet ON.
# **Time:** ~40-70 min, resumable. **Writes:** `ob_cache_0000.pkl ...`  (28 shards).

# %%
#!pip install -q faiss-cpu rank-bm25 sentence-transformers nltk
from adsir_common import *
import faiss, nltk, torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
bge = SentenceTransformer(BGE_NAME, device=dev); ce = CrossEncoder(CE_NAME, device=dev)

OB = WORK / "ob_artifacts"
op = pickle.load(open(OB / "clean_passages_obliqua.pkl", "rb"))
ob_index = faiss.read_index(str(OB / "faiss_index_obliqua.bin"))
ob_bm25 = pickle.load(open(OB / "bm25_obliqua.pkl", "rb"))
test_data = json.load(open(OB / "ObliQA_test.json"))
assert len(op) == ob_index.ntotal
print("passages", len(op), "| questions", len(test_data))

# %% [markdown]
# ### Structure helpers: passage ids, neighbours, sentence splitting (cached)

# %%
ob_pid = [(p["DocumentID"], p["PassageID"]) for p in op]
ob_pid2idx = {}
for i, pid in enumerate(ob_pid): ob_pid2idx.setdefault(pid, i)
ob_docs = {}
for i, p in enumerate(op): ob_docs.setdefault(p["DocumentID"], []).append(i)
ob_pos = {i: (d, k) for d, ids in ob_docs.items() for k, i in enumerate(ids)}
def neighbours(i):
    d, k = ob_pos[i]; ids = ob_docs[d]; return [ids[j] for j in (k - 1, k + 1) if 0 <= j < len(ids)]
_sc = {}
def sents_of(i):
    if i not in _sc: _sc[i] = sent_tokenize(op[i]["Passage"])
    return _sc[i]

# %% [markdown]
# ### One question: dense + sparse top-100, CE-score every sentence of every candidate passage

# %%
def process_ob(qi, item, need_window):
    query = item["Question"]
    qe = bge.encode([query], normalize_embeddings=True).astype(np.float32)
    _, bi = ob_index.search(qe, PRE_K); bi = [int(x) for x in bi[0]]
    bm = [int(x) for x in np.argsort(ob_bm25.get_scores(tokenize(query)))[::-1][:PRE_K]]
    need = set(bi) | set(bm)
    if need_window:                                     # neighbours of the RRF top-50 (window-max ablation only)
        for pid, _ in rrf_fuse([bi, bm], key_fn=lambda i: ob_pid[i])[:50]:
            j = ob_pid2idx.get(pid)
            if j is not None: need.update(neighbours(j))
    need = sorted(need); pairs, owner = [], []
    for i in need:
        for s in sents_of(i): pairs.append([query, s]); owner.append(i)
    sc = np.asarray(ce.predict(pairs, batch_size=256, show_progress_bar=False), dtype=np.float32) if pairs else np.zeros(0, np.float32)
    mx = {}
    for i, v in zip(owner, sc): mx[i] = max(mx.get(i, -1e9), float(v))      # passage score = best sentence score
    return dict(q=qi, bge=bi, bm=bm, ce_max=mx, n_sents={i: len(sents_of(i)) for i in need},
                gt=[(p["DocumentID"], p["PassageID"]) for p in item["Passages"]])

# %%
SHARD, t0 = 100, time.time()
for start in range(0, len(test_data), SHARD):
    path = WORK / f"ob_cache_{start:04d}.pkl"
    if path.exists(): continue
    out = [process_ob(i, test_data[i], need_window=(i < 200)) for i in range(start, min(start + SHARD, len(test_data)))]
    pickle.dump(out, open(path, "wb"))
    print(f"shard {start:4d}-{start+len(out)-1:4d} | {time.time()-t0:6.0f}s | avg scored passages/question {np.mean([len(o['ce_max']) for o in out]):.0f}")
print("shards:", len(glob.glob(str(WORK / 'ob_cache_*.pkl'))), "(expect 28)")
