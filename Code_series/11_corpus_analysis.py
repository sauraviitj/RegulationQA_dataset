# %% [markdown]
# # STEP 11 - Corpus analysis (paper: per-source F1, document similarity, DPDP Act -> Rules confusion, single vs cross-document)
# **What this does** (offline except the BGE centroid encoding, ~3 min on GPU):
# 1. Per-instrument F1 (oracle best-of-3 and top-1) from step 03.
# 2. Document centroids (mean of normalised BGE chunk embeddings) and the cosine-similarity matrix between instruments;
#    the 15 most similar pairs (explains retrieval confusions between sibling instruments).
# 3. DPDP Act -> DPDP Rules confusion: among DPDP-Act questions, the % whose TOP-1 document is the Rules (paper: 27.5%).
# 4. Single-source vs cross-document top-1 F1.
# **Writes:** `doc_similarity_matrix.csv`, `dpdp_confusion.csv`, `corpus_summary.json`. Interactive graphs of repo notebook 06 are
# cosmetic and live in that notebook (`pyvis`); they are not needed for any table.

# %%
from adsir_common import *
cache = load_cache("rqa_cache"); rqa = load_regulationqa(); R = load_rqa_resources(load_models=True)
chunks, bge = R["chunks"], R["bge"]; per_q = pd.read_csv(WORK / "rqa_per_question.csv")

# %% [markdown]
# ### 1-2. Per-source F1 and similarity of instruments

# %%
src = per_q.groupby("source").agg(n=("row", "count"), top1_F1=("sv_top1_f1", "mean"), best3_F1=("sv_orc_f1", "mean")).sort_values("best3_F1", ascending=False)
print(src.round(4).to_string())
emb = bge.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
cent = {}
for d, ids in R["doc_list"].items():
    v = emb[ids].mean(0); cent[d] = v / np.linalg.norm(v)
names = list(cent); M = np.array([[float(cent[a] @ cent[b]) for b in names] for a in names])
pd.DataFrame(M, index=names, columns=names).to_csv(WORK / "doc_similarity_matrix.csv")
pairs = sorted(((M[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))), reverse=True)
print("\nTop-15 most similar document pairs"); [print(f"{s:.4f}  {a[:38]} <-> {b[:38]}") for s, a, b in pairs[:15]]

# %% [markdown]
# ### 3. DPDP Act -> Rules confusion (top-1 document under the saved-run rule: candidate with the highest own-passage CE score)

# %%
def top1_doc(q):
    S, C = q["sents"], q["ce"]
    if not q["cands"]: return ""
    return max(q["cands"], key=lambda c: max(float(C[j]) for j in c["own"]))["doc"]
dp = [(q["row"], top1_doc(q)) for q in cache if per_q.loc[per_q.row == q["row"], "source"].iloc[0] == "DPDP Act 2023"]
conf = np.mean([("data protection rules" in d.lower() or "dpdp rules" in d.lower()) for _, d in dp]) if dp else float("nan")
print(f"\nDPDP Act questions: {len(dp)} | top-1 document is the DPDP Rules: {conf:.1%}   (paper: 27.5%)")
pd.DataFrame(dp, columns=["row", "top1_doc"]).to_csv(WORK / "dpdp_confusion.csv", index=False)

# %% [markdown]
# ### 4. Single-source vs cross-document

# %%
sc = per_q.groupby("is_cross").agg(n=("row", "count"), top1_F1=("sv_top1_f1", "mean"), best3_F1=("sv_orc_f1", "mean")); sc.index = ["single-source", "cross-document"]
print(sc.round(4).to_string())
save_json(dict(per_source=src.reset_index().to_dict("records"), dpdp_rules_confusion=float(conf), single_vs_cross=sc.reset_index().to_dict("records")), "corpus_summary.json")
