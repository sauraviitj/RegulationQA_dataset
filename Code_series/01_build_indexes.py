# %% [markdown]
# # STEP 01 - Build the search indexes
# **What this does:** turns the regulatory documents into (a) a dense BGE FAISS index and (b) a BM25 index, for both corpora.
# **Paper link:** Section "Setup" (corpora, chunking, indexing). Nothing here produces a table; it creates the inputs of every later step.
#
# **You normally do NOT need to run Part A.** The repo already ships the finished RegulationQA index in `index/`
# (`chunks.pkl`, `faiss_index.bin`, `bm25.pkl`). Use Part A only if you want to rebuild from the cleaned text files;
# the rebuilt chunks can differ slightly from the shipped ones, so for reproducing the paper always use the SHIPPED index.
# Part B builds the ObliQA index (needed by steps 06-07) from the public ObliQA repository (internet ON).
#
# **Run on:** Kaggle, GPU T4 (BGE encoding), Internet ON.   **Time:** Part A ~3 min, Part B ~6 min.
# **Writes:** `outputs/index_rebuilt/*` (Part A), `outputs/ob_artifacts/*` (Part B).

# %%
#!pip install -q faiss-cpu rank-bm25 sentence-transformers nltk requests
from adsir_common import *
import faiss, nltk, requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
bge = SentenceTransformer(BGE_NAME, device=device)

# %% [markdown]
# ## Part A (optional) - rebuild the RegulationQA index from `documents/cleaned_texts/*.txt`
# Chunking rule (from repo notebook 01): 500-word windows, stride 400 words (100-word overlap); a final chunk shorter than
# 100 characters is dropped. `doc_name` is the file name without `.txt`, with underscores turned into spaces for display only.

# %%
REBUILD_RQA = False          # set True only if you really want to rebuild
if REBUILD_RQA:
    txt_dir = Path(find("GDPR_2016.txt")).parent                   # folder with the cleaned text files
    all_chunks = []
    for f in sorted(txt_dir.glob("*.txt")):
        words = f.read_text(encoding="utf-8", errors="ignore").split()
        start, idx = 0, 0
        while start < len(words):
            chunk = " ".join(words[start:start + 500])
            if len(chunk) < 100 and start > 0: break
            if len(chunk) >= 30:
                all_chunks.append(dict(doc_name=f.stem, chunk_idx=idx, text=chunk)); idx += 1
            start += 400
    print("chunks:", len(all_chunks))
    emb = bge.encode([c["text"] for c in all_chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    ix = faiss.IndexFlatIP(emb.shape[1]); ix.add(emb)                  # cosine similarity (vectors are normalised)
    bm = BM25Okapi([tokenize(c["text"]) for c in all_chunks])
    out = WORK / "index_rebuilt"; out.mkdir(exist_ok=True)
    faiss.write_index(ix, str(out / "faiss_index.bin"))
    pickle.dump(all_chunks, open(out / "chunks.pkl", "wb")); pickle.dump(bm, open(out / "bm25.pkl", "wb"))
    print("saved to", out)

# %% [markdown]
# ## Part B - build the ObliQA index (needed for steps 06-07)
# Passages longer than 1,500 characters are cut into 1,000-character pieces with stride 800 (so one official passage can become
# several index entries that share the same `(DocumentID, PassageID)`; their RRF scores are pooled later). Passages under 30
# characters are dropped.

# %%
RAW = "https://raw.githubusercontent.com/RegNLP/ObliQADataset/main"
OB = WORK / "ob_artifacts"; OB.mkdir(exist_ok=True)
if not (OB / "faiss_index_obliqua.bin").exists():
    test_data = json.loads(requests.get(f"{RAW}/ObliQA_test.json", timeout=120).text)
    print("ObliQA test questions:", len(test_data), "(paper: 2,786)")
    all_passages = []
    for doc_id in range(1, 41):
        r = requests.get(f"{RAW}/StructuredRegulatoryDocuments/{doc_id}.json", timeout=120)
        if r.status_code == 200: all_passages.extend(json.loads(r.text))
        time.sleep(0.2)
    clean = []
    for p in all_passages:
        text = p["Passage"].strip()
        if len(text) < 30: continue
        if len(text) > 1500:
            s = 0
            while s < len(text):
                piece = text[s:s + 1000]
                if len(piece) < 100 and s > 0: break
                clean.append(dict(DocumentID=p["DocumentID"], PassageID=p["PassageID"], Passage=piece)); s += 800
        else:
            clean.append(dict(DocumentID=p["DocumentID"], PassageID=p["PassageID"], Passage=text))
    print("raw passages:", len(all_passages), "-> index entries:", len(clean))
    emb = bge.encode([p["Passage"] for p in clean], batch_size=128, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    ix = faiss.IndexFlatIP(emb.shape[1]); ix.add(emb)
    bm = BM25Okapi([tokenize(p["Passage"]) for p in clean])
    faiss.write_index(ix, str(OB / "faiss_index_obliqua.bin"))
    pickle.dump(clean, open(OB / "clean_passages_obliqua.pkl", "wb")); pickle.dump(bm, open(OB / "bm25_obliqua.pkl", "wb"))
    json.dump(test_data, open(OB / "ObliQA_test.json", "w"))
    print("saved to", OB)
else:
    print("ObliQA index already built in", OB)
