"""
adsir_common.py  --  shared configuration and helper functions for the ADSIR / RegulationQA code series.

EVERY numbered step (01_...py to 11_...py) starts with:   from adsir_common import *
so that a setting (paths, pool size, thresholds, model names) is defined in exactly ONE place.

Sections
  1. Paths and file discovery (works on Kaggle and on a local checkout of the repo)
  2. Pipeline constants (the paper's configuration)
  3. Text helpers: tokenizer, token F1, inner decay, RRF fusion
  4. RegulationQA label normalisation (single-source vs cross-document, pooled variants)
  5. Statistics helper: paired bootstrap
"""
import os, re, sys, json, glob, pickle, string, time, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------------------------
IS_KAGGLE = Path("/kaggle/input").exists()
WORK = Path("/kaggle/working") if IS_KAGGLE else Path("./outputs")      # everything we write goes here
WORK.mkdir(parents=True, exist_ok=True)

# Extra places to look for artifacts when running locally (the repo layout).
REPO_DIRS = [Path(p) for p in ["index", "dataset", "evaluation", "../index", "../dataset", "../evaluation", "."]]

def find(name, required=True):
    """Locate a file by name: first in WORK, then anywhere under /kaggle/input, then in the repo folders."""
    cands = [WORK / name]
    if IS_KAGGLE: cands += list(Path("/kaggle/input").rglob(name))
    cands += [d / name for d in REPO_DIRS]
    for c in cands:
        if Path(c).exists():
            print(f"[find] {name:32s} -> {c}")
            return str(c)
    if required: raise FileNotFoundError(f"{name} not found. Add the dataset that contains it to this notebook, or run the step that creates it.")
    return None

# ---------------------------------------------------------------------------------------------
# 2. PIPELINE CONSTANTS  (the configuration used in the paper)
# ---------------------------------------------------------------------------------------------
RRF_K          = 60       # Reciprocal Rank Fusion constant, rank is 0-based so the term is 1/(RRF_K + rank + 1)
PRE_K          = 100      # passages taken from EACH retriever (BGE and BM25) before fusion
M_RQA          = 20       # fused candidates scored by the cross-encoder on RegulationQA (paper text: m = 20)
M_OBLIQA       = 20       # same on ObliQA (the released ObliQA notebooks use 50; both are reported in the paper)
WINDOW_RADIUS  = 1        # +-1 neighbouring passage around each candidate
TAU_IN         = 0.0      # inner decay score floor
DELTA_IN       = 2.0      # inner decay drop tolerance of the repo code (src/pipeline.py default). It only matters
                          # when SEVERAL sentences are kept (best-of-3, query-guided, the multi-sentence variant).
BGE_NAME       = "BAAI/bge-base-en-v1.5"
CE_NAME        = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BERT_MODEL     = "roberta-large"           # BERTScore (raw F1, no baseline rescaling)
QWEN_NAME      = "Qwen/Qwen3-4B"
MISTRAL_NAME   = "mistralai/Mistral-7B-Instruct-v0.3"
NLI_NAME       = "cross-encoder/nli-deberta-v3-base"

# ---------------------------------------------------------------------------------------------
# 3. TEXT HELPERS
# ---------------------------------------------------------------------------------------------
def tokenize(text):
    """BM25 tokenizer: lowercase, strip punctuation, split on whitespace (same as the repo)."""
    return str(text).lower().translate(str.maketrans("", "", string.punctuation)).split()

def token_f1(prediction, ground_truth):
    """Set-based token F1 used throughout the paper (identical to src/evaluate.py):
    precision = |overlap of word SETS| / (number of predicted tokens), recall likewise with the reference length."""
    pred = str(prediction).lower().split(); gt = str(ground_truth).lower().split()
    common = set(pred) & set(gt)
    if not common: return 0.0
    p, r = len(common) / len(pred), len(common) / len(gt)
    return 2 * p * r / (p + r)

def inner_decay(scored, tau=TAU_IN, delta=DELTA_IN):
    """Keep sentences in descending cross-encoder score while (a) score >= tau and (b) the drop from the previous kept
    sentence is <= delta. Always keeps at least the best sentence. `scored` = list of (score, sentence)."""
    if not scored: return []
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    kept = []
    for i, (s, t) in enumerate(scored):
        if s < tau: break
        if i > 0 and (scored[i - 1][0] - s) > delta: break
        kept.append((s, t))
    return kept if kept else [scored[0]]

def rrf_fuse(rank_lists, key_fn, k=RRF_K):
    """Reciprocal Rank Fusion over several ranked lists of indices. Scores are accumulated PER PASSAGE KEY
    (e.g. (doc, chunk_idx)), so a passage that appears several times accumulates score. Returns [(key, score)] sorted."""
    acc = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst):
            kk = key_fn(idx); acc[kk] = acc.get(kk, 0.0) + 1.0 / (k + rank + 1)
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------------------------------------------
# 4. REGULATIONQA LABELS
# ---------------------------------------------------------------------------------------------
# The released CSV stores 40 raw `document_source` strings: variants of the same instrument (e.g. "GDPR" and
# "General Data Protection Regulation (EU) 2016/679") and cross-document labels containing " + ".  This maps the
# 12 single-source instruments to ONE name each. Derived from the actual labels in regulationqa_v1.csv; the counts
# reproduce the paper (607 single-source, 395 cross-document) and step 03 asserts that.
SOURCE_CANON = [   # (substring that identifies the label, canonical instrument name)
    ("e-Governance Policy Initiatives",            "e-Governance Policy India"),
    ("General Data Protection Regulation",         "GDPR"),
    ("GDPR",                                       "GDPR"),
    ("OECD/LEGAL/0479",                            "OECD Digital Security Guidelines"),
    ("NIST SP 800-66r2",                           "NIST SP 800-66r2"),
    ("DPDP Act 2023",                              "DPDP Act 2023"),
    ("NCIIPC Guidelines",                          "NCIIPC Guidelines"),
    ("ISO 27001:2022",                             "ISO/IEC 27001:2022"),
    ("ISO/IEC 27001:2022",                         "ISO/IEC 27001:2022"),
    ("Presentation - CSCRF",                       "SEBI CSCRF"),
    ("State Data Centre security guidelines",      "State DC Security Guidelines"),
    ("Digital Personal Data Protection Rules, 2025", "DPDP Rules 2025"),
    ("PCI Security Standards",                     "PCI Security Standards"),
    ("Takshashila",                                "TRAI Telecom Advisory"),
]
def is_cross(label):
    label = str(label); return (" + " in label) or label.strip() == "Cross-Document"
def canonical_source(label):
    """Canonical single-source instrument name, or 'CROSS' for cross-document questions."""
    if is_cross(label): return "CROSS"
    for sub, name in SOURCE_CANON:
        if sub in str(label): return name
    raise ValueError(f"Unmapped document_source label: {label!r} -- add it to SOURCE_CANON in adsir_common.py")

def load_regulationqa():
    df = pd.read_csv(find("regulationqa_v1.csv"))
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]].copy()
    df["source"] = df["document_source"].map(canonical_source)
    df["is_cross"] = (df["source"] == "CROSS").astype(int)
    assert len(df) == 1002 and df.is_cross.sum() == 395 and (df.is_cross == 0).sum() == 607, "RegulationQA counts differ from the paper"
    return df

# ---------------------------------------------------------------------------------------------
# 5. STATISTICS
# ---------------------------------------------------------------------------------------------
def paired_bootstrap(a, b, B=2000, seed=0):
    """Mean difference a-b with a 95% percentile bootstrap CI over questions (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; n = len(d)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def save_json(obj, name):
    p = WORK / name
    with open(p, "w") as f: json.dump(obj, f, indent=2, default=float)
    print("saved", p)

# ---------------------------------------------------------------------------------------------
# 6. REGULATIONQA RESOURCES + CACHE HELPERS  (used by steps 02-05, 08-11)
# ---------------------------------------------------------------------------------------------
def load_rqa_resources(load_models=True):
    """Loads the released RegulationQA index (chunks.pkl, faiss_index.bin, bm25.pkl) and, optionally, BGE + cross-encoder.
    Returns a dict R with: chunks, index, bm25, bge, ce, doc_list, pos_in_doc, pid_to_idx, window_text()."""
    import faiss
    with open(find("chunks.pkl"), "rb") as f: chunks = pickle.load(f)
    index = faiss.read_index(find("faiss_index.bin"))
    with open(find("bm25.pkl"), "rb") as f: bm25 = pickle.load(f)
    assert len(chunks) == index.ntotal, (len(chunks), index.ntotal)
    R = dict(chunks=chunks, index=index, bm25=bm25)
    # Chunks of one document are stored consecutively, so the +-1 window is just the neighbouring chunks in the list.
    doc_list = defaultdict(list)
    for i, c in enumerate(chunks): doc_list[c["doc_name"]].append(i)
    R["doc_list"] = doc_list
    R["pos_in_doc"] = {i: (d, p) for d, ids in doc_list.items() for p, i in enumerate(ids)}
    pid_to_idx = {}
    for i, c in enumerate(chunks): pid_to_idx.setdefault((c["doc_name"], c["chunk_idx"]), i)
    R["pid_to_idx"] = pid_to_idx
    def window_text(idx, radius=WINDOW_RADIUS):
        d, p = R["pos_in_doc"][idx]; ids = doc_list[d]
        return " ".join(chunks[ids[q]]["text"] for q in range(p - radius, p + radius + 1) if 0 <= q < len(ids))
    R["window_text"] = window_text
    if load_models:
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        R["bge"] = SentenceTransformer(BGE_NAME, device=dev)
        R["ce"] = CrossEncoder(CE_NAME, device=dev)
    print(f"RegulationQA resources: {len(chunks)} chunks, {len(doc_list)} documents")
    return R

def load_cache(prefix):
    """Concatenate all shards `<prefix>_NNNN.pkl` found in WORK (sorted by question row / index)."""
    out = []
    for p in sorted(glob.glob(str(WORK / f"{prefix}_*.pkl"))): out += pickle.load(open(p, "rb"))
    key = "row" if (out and "row" in out[0]) else "q"
    out.sort(key=lambda q: q[key]); return out

def results_for(q, tau=TAU_IN, delta=DELTA_IN, use_window=True):
    """Repo rule (src/pipeline.py). For every candidate passage: rank key = best cross-encoder score among the passage's OWN
    sentences; the text is the inner-decay-kept sentences of the +-1 window joined with spaces (or of the own passage when
    use_window=False). Returns candidates sorted by rank key."""
    S, C = q["sents"], q["ce"]; res = []
    for c in q["cands"]:
        top_ce = max(float(C[j]) for j in c["own"])
        ids = c["win"] if use_window else c["own"]
        kept = inner_decay([(float(C[j]), S[j]) for j in ids], tau, delta)
        res.append(dict(doc=c["doc"], chunk_idx=c["chunk_idx"], top_ce=top_ce, kept=kept, text=" ".join(t for _, t in kept)))
    res.sort(key=lambda r: r["top_ce"], reverse=True)
    return res

def span_modes_repo(res, gt):
    """The three reported modes under the REPO rule (joined kept sentences), for diagnostics / sensitivity (step 04)."""
    if not res: return dict(top1="", qg="", orc="")
    top3 = res[:3]; sents = [(s, t) for r in top3 for s, t in r["kept"]]
    return dict(top1=res[0]["text"],
                qg=max(sents, key=lambda x: x[0])[1],                    # highest cross-encoder sentence over the top-3 candidates
                orc=max(sents, key=lambda x: token_f1(x[1], gt))[1])     # oracle: best token-F1 sentence (uses the reference!)

def span_modes_saved_run(q, gt, n_top=3):
    """The rule that REPRODUCES the saved submitted run (about 88% of top-1 spans identical; see README, 'honest note').
    Candidates ranked by own-passage max CE score; each candidate contributes ONE span = the best-scoring sentence of its
    +-1 window. top-1 = best sentence of candidate 1. query-guided = highest-scoring of the three. oracle = best token-F1 of the three."""
    S, C = q["sents"], q["ce"]
    if not q["cands"]: return dict(top1="", qg="", orc="")
    top = sorted(q["cands"], key=lambda c: max(float(C[j]) for j in c["own"]), reverse=True)[:n_top]
    best = [max(c["win"], key=lambda j: float(C[j])) for c in top]
    return dict(top1=S[best[0]],
                qg=S[max(best, key=lambda j: float(C[j]))],
                orc=max((S[j] for j in best), key=lambda s: token_f1(s, gt)))
