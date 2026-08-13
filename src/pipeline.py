"""
RegulationQA Retrieval Pipeline
Hybrid BM25 + BGE retrieval with sliding window 
sentence extraction and adaptive inner decay.
"""

import numpy as np
import faiss
import pickle
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import string


def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


def inner_decay(sentences, ce_scores, threshold=0.0, delta=2.0):
    if not sentences:
        return []
    scored = sorted(zip(ce_scores, sentences), reverse=True)
    selected = []
    for i, (score, sent) in enumerate(scored):
        if score < threshold:
            break
        if i > 0 and (scored[i-1][0] - score) > delta:
            break
        selected.append((score, sent))
    return selected if selected else [scored[0]]


def get_window(doc_name, passage_idx, all_chunks):
    doc_chunks = [(i, c) for i, c in enumerate(all_chunks)
                  if c['doc_name'] == doc_name]
    current_pos = next(
        (pos for pos, (i, c) in enumerate(doc_chunks)
         if i == passage_idx), None
    )
    if current_pos is None:
        return all_chunks[passage_idx]['text']
    window = []
    for offset in [-1, 0, 1]:
        pos = current_pos + offset
        if 0 <= pos < len(doc_chunks):
            window.append(doc_chunks[pos][1]['text'])
    return ' '.join(window)


class RegulationQAPipeline:
    def __init__(self, index_dir='index/'):
        print("Loading pipeline...")
        
        # Load chunks
        with open(f'{index_dir}/chunks.pkl', 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(
            f'{index_dir}/faiss_index.bin'
        )
        
        # Load BM25
        with open(f'{index_dir}/bm25.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        
        # Load models
        self.bge = SentenceTransformer('BAAI/bge-base-en-v1.5')
        self.ce  = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        print("Pipeline ready.")

    def retrieve(self, query, top_n=100, rrf_k=60,
                 ce_threshold=0.0, ce_delta=2.0):
        # BGE retrieval
        qe = self.bge.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        _, bge_indices = self.index.search(qe, top_n)
        bge_indices = bge_indices[0]

        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_n]

        # RRF fusion
        rrf_scores = {}
        for rank, idx in enumerate(bge_indices):
            c = self.chunks[idx]
            pid = (c['doc_name'], c['chunk_idx'])
            rrf_scores[pid] = rrf_scores.get(pid, 0) + \
                              1/(rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_indices):
            c = self.chunks[idx]
            pid = (c['doc_name'], c['chunk_idx'])
            rrf_scores[pid] = rrf_scores.get(pid, 0) + \
                              1/(rrf_k + rank + 1)

        sorted_pids = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], reverse=True
        )[:20]

        # Build lookup
        pid_to_idx = {}
        for i, c in enumerate(self.chunks):
            pid = (c['doc_name'], c['chunk_idx'])
            if pid not in pid_to_idx:
                pid_to_idx[pid] = i

        # CE + window extraction
        results = []
        seen = set()
        for pid, rrf_score in sorted_pids:
            if pid in seen:
                continue
            seen.add(pid)
            idx = pid_to_idx.get(pid)
            if idx is None:
                continue
            c = self.chunks[idx]
            curr_sents = sent_tokenize(c['text'])
            if not curr_sents:
                continue
            curr_ce = self.ce.predict(
                [[query, s] for s in curr_sents]
            )
            top_ce = float(max(curr_ce))
            window = get_window(
                c['doc_name'], idx, self.chunks
            )
            win_sents = sent_tokenize(window)
            win_ce = self.ce.predict(
                [[query, s] for s in win_sents]
            )
            kept = inner_decay(
                win_sents, win_ce, 
                ce_threshold, ce_delta
            )
            if kept:
                results.append({
                    'doc_name':       c['doc_name'],
                    'chunk_idx':      c['chunk_idx'],
                    'rrf_score':      float(rrf_score),
                    'top_ce_score':   top_ce,
                    'extracted_spans': [s for _, s in kept],
                })

        results.sort(
            key=lambda x: x['top_ce_score'], reverse=True
        )
        return results


if __name__ == '__main__':
    pipeline = RegulationQAPipeline()
    query = "What are the obligations of a Data Fiduciary?"
    results = pipeline.retrieve(query)
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    for r in results[:3]:
        print(f"\nDoc: {r['doc_name']}")
        print(f"CE Score: {r['top_ce_score']:.3f}")
        for span in r['extracted_spans'][:2]:
            print(f"  → {span[:100]}")