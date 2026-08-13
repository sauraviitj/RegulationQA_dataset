"""
RegulationQA Utility Functions

Shared helpers for text processing, chunking, document extraction,
index building, and result serialization.

Usage:
    from utils import load_documents, chunk_text, build_index
"""

import os
import re
import json
import pickle
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted document text.
    Fixes hyphenation breaks, normalizes whitespace,
    removes excessive blank lines.
    """
    text = re.sub(r'-\n', '', text)           # fix hyphenation
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)       # normalize spaces
    return text.strip()


def tokenize_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25: lowercase + remove punctuation.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


# ─────────────────────────────────────────────────────────────
# Document extraction
# ─────────────────────────────────────────────────────────────

def extract_pdf(filepath: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        import pymupdf
        doc  = pymupdf.open(filepath)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return clean_text(text)
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Run: pip install pymupdf"
        )


def extract_docx(filepath: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx
        doc  = docx.Document(filepath)
        text = "\n".join(para.text for para in doc.paragraphs)
        return clean_text(text)
    except ImportError:
        raise ImportError(
            "python-docx not installed. Run: pip install python-docx"
        )


def extract_document(filepath: str) -> str:
    """
    Extract text from PDF or DOCX based on file extension.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return extract_pdf(filepath)
    elif ext == '.docx':
        return extract_docx(filepath)
    elif ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return clean_text(f.read())
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_documents(
    doc_dir: str,
    doc_name_map: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Load and extract all documents from a directory.

    Args:
        doc_dir:      Directory containing PDF/DOCX/TXT files
        doc_name_map: Optional mapping from filename to clean name
        verbose:      Print progress

    Returns:
        Dict mapping clean_name -> extracted text
    """
    documents = {}
    files     = [f for f in os.listdir(doc_dir)
                 if f.endswith(('.pdf', '.docx', '.txt'))]

    for fname in sorted(files):
        fpath      = os.path.join(doc_dir, fname)
        clean_name = doc_name_map.get(fname, fname) \
                     if doc_name_map else fname
        try:
            text           = extract_document(fpath)
            documents[clean_name] = text
            if verbose:
                words = len(text.split())
                print(f"  ✓ {clean_name}: {words:,} words")
        except Exception as e:
            if verbose:
                print(f"  ✗ {fname}: {e}")

    return documents


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    doc_name: str,
    chunk_size: int = 500,
    overlap: int = 100,
    min_chunk_len: int = 30,
    min_tail_len: int = 100,
) -> List[Dict]:
    """
    Split document text into overlapping word-level chunks.

    Args:
        text:          Document text
        doc_name:      Document identifier
        chunk_size:    Target chunk size in words
        overlap:       Overlap between consecutive chunks in words
        min_chunk_len: Minimum characters to keep a chunk
        min_tail_len:  Minimum characters for tail fragment

    Returns:
        List of chunk dicts with keys
        [doc_name, chunk_idx, text, word_start, word_end]
    """
    words   = text.split()
    chunks  = []
    start   = 0
    idx     = 0
    step    = chunk_size - overlap

    while start < len(words):
        end        = min(start + chunk_size, len(words))
        chunk_text = ' '.join(words[start:end])

        # Skip very short tail fragments
        if len(chunk_text) < min_tail_len and start > 0:
            break

        # Skip chunks below minimum length
        if len(chunk_text) >= min_chunk_len:
            chunks.append({
                'doc_name':   doc_name,
                'chunk_idx':  idx,
                'text':       chunk_text,
                'word_start': start,
                'word_end':   end,
            })
            idx += 1

        start += step

    return chunks


def chunk_documents(
    documents: Dict[str, str],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict]:
    """
    Chunk all documents in a collection.

    Args:
        documents:  Dict mapping doc_name -> text
        chunk_size: Target chunk size in words
        overlap:    Overlap between chunks in words

    Returns:
        List of all chunk dicts across all documents
    """
    all_chunks = []
    for doc_name, text in documents.items():
        chunks     = chunk_text(text, doc_name, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks


# ─────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────

def build_faiss_index(
    chunks: List[Dict],
    model_name: str = 'BAAI/bge-base-en-v1.5',
    batch_size: int = 64,
    text_key: str = 'text',
):
    """
    Build a FAISS inner-product index from chunk embeddings.

    Args:
        chunks:     List of chunk dicts
        model_name: SentenceTransformer model name
        batch_size: Embedding batch size
        text_key:   Key in chunk dict for text field

    Returns:
        Tuple of (faiss_index, embeddings, bge_model)
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    model      = SentenceTransformer(model_name)
    texts      = [c[text_key] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings, model


def build_bm25_index(chunks: List[Dict], text_key: str = 'text'):
    """
    Build a BM25Okapi index from chunks.

    Args:
        chunks:   List of chunk dicts
        text_key: Key in chunk dict for text field

    Returns:
        BM25Okapi index
    """
    from rank_bm25 import BM25Okapi
    corpus = [tokenize_bm25(c[text_key]) for c in chunks]
    bm25   = BM25Okapi(corpus)
    print(f"BM25 index built: {len(corpus)} chunks")
    return bm25


# ─────────────────────────────────────────────────────────────
# Index I/O
# ─────────────────────────────────────────────────────────────

def save_index(
    index,
    chunks: List[Dict],
    bm25,
    output_dir: str = 'index/',
) -> None:
    """
    Save FAISS index, chunks, and BM25 index to disk.
    """
    import faiss
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, 'faiss_index.bin'))
    with open(os.path.join(output_dir, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)
    with open(os.path.join(output_dir, 'bm25.pkl'), 'wb') as f:
        pickle.dump(bm25, f)
    print(f"Index saved to {output_dir}")


def load_index(index_dir: str = 'index/') -> Tuple:
    """
    Load FAISS index, chunks, and BM25 index from disk.

    Returns:
        Tuple of (faiss_index, chunks, bm25)
    """
    import faiss
    index = faiss.read_index(
        os.path.join(index_dir, 'faiss_index.bin')
    )
    with open(os.path.join(index_dir, 'chunks.pkl'), 'rb') as f:
        chunks = pickle.load(f)
    with open(os.path.join(index_dir, 'bm25.pkl'), 'rb') as f:
        bm25 = pickle.load(f)
    print(f"Loaded: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks, bm25


# ─────────────────────────────────────────────────────────────
# Window construction
# ─────────────────────────────────────────────────────────────

def get_window(
    doc_name: str,
    chunk_idx: int,
    all_chunks: List[Dict],
    window_size: int = 1,
) -> str:
    """
    Build a sliding window of text around a given chunk.

    Args:
        doc_name:    Document name to scope the window
        chunk_idx:   Global index of the anchor chunk
        all_chunks:  Full list of all chunks
        window_size: Number of chunks before and after (default ±1)

    Returns:
        Concatenated window text
    """
    doc_chunks = [
        (i, c) for i, c in enumerate(all_chunks)
        if c['doc_name'] == doc_name
    ]
    current_pos = next(
        (pos for pos, (i, c) in enumerate(doc_chunks)
         if i == chunk_idx),
        None
    )
    if current_pos is None:
        return all_chunks[chunk_idx]['text']

    window = []
    for offset in range(-window_size, window_size + 1):
        pos = current_pos + offset
        if 0 <= pos < len(doc_chunks):
            window.append(doc_chunks[pos][1]['text'])
    return ' '.join(window)


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────

def load_regulationqa(
    csv_path: str = 'dataset/regulationqa_v1.csv',
) -> pd.DataFrame:
    """
    Load the RegulationQA dataset.

    Returns:
        DataFrame with columns [id, document_source, question,
        ground_truth_answer]
    """
    df = pd.read_csv(csv_path)
    print(f"RegulationQA loaded: {len(df)} questions, "
          f"{df['document_source'].nunique()} document sources")
    return df


def load_obliqua(split: str = 'test') -> List[Dict]:
    """
    Download and return ObliQA data from GitHub.

    Args:
        split: 'train', 'dev', or 'test'

    Returns:
        List of question dicts
    """
    import requests
    url = (
        f"https://raw.githubusercontent.com/RegNLP/ObliQADataset/"
        f"main/ObliQA_{split}.json"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print(f"ObliQA {split} loaded: {len(data)} questions")
    return data


# ─────────────────────────────────────────────────────────────
# Result serialization
# ─────────────────────────────────────────────────────────────

def save_results(results: Dict, path: str) -> None:
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False,
                  default=str)
    print(f"Saved: {path}")


def load_results(path: str) -> Dict:
    """Load results dict from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# Corpus statistics
# ─────────────────────────────────────────────────────────────

def corpus_stats(chunks: List[Dict], text_key: str = 'text') -> Dict:
    """
    Compute corpus statistics from a list of chunks.

    Returns:
        Dict with total_chunks, total_words, unique_docs,
        mean_chunk_words, min_chunk_words, max_chunk_words
    """
    lengths   = [len(c[text_key].split()) for c in chunks]
    unique_docs = len(set(c['doc_name'] for c in chunks))
    return {
        'total_chunks':    len(chunks),
        'unique_docs':     unique_docs,
        'total_words':     sum(lengths),
        'mean_chunk_words': np.mean(lengths),
        'min_chunk_words':  min(lengths),
        'max_chunk_words':  max(lengths),
    }


if __name__ == '__main__':
    print("utils.py loaded successfully.")
    print("Available functions:")
    print("  clean_text(text)")
    print("  tokenize_bm25(text)")
    print("  extract_pdf(filepath)")
    print("  extract_docx(filepath)")
    print("  extract_document(filepath)")
    print("  load_documents(doc_dir, doc_name_map)")
    print("  chunk_text(text, doc_name)")
    print("  chunk_documents(documents)")
    print("  build_faiss_index(chunks)")
    print("  build_bm25_index(chunks)")
    print("  save_index(index, chunks, bm25, output_dir)")
    print("  load_index(index_dir)")
    print("  get_window(doc_name, chunk_idx, all_chunks)")
    print("  load_regulationqa(csv_path)")
    print("  load_obliqua(split)")
    print("  save_results(results, path)")
    print("  load_results(path)")
    print("  corpus_stats(chunks)")
