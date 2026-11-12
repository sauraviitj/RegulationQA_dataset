# RegulationQA

**A regulatory document retrieval benchmark spanning Indian, European, and international cybersecurity and data protection frameworks.**

RegulationQA comprises **1,002 questions** over **20 documents**, including cross-document questions that need evidence from two regulatory instruments. It is designed to evaluate retrieval systems under realistic compliance constraints: heterogeneous documents, precise legal terminology, and offline deployment requirements.

> **Reproducing the paper?** Start with the numbered, annotated series in [`code_series/`](code_series/README.md). It covers indexing, retrieval, span extraction, ablations, BERTScore, ObliQA, LLM judges, NLI and corpus analysis, and says which step produces which paper table.

---

## Pipeline Architecture

![RegulationQA Pipeline Architecture](architecture.png)

The retrieval pipeline combines dense (BGE) and sparse (BM25) retrieval via Reciprocal Rank Fusion (k = 60), followed by cross-encoder scoring of the fused candidates (pool m = 20), a ±1 passage sliding window, and sentence-level span extraction with adaptive inner decay (τ = 0, δ = 2). No generative model is required at inference time.

---

## Dataset

### Format

```csv
id, document_source, question, ground_truth_answer
```

| Field | Description |
|---|---|
| `id` | Unique question identifier |
| `document_source` | Source document name (or `Doc A + Doc B` for cross-document) |
| `question` | Natural language compliance question |
| `ground_truth_answer` | Reference answer extracted from source document |

607 questions are single-source and 395 are cross-document. The raw `document_source` strings contain naming variants of the same instrument; `code_series/adsir_common.py` maps them to one canonical name each.

### Question Types

| Type | Description |
|---|---|
| Definitional | What is X? What does X mean? |
| Obligation | Who must do X? What is required? |
| Procedural | How must X be done? What steps are required? |
| Penalty | What happens if X is violated? |
| Scope | Does X apply to Y? Who is covered? |
| Cross-document | Requires evidence from two instruments |

### Document Corpus (20 documents)

#### Indian Regulatory (14 documents)

| Document | Type | Words |
|---|---|---|
| DPDP Act 2023 | Indian statute | 10,685 |
| Digital Personal Data Protection Rules 2025 | Indian rules | 10,293 |
| CERT-In Directions 2022 | Indian direction | 1,844 |
| IT Act 2000 | Indian statute | 19,525 |
| Electricity Act 2003 | Indian statute | 41,155 |
| NCIIPC Guidelines | Indian guideline | 14,696 |
| RBI Cyber Security Framework | Indian direction | 6,732 |
| e-Governance Policy India | Indian policy | 43,569 |
| SEBI CSCRF | Indian direction | 2,008 |
| State DC Security Guidelines | Indian guideline | 2,657 |
| TRAI Telecom Advisory | Indian direction | 4,057 |
| NCIIPC Rules | Indian rules | 4,044 |
| IT Protected System Rules 2018 | Indian rules | 4,045 |
| Cyber Security Power Sector 2021 | Indian guideline | 8,240 |

#### European (1 document)

| Document | Type | Words |
|---|---|---|
| GDPR (EU) 2016/679 | European statute | 55,799 |

#### International Standards (5 documents)

| Document | Type | Words |
|---|---|---|
| ISO/IEC 27001:2022 | International standard | 7,194 |
| NIST SP 800-66r2 | International standard | 42,128 |
| NIST Cybersecurity Framework | International standard | 7,834 |
| OECD Digital Security Guidelines | International standard | 4,154 |
| PCI Security Standards | International standard | 1,107 |

---

## Evaluation Results

### RegulationQA, span level (full pipeline)

| Span mode | Token F1 | Uses reference answer? |
|---|---|---|
| Top-1 | 0.2994 | No |
| Query-guided (highest cross-encoder sentence among top-3) | 0.3083 | No |
| Oracle best-of-3 | 0.3737 | **Yes** (upper bound, not a deployable system) |

BERTScore (roberta-large, raw F1): 0.8804 (top-1), 0.8801 (query-guided), 0.8939 (best-of-3).
Step 05 of the code series also reports length-matched controls, because BERTScore can reward length.

### RegulationQA, baselines and LLM judges (oracle best-of-3 spans)

| System | Token F1 | R@F1≥0.3 | R@F1≥0.5 | Qwen3-4B | Mistral-7B |
|---|---|---|---|---|---|
| BGE only | 0.2229 | 0.1377 | 0.0539 | 46.2% | 79.3% |
| BM25 only | 0.2156 | 0.1188 | 0.0509 | 39.5% | 77.1% |
| **Full pipeline** | **0.3623** | **0.4062** | **0.2764** | **82.8%** | **91.4%** |

The judges score the oracle best-of-3 span (chosen using the reference answer), so these percentages measure whether the answer is present in the top-3, not top-1 accuracy. Judge percentages are "correct or partially correct".
Inter-model agreement between Qwen3-4B and Mistral-7B: Cohen's κ = 0.296 (Fair). The gap between token F1 (0.36) and LLM-judged correctness (82.8–91.4%) reflects the limits of overlap metrics on regulatory text, where the same obligation can be phrased in many valid ways.

### ObliQA (Passage-Level, 2,786 test questions)

| System | R@1 | R@5 | R@10 | MAP@10 |
|---|---|---|---|---|
| BGE only | 0.4986 | 0.7146 | 0.7778 | 0.5400 |
| BM25 only | 0.5696 | 0.7545 | 0.8008 | 0.5808 |
| BGE + CE + Window | 0.5754 | 0.7373 | 0.7778 | 0.5841 |
| **Full pipeline (pool m = 50)** | **0.5750** | **0.7624** | **0.8029** | **0.5915** |

Differences between the full pipeline and BM25 are small; `code_series/07_obliqa_results_ablation` gives paired-bootstrap 95% confidence intervals and the m = 20 variant.
Published baselines on ObliQA: BM25 MAP@10 = 0.6236, BGE-5shot MAP@10 = 0.6178 (Gokhan et al., COLING 2025).

### Reproducibility notes

* The released spans in `evaluation/extraction_results.csv` come from the original submitted run. Re-running the released code reproduces about 88% of the saved top-1 spans exactly; the rest differ because of small retrieval-pool differences.
* The saved spans are single sentences (the best-scoring sentence of the ±1 window of each candidate). `src/pipeline.py` returns all sentences kept by inner decay; `code_series/03_rqa_span_extraction` computes both and labels them.
* Hallucination (NLI) numbers in the paper come from the original run; `code_series/09_hallucination_nli` is a fresh re-run and will differ.

---

## Repository Structure

```
RegulationQA/
├── README.md
├── requirements.txt
├── architecture.png              <- pipeline diagram
│
├── code_series/                  <- numbered, annotated pipeline (start here)
│   ├── README.md
│   ├── adsir_common.py           <- shared settings and helpers
│   ├── 01_build_indexes.py ... 11_corpus_analysis.py
│   └── notebooks/                <- the same steps as .ipynb
│
├── dataset/
│   └── regulationqa_v1.csv       <- 1002 Q&A pairs
│
├── documents/
│   └── cleaned_texts/
│       └── *.txt                 <- cleaned text files
│
├── index/
│   ├── faiss_index.bin           <- BGE FAISS index (766 chunks)
│   ├── chunks.pkl                <- chunked passages with metadata
│   └── bm25.pkl                  <- BM25Okapi sparse index
│
├── evaluation/
│   ├── extraction_log.json       <- retrieval results per question
│   ├── extraction_results.csv    <- summary CSV (saved submitted run)
│   ├── qwen_eval.csv             <- Qwen3-4B judge scores
│   ├── combined_eval.csv         <- all evaluation scores
│   ├── overlap_analysis.csv      <- document overlap analysis
│   ├── ablation_results.json     <- ablation study results
│   ├── all_llm_eval.json         <- full LLM evaluation
│   └── baseline_llm_eval.json    <- baseline LLM evaluation
│
├── visualisations/
│   ├── doc_similarity_graph.html <- document similarity graph
│   ├── kg_answers_v2.html        <- retrieval quality graph
│   └── doc_similarity_matrix.csv <- pairwise similarity matrix
│
├── Notebooks/                    <- original working notebooks (kept for history)
│
└── src/
    ├── pipeline.py               <- retrieval pipeline
    ├── evaluate.py               <- evaluation functions
    └── utils.py                  <- helper utilities
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/[anonymous]/RegulationQA
cd RegulationQA
pip install -r requirements.txt
```

### Load the Dataset

```python
import pandas as pd

df = pd.read_csv('dataset/regulationqa_v1.csv')
print(f"Questions: {len(df)}")
print(df.head())
```

### Load the Index

```python
import pickle
import faiss

with open('index/chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
index = faiss.read_index('index/faiss_index.bin')
with open('index/bm25.pkl', 'rb') as f:
    bm25 = pickle.load(f)

print(f"Chunks: {len(chunks)}, Index vectors: {index.ntotal}")
```

### Run the Pipeline

```python
from src.pipeline import RegulationQAPipeline

pipeline = RegulationQAPipeline(index_dir='index/')

query   = "What are the obligations of a Data Fiduciary under DPDP Act?"
results = pipeline.retrieve(query)

for r in results[:3]:
    print(f"\nDocument: {r['doc_name']}")
    print(f"CE Score: {r['top_ce_score']:.3f}")
    for span in r['extracted_spans'][:2]:
        print(f"  -> {span}")
```

### Evaluate

```python
from src.evaluate import evaluate_regulationqa, print_report

metrics = evaluate_regulationqa(df=df, retrieve_fn=pipeline.retrieve)
print_report(metrics, "RegulationQA Results")
```

### Reproduce the paper tables

Follow [`code_series/README.md`](code_series/README.md): run steps 01 to 11 in order (GPU T4 for steps 02, 05, 06, 08, 09, 11).

### Build Index From Scratch

```python
from src.utils import (
    load_documents, chunk_documents,
    build_faiss_index, build_bm25_index, save_index
)

documents  = load_documents('documents/cleaned_texts/')
chunks     = chunk_documents(documents, chunk_size=500, overlap=100)

index, embeddings, model = build_faiss_index(chunks)
bm25                     = build_bm25_index(chunks)

save_index(index, chunks, bm25, output_dir='index/')
```

For reproducing the paper, use the shipped index in `index/`; a rebuilt index can differ slightly.

---

## Requirements

```
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
rank-bm25>=0.2.2
nltk>=3.7
pymupdf>=1.23.0
python-docx>=0.8.11
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.64.0
transformers>=4.35.0
torch>=2.0.0
```

Code-series steps 05 and 08 additionally use `bert-score`, `accelerate` and `bitsandbytes` (installed by the first cell of each notebook).

---

## License

### Dataset License

Questions and annotations in `dataset/regulationqa_v1.csv` are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

You are free to share and adapt the material for any purpose, provided appropriate credit is given.

### Document License

The cleaned document texts in `documents/cleaned_texts/` are sourced from publicly available government publications and international standards bodies:

| Document | Source | License |
|---|---|---|
| DPDP Act 2023 | Government of India | Public domain |
| CERT-In Directions 2022 | MeitY, Government of India | Public domain |
| IT Act 2000 | Government of India | Public domain |
| Electricity Act 2003 | Government of India | Public domain |
| NCIIPC Guidelines | Government of India | Public domain |
| RBI Cyber Security Framework | Reserve Bank of India | Public domain |
| e-Governance Policy India | Government of India | Public domain |
| SEBI CSCRF | SEBI | Public domain |
| State DC Security Guidelines | Government of India | Public domain |
| TRAI Telecom Advisory | TRAI | Public domain |
| NCIIPC Rules | Government of India | Public domain |
| IT Protected System Rules 2018 | Government of India | Public domain |
| Cyber Security Power Sector 2021 | Government of India | Public domain |
| GDPR (EU) 2016/679 | European Union | Public domain |
| NIST SP 800-66r2 | NIST (US Government) | Public domain |
| NIST Cybersecurity Framework | NIST (US Government) | Public domain |
| OECD Digital Security Guidelines | OECD | Research use |
| ISO/IEC 27001:2022 | ISO/IEC | Research use only |
| PCI Security Standards | PCI SSC | Research use only |
| DPDP Rules 2025 | Government of India | Public domain |

**Note:** ISO/IEC 27001:2022 and PCI Security Standards are proprietary standards. The cleaned texts are included solely for academic research purposes. Users must comply with the original licensing terms of these documents and must not use them for commercial purposes.

### Code License

Source code in `src/` and `code_series/` is released under the **MIT License**.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citation

If you use RegulationQA in your research, please cite:

To be updated

## Acknowledgements

Documents sourced from publicly available government publications of India, the European Union, and international standards bodies. We thank the annotators who contributed to the human evaluation.

---

## Contact

For questions about the dataset or pipeline, please open a GitHub issue.
