"""
RegulationQA Evaluation Module

Supports three evaluation dimensions:
  1. Automatic metrics (Token F1, Recall@threshold)
  2. Passage-level metrics (Recall@k, MAP@k) for ObliQA-style benchmarks
  3. LLM-based semantic evaluation (Qwen3-4B / Mistral-7B)

Usage:
    from evaluate import evaluate_regulationqa, evaluate_obliqua, llm_evaluate
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from typing import List, Dict, Callable, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Token F1
# ─────────────────────────────────────────────────────────────

def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between prediction and ground truth.
    Tokenizes by whitespace after lowercasing.
    """
    pred_tokens = prediction.lower().split()
    gt_tokens   = ground_truth.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────
# RegulationQA evaluation (span-level, F1-based)
# ─────────────────────────────────────────────────────────────

def evaluate_regulationqa(
    df: pd.DataFrame,
    retrieve_fn: Callable,
    top_n: int = 50,
    f1_thresholds: List[float] = [0.3, 0.5],
) -> Dict:
    """
    Evaluate a retrieval function on RegulationQA.

    Args:
        df:            DataFrame with columns [question, ground_truth_answer,
                       document_source]
        retrieve_fn:   Function (query: str) -> List[Dict] where each dict
                       has 'extracted_spans' key
        top_n:         Candidate pool size
        f1_thresholds: List of F1 thresholds for Recall@F1>=t metrics

    Returns:
        Dict with Mean F1 and Recall@threshold metrics, plus per-question log
    """
    f1_scores   = []
    recall_bins = {t: [] for t in f1_thresholds}
    results_log = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query        = row['question']
        ground_truth = str(row['ground_truth_answer'])
        doc_source   = row['document_source']

        results = retrieve_fn(query, top_n=top_n)

        best_f1   = 0.0
        best_span = ""
        best_doc  = ""

        for r in results[:3]:
            spans = r.get('extracted_spans', [])
            doc   = r.get('doc_name', r.get('DocumentID', ''))
            for span in spans:
                f1 = token_f1(span, ground_truth)
                if f1 > best_f1:
                    best_f1   = f1
                    best_span = span
                    best_doc  = doc

        f1_scores.append(best_f1)
        for t in f1_thresholds:
            recall_bins[t].append(1.0 if best_f1 >= t else 0.0)

        results_log.append({
            'question':     query,
            'doc_source':   doc_source,
            'ground_truth': ground_truth,
            'best_span':    best_span,
            'best_doc':     best_doc,
            'f1':           best_f1,
        })

    metrics = {'Mean F1': np.mean(f1_scores)}
    for t in f1_thresholds:
        metrics[f'Recall@F1>={t}'] = np.mean(recall_bins[t])
    metrics['results_log'] = results_log
    return metrics


# ─────────────────────────────────────────────────────────────
# ObliQA evaluation (passage-level, MAP/Recall@k)
# ─────────────────────────────────────────────────────────────

def evaluate_obliqua(
    test_data: List[Dict],
    retrieve_fn: Callable,
    top_n: int = 100,
    k_values: List[int] = [1, 5, 10],
) -> Dict:
    """
    Evaluate a retrieval function on ObliQA-style data.

    Args:
        test_data:   List of dicts with keys [Question, Passages]
                     where Passages is list of {DocumentID, PassageID}
        retrieve_fn: Function (query: str) -> List[Dict] where each dict
                     has 'DocumentID' and 'PassageID' keys
        top_n:       Candidate pool size
        k_values:    List of k values for Recall@k

    Returns:
        Dict with Recall@k and MAP@10 metrics
    """
    recall_at_k = {k: [] for k in k_values}
    map_scores  = []

    for item in tqdm(test_data):
        query   = item['Question']
        gt_pids = set(
            (p['DocumentID'], p['PassageID'])
            for p in item['Passages']
        )

        results   = retrieve_fn(query, top_n=top_n)
        retrieved = [(r['DocumentID'], r['PassageID']) for r in results]

        for k in k_values:
            hit = any(p in gt_pids for p in retrieved[:k])
            recall_at_k[k].append(1.0 if hit else 0.0)

        ap, hits = 0.0, 0
        for rank, pid in enumerate(retrieved[:10], 1):
            if pid in gt_pids:
                hits += 1
                ap   += hits / rank
        map_scores.append(ap / max(len(gt_pids), 1))

    metrics = {}
    for k in k_values:
        metrics[f'Recall@{k}'] = np.mean(recall_at_k[k])
    metrics['MAP@10'] = np.mean(map_scores)
    return metrics


# ─────────────────────────────────────────────────────────────
# LLM-based semantic evaluation
# ─────────────────────────────────────────────────────────────

LLM_PROMPT = """You are evaluating a regulatory information retrieval system.

QUESTION: {question}

REFERENCE ANSWER: {ground_truth}

EXTRACTED SPAN: {extracted_span}

Does the extracted span correctly answer the question?
Consider it correct even if the wording differs from the reference,
as long as the information is accurate and relevant.

Respond with exactly one of:
2 - Fully correct: span contains the complete answer
1 - Partially correct: span contains relevant but incomplete information
0 - Incorrect: span does not answer the question

Your response (single digit only):"""


def _score_one(model, tokenizer, device: str,
               question: str, ground_truth: str,
               span: str) -> int:
    """Score a single span with a causal LLM judge."""
    import torch
    if not span or span.strip() == "":
        return 0
    prompt = LLM_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        extracted_span=span
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response   = tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()
    del inputs, outputs, new_tokens
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    for char in response:
        if char in ['0', '1', '2']:
            return int(char)
    return 0


def llm_evaluate(
    results_log: List[Dict],
    model_name: str = "Qwen/Qwen3-4B",
    device: str = "auto",
) -> Dict:
    """
    Evaluate retrieved spans using an LLM judge.

    Args:
        results_log: List of dicts with keys
                     [question, ground_truth, best_span, doc_source]
        model_name:  HuggingFace model ID for judge model
        device:      Device string ('auto', 'cuda:0', 'cpu')

    Returns:
        Dict with score distribution and per-item scores
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    actual_device = next(model.parameters()).device
    print(f"Judge ready on {actual_device}")

    scores   = []
    out_log  = []

    for item in tqdm(results_log):
        score = _score_one(
            model, tokenizer, actual_device,
            item['question'],
            item['ground_truth'],
            item.get('best_span', '')
        )
        scores.append(score)
        out_log.append({**item, 'llm_score': score})

    scores_arr = np.array(scores)
    return {
        'model':            model_name,
        'fully_correct':    float((scores_arr == 2).mean()),
        'partially_correct': float((scores_arr == 1).mean()),
        'incorrect':        float((scores_arr == 0).mean()),
        'correct_partial':  float((scores_arr >= 1).mean()),
        'mean_score':       float(scores_arr.mean()),
        'scores':           scores,
        'log':              out_log,
    }


# ─────────────────────────────────────────────────────────────
# Inter-annotator / inter-model agreement
# ─────────────────────────────────────────────────────────────

def compute_kappa(
    scores_a: List[int],
    scores_b: List[int],
    label_a: str = "Judge A",
    label_b: str = "Judge B",
) -> Dict:
    """
    Compute Cohen's kappa between two score lists.

    Args:
        scores_a: List of integer scores from judge A
        scores_b: List of integer scores from judge B
        label_a:  Name of judge A for reporting
        label_b:  Name of judge B for reporting

    Returns:
        Dict with kappa, interpretation, and confusion matrix
    """
    assert len(scores_a) == len(scores_b), \
        "Score lists must have equal length"

    kappa = cohen_kappa_score(scores_a, scores_b)

    if kappa >= 0.80:
        interp = "Almost perfect"
    elif kappa >= 0.60:
        interp = "Substantial"
    elif kappa >= 0.40:
        interp = "Moderate"
    elif kappa >= 0.20:
        interp = "Fair"
    else:
        interp = "Slight"

    df_a = pd.Series(scores_a, name=label_a)
    df_b = pd.Series(scores_b, name=label_b)
    confusion = pd.crosstab(df_a, df_b)

    return {
        'kappa':          kappa,
        'interpretation': interp,
        'confusion':      confusion,
        'n':              len(scores_a),
    }


# ─────────────────────────────────────────────────────────────
# Per-document breakdown
# ─────────────────────────────────────────────────────────────

def per_document_f1(results_log: List[Dict]) -> pd.DataFrame:
    """
    Compute mean F1 and question count per document source.

    Args:
        results_log: List of dicts with keys [doc_source, f1]

    Returns:
        DataFrame sorted by mean F1 descending
    """
    df = pd.DataFrame(results_log)
    summary = df.groupby('doc_source')['f1'].agg(
        mean_f1='mean', count='count'
    ).sort_values('mean_f1', ascending=False)
    return summary


# ─────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: List[float],
    metric_fn: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        scores:     List of per-query scores
        metric_fn:  Aggregation function (default: np.mean)
        n_bootstrap: Number of bootstrap samples
        ci:         Confidence level (default: 0.95)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    scores_arr = np.array(scores)
    point_est  = metric_fn(scores_arr)
    bootstrap  = [
        metric_fn(np.random.choice(scores_arr, size=len(scores_arr),
                                   replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap, alpha * 100)
    upper = np.percentile(bootstrap, (1 - alpha) * 100)
    return point_est, lower, upper


# ─────────────────────────────────────────────────────────────
# Quick evaluation report
# ─────────────────────────────────────────────────────────────

def print_report(metrics: Dict, title: str = "Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        if k in ['results_log', 'log', 'scores', 'confusion']:
            continue
        if isinstance(v, float):
            print(f"  {k:<25} {v:.4f}")
        else:
            print(f"  {k:<25} {v}")
    print(f"{'=' * 50}\n")


if __name__ == '__main__':
    # Quick smoke test
    print("evaluate.py loaded successfully.")
    print("Available functions:")
    print("  token_f1(prediction, ground_truth)")
    print("  evaluate_regulationqa(df, retrieve_fn)")
    print("  evaluate_obliqua(test_data, retrieve_fn)")
    print("  llm_evaluate(results_log, model_name)")
    print("  compute_kappa(scores_a, scores_b)")
    print("  per_document_f1(results_log)")
    print("  bootstrap_ci(scores)")
    print("  print_report(metrics)")
