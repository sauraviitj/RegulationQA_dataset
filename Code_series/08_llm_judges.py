# %% [markdown]
# # STEP 08 - LLM-as-judge (paper Table: Qwen3-4B and Mistral-7B judges, agreement)
# **What this does:** two independent LLM judges read (question, reference answer, extracted span) and output 0 / 1 / 2
# (incorrect / partially correct / fully correct). Reported: % correct-or-partial per system, mean score, and Cohen's kappa
# between the two judges. Systems judged: Full pipeline, BGE only, BM25 only (the saved `*_extraction_log.json` files).
#
# **Judged span:** `best_span` of each log item = the ORACLE best-of-3 span (it was chosen with the reference answer). So these
# percentages measure answer availability within the top-3, not a deployable top-1 system. Say so in the paper text.
# **Prompt:** identical for both judges (below). Greedy decoding, `max_new_tokens=5`, the first digit 0/1/2 in the reply is the score;
# an empty span or an unparsable reply scores 0.
#
# **Run on:** Kaggle with 2 x T4 (Qwen on GPU 1 in fp16, Mistral on GPU 0 in 4-bit) OR one T4 (set SINGLE_GPU=True: runs one
# model at a time). Internet ON (Hugging Face downloads). **Time:** ~60-90 min per model for 3 x 1,002 spans.
# **Writes:** `llm_evaluation_results.json`, `combined_llm_scores.csv`.

# %%
#!pip install -q transformers accelerate bitsandbytes scikit-learn
from adsir_common import *
import torch, gc
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import cohen_kappa_score
SINGLE_GPU = torch.cuda.device_count() < 2

def load_log(name): return json.load(open(find(name)))
logs = {"Full pipeline": load_log("extraction_log.json"), "BGE only": load_log("bge_extraction_log.json"), "BM25 only": load_log("bm25_extraction_log.json")}
print({k: len(v) for k, v in logs.items()})

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

# %% [markdown]
# ### Scoring function (one span -> 0/1/2)

# %%
def score_one(model, tok, device, question, gt, span):
    if not span or not str(span).strip(): return 0
    prompt = LLM_PROMPT.format(question=question, ground_truth=gt, extracted_span=span)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del inp, out; torch.cuda.empty_cache(); gc.collect()
    for ch in reply:
        if ch in "012": return int(ch)
    return 0

def run_judge(model, tok, device, label):
    res = {}
    for name, log in logs.items():
        sc = [score_one(model, tok, device, it["question"], it["ground_truth"], it.get("best_span", "")) for it in tqdm(log, desc=f"{label}: {name}")]
        a = np.array(sc)
        res[name] = dict(fully_correct=float((a == 2).mean()), partially_correct=float((a == 1).mean()), incorrect=float((a == 0).mean()),
                         correct_partial=float((a >= 1).mean()), mean_score=float(a.mean()), scores=sc)
        print(f"{label:8s} {name:14s} correct+partial {res[name]['correct_partial']*100:5.1f}%  mean {res[name]['mean_score']:.4f}")
    return res

# %% [markdown]
# ### Judge 1 - Qwen3-4B (fp16)

# %%
dev0 = "cuda:0" if SINGLE_GPU else "cuda:1"
q_tok = AutoTokenizer.from_pretrained(QWEN_NAME)
q_model = AutoModelForCausalLM.from_pretrained(QWEN_NAME, torch_dtype=torch.float16, device_map=dev0).eval()
qwen = run_judge(q_model, q_tok, dev0, "Qwen")
if SINGLE_GPU: del q_model; gc.collect(); torch.cuda.empty_cache()           # free the T4 before loading Mistral

# %% [markdown]
# ### Judge 2 - Mistral-7B-Instruct-v0.3 (4-bit NF4)

# %%
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
m_tok = AutoTokenizer.from_pretrained(MISTRAL_NAME)
m_model = AutoModelForCausalLM.from_pretrained(MISTRAL_NAME, quantization_config=qc, device_map="cuda:0").eval()
mistral = run_judge(m_model, m_tok, m_model.device, "Mistral")

# %% [markdown]
# ### Agreement and saving

# %%
kappa = cohen_kappa_score(qwen["Full pipeline"]["scores"], mistral["Full pipeline"]["scores"])
interp = "Almost perfect" if kappa >= .8 else "Substantial" if kappa >= .6 else "Moderate" if kappa >= .4 else "Fair" if kappa >= .2 else "Slight"
print(f"\nCohen kappa (Qwen vs Mistral, full pipeline): {kappa:.4f} ({interp})")
strip = lambda r: {k: {m: v for m, v in d.items() if m != "scores"} for k, d in r.items()}
save_json(dict(qwen=strip(qwen), mistral=strip(mistral), cohen_kappa_full_pipeline=float(kappa), interpretation=interp), "llm_evaluation_results.json")
fp = logs["Full pipeline"]
pd.DataFrame(dict(question=[i["question"] for i in fp], ground_truth=[i["ground_truth"] for i in fp], best_span=[i.get("best_span", "") for i in fp],
                  f1=[i.get("f1", 0) for i in fp], qwen_score=qwen["Full pipeline"]["scores"], mistral_score=mistral["Full pipeline"]["scores"])).to_csv(WORK / "combined_llm_scores.csv", index=False)
