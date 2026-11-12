# %% [markdown]
# # STEP 09 - Hallucination / faithfulness with NLI (paper: "full chunk vs extracted span" generation experiment)
# **What this does:** for 100 questions, Qwen3-4B writes an answer once from the FULL top-1 chunk (~500 words) and once from the
# EXTRACTED span (~50 words). Each sentence of each generated answer is then checked against the context it was given with an
# NLI model (premise = context, hypothesis = answer sentence): entailment = supported, neutral = unsupported, contradiction.
# Paper values (original run): faithfulness 14.7% -> 43.1%, neutral 71.7% -> 44.1%, contradiction 13.6% -> 12.8%.
#
# **IMPORTANT (honest note):** the original generated-answer file was not recovered. This step is therefore a FRESH RE-RUN, not a
# recovery. The 100 questions are drawn with a fixed seed (42), the generation prompt below is a reasonable reconstruction, and the
# numbers will differ somewhat from the paper. If you quote the paper's numbers, say they come from the original run; if you quote
# this step's numbers, say they come from the re-run. Never mix the two in one table.
#
# **Run on:** Kaggle GPU T4, Internet ON. Needs steps 02-03 (`rqa_per_question.csv`, cache). **Time:** ~25 min. **Writes:** `nli_generations.csv`, `nli_summary.json`.

# %%
from adsir_common import *
import torch, gc
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize
import nltk; nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
cache = load_cache("rqa_cache"); rqa = load_regulationqa(); R = load_rqa_resources(load_models=False); chunks = R["chunks"]
per_q = pd.read_csv(WORK / "rqa_per_question.csv")
N_Q = 100
rows = np.random.default_rng(42).choice(len(rqa), size=N_Q, replace=False).tolist()
dev = "cuda"

GEN_PROMPT = """Answer the question using ONLY the context below. Be concise (2-3 sentences).
If the context does not contain the answer, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# %% [markdown]
# ### 1. Generate with Qwen3-4B from two contexts (full chunk vs extracted top-1 span)

# %%
tok = AutoTokenizer.from_pretrained(QWEN_NAME)
llm = AutoModelForCausalLM.from_pretrained(QWEN_NAME, torch_dtype=torch.float16, device_map=dev).eval()
def generate(question, context):
    prompt = GEN_PROMPT.format(context=context, question=question)
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)   # no <think> block
    inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
    with torch.no_grad(): out = llm.generate(**inp, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

recs = []
for r in tqdm(rows):
    q = str(rqa.iloc[r]["question"]); c = next(x for x in cache if x["row"] == r)
    full_ctx = chunks[c["cands"][0]["idx"]]["text"] if c["cands"] else ""        # top-1 candidate chunk (full)
    span_ctx = str(per_q.loc[per_q.row == r, "sv_top1"].iloc[0])                  # extracted top-1 span (step 03)
    recs.append(dict(row=r, question=q, full_ctx=full_ctx, span_ctx=span_ctx,
                     ans_full=generate(q, full_ctx), ans_span=generate(q, span_ctx)))
gen = pd.DataFrame(recs); gen.to_csv(WORK / "nli_generations.csv", index=False)
print("avg context words: full", gen.full_ctx.str.split().str.len().mean().round(1), "| span", gen.span_ctx.str.split().str.len().mean().round(1))
del llm; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ### 2. NLI check of every answer sentence against its own context
# `cross-encoder/nli-deberta-v3-base` outputs logits in the order **[contradiction, entailment, neutral]**.

# %%
nli = CrossEncoder(NLI_NAME, device=dev); NAMES = ["contradiction", "entailment", "neutral"]
def classify(answers, contexts):
    labels = []
    for a, ctx in zip(answers, contexts):
        ss = sent_tokenize(str(a))
        if not ss: continue
        sc = np.asarray(nli.predict([[ctx, s] for s in ss]))
        labels += [NAMES[int(np.argmax(x))] for x in sc]
    return labels
def summarize(labels):
    c = Counter(labels); n = max(sum(c.values()), 1)
    return dict(faithful_entailment_pct=100 * c["entailment"] / n, neutral_pct=100 * c["neutral"] / n, contradiction_pct=100 * c["contradiction"] / n, n_sentences=n)
summ = {"full chunk": summarize(classify(gen.ans_full, gen.full_ctx)), "extracted span": summarize(classify(gen.ans_span, gen.span_ctx))}
print(pd.DataFrame(summ).T.round(1).to_string()); print("paper (original run): full 14.7 / 71.7 / 13.6  ->  span 43.1 / 44.1 / 12.8")
save_json(summ, "nli_summary.json")
