# 🤖 PromptSense AI

**Hybrid rule-based + transformer prompt quality analyzer and optimizer.**
Paste any LLM prompt in and get an 8-dimension quality breakdown, a hallucination-risk
estimate, and seven rewritten versions of it using proven prompt-engineering techniques —
all running locally, with zero external API calls for the analysis itself.

## What it does

1. **Analyzes** a prompt across 8 quality dimensions — clarity, specificity, context,
   role definition, constraints, audience, output format, and ambiguity — using a
   **hybrid scorer**: fast regex/pattern rules blended 50/50 with a local
   **DeBERTa-v3** zero-shot semantic classifier (a BERT-family transformer), so a
   prompt gets credit for satisfying a criterion in wording the regex rules don't
   literally cover.
2. **Reports** informational insight metrics: token usage (`tiktoken`), prompt
   complexity (readability + dependency-tree depth via `spaCy`/`textstat`), and a
   heuristic hallucination-risk estimate.
3. **Optimizes** the prompt into 7 rewritten versions — Standard, Role-Based,
   Chain-of-Thought, Few-Shot, Structured, Zero-Shot, and Expert — generated locally
   from templates, driven by *which specific dimensions the assessment scored weak*
   rather than an independent second guess.
4. Ships with a **benchmark harness** (`benchmark.py`, 18 prompts across 3 difficulty
   tiers) and an optional **real LLM evaluation pipeline** (`llm_eval.py`) that sends
   original vs. optimized prompts to Google Gemini and scores the actual responses —
   see [`RESULTS_FOR_PAPER.md`](RESULTS_FOR_PAPER.md) for real numbers.

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first analysis it downloads the
~370MB DeBERTa-v3 checkpoint (one-time, needs internet); after that it's cached
under `~/.cache/huggingface` and runs fully offline. If the transformer backend
can't load for any reason (no internet, missing packages), the app automatically
falls back to rule-based-only scoring instead of crashing — you'll see which mode
is active in the UI.

Try it with the built-in example buttons in the app (vague / moderate /
well-structured prompts pulled straight from the benchmark set), or paste your own,
e.g.:

```
Act as an experienced data scientist.
Context: I am a beginner learning about regression models.
Audience: someone with no statistics background.
Explain linear regression step by step.
Constraints: keep it under 300 words.
Output Format: use bullet points and end with a summary.
```

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — prompt input, quality dashboard, optimized-version browser |
| `preprocessing.py` | Tokenization, sentence segmentation, POS/dependency parsing (spaCy), readability (`textstat`) |
| `analyzer.py` | Core scoring: rule-based pattern matching blended with the semantic backend |
| `semantic_analyzer.py` | Zero-shot NLI semantic scoring via DeBERTa-v3 |
| `optimizer.py` | Template-based prompt rewriting into 7 prompt-engineering styles |
| `benchmark.py` | Runs the 18-prompt benchmark through the full pipeline → `results.json` |
| `llm_eval.py` | Real before/after response-quality comparison via Google Gemini → `llm_eval_results.json` |
| `summarize_results.py` | Aggregates benchmark + LLM-eval results into report tables → `summary.json` |
| `RESULTS_FOR_PAPER.md` | Real experimental results and limitations, generated from the files above |
| `README_SEMANTIC_UPGRADE.md` | Design notes and history for the hybrid semantic-scoring layer |

## Optional: real LLM evaluation

`llm_eval.py` requires a free-tier Google Gemini API key
([aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

```bash
export GEMINI_API_KEY=...
python llm_eval.py
```

This sends both the original and best-scoring optimized prompt to Gemini, then uses
a second Gemini call as an independent judge (relevance, completeness, clarity,
format adherence, unsupported-claims count). Free-tier quotas are modest (~20
requests/day/model); the script supports `--limit` and resumes from partial progress.

## Known limitations

- Hallucination risk is a heuristic proxy from ambiguity/specificity/context/constraint
  signals — not a measured rate from actual LLM output.
- The semantic scorer, even with a well-calibrated model, is not perfectly precise on
  every dimension (e.g. it can be generous on "context" for any topic-bearing prompt).
  Rule-based scoring alone is more predictable; the hybrid blend trades some of that
  predictability for the ability to recognize criteria satisfied in non-literal wording.
- See [`RESULTS_FOR_PAPER.md`](RESULTS_FOR_PAPER.md) for sample-size and LLM-as-judge
  caveats on the real Gemini evaluation results.
