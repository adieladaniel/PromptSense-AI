# PromptSense AI — DistilBERT Semantic Scoring Upgrade

## What changed
- **New file:** `semantic_analyzer.py` — loads `typeform/distilbert-base-uncased-mnli`
  (DistilBERT fine-tuned for NLI) via HuggingFace `transformers`, and uses
  zero-shot classification to score each of the 7 quality criteria based on
  semantic entailment, not just regex keyword matches.
- **Modified:** `analyzer.py` — `analyze_prompt()` now blends the original
  rule-based score with the new semantic score (50/50) for each criterion.
  If the semantic backend can't load (no internet, packages missing), it
  automatically falls back to the original rule-based-only scoring — the
  app never breaks.
- **Modified:** `app.py` — shows which scoring mode was actually used
  ("Hybrid" vs "Rule-based only") so it's never ambiguous which one ran.
- **Unchanged:** `optimizer.py` (template-based prompt rewriting).

## Setup
```bash
pip install -r requirements.txt
```
First run of `analyze_prompt()` downloads the ~260MB DistilBERT-MNLI
checkpoint (needs internet, one time). After that it's cached locally
(`~/.cache/huggingface`) and runs fully offline.

## Running the app
```bash
streamlit run app.py
```

## Expected behavior change vs. the old rule-based-only version
- Latency per `analyze_prompt()` call goes from ~0.3 ms to roughly
  **0.2–1 second on CPU** (one zero-shot classification forward pass per
  prompt, across 7 candidate labels), since it now does real inference
  instead of only regex matching. This is the real, expected trade-off of
  adding the transformer component your Design Methodology describes —
  it's slower but semantically aware.
- Scores can shift for prompts that satisfy a criterion in wording the
  regex patterns don't cover — e.g. a persona set in a way that
  doesn't match `ROLE_PATTERNS` literally.

## To get real numbers for your report
Run the attached `benchmark.py` (same 18-prompt benchmark used in the
rule-based-only Results section) after `pip install -r requirements.txt`,
and send me the resulting `results.json` — I'll rewrite Section V–VII
around the actual hybrid-model output.
