# PromptSense AI — Transformer Semantic Scoring Upgrade

## What changed
- **New file:** `semantic_analyzer.py` — loads `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  (DeBERTa-v3, a direct architectural descendant of BERT, fine-tuned on
  combined NLI datasets and validated specifically for zero-shot
  classification) via HuggingFace `transformers`, and uses zero-shot
  classification to score each of the 8 quality criteria based on semantic
  entailment, not just regex keyword matches.
- **Modified:** `analyzer.py` — `analyze_prompt()` now blends the original
  rule-based score with the new semantic score (50/50) for each criterion.
  If the semantic backend can't load (no internet, packages missing), it
  automatically falls back to the original rule-based-only scoring — the
  app never breaks. It also imports `torch` defensively before `preprocessing`
  loads `textstat`, working around a Windows-only bug where the two
  packages' bundled native DLLs collide and crash whichever loads torch
  second (see git history on `analyzer.py` for details).
- **Modified:** `app.py` — shows which scoring mode was actually used
  ("Hybrid" vs "Rule-based only") so it's never ambiguous which one ran.
- **Unchanged:** `optimizer.py` (template-based prompt rewriting).

### Model revision history
The first version of this upgrade used `typeform/distilbert-base-uncased-mnli`
(66M params, ~260MB). In practice it turned out badly miscalibrated for this
task — e.g. scoring "role"/"context"/"constraints" near-maximum for a prompt
containing none of them, and near-zero for a fully prompt-engineered one,
verified empirically against a 3-tier benchmark. It was replaced with
DeBERTa-v3-base (~184M params, ~370MB), which is both larger and, more
importantly, was actually fine-tuned and benchmarked for zero-shot
classification rather than repurposed from a plain MNLI classifier — this is
what fixed the miscalibration. Blended (hybrid) scores across the vague /
moderate / well-structured benchmark tiers now correctly order low → high,
matching the rule-based-only layer's ordering.

## Setup
```bash
pip install -r requirements.txt
```
First run of `analyze_prompt()` downloads the ~370MB DeBERTa-v3-base
checkpoint (needs internet, one time). After that it's cached locally
(`~/.cache/huggingface`) and runs fully offline.

## Running the app
```bash
streamlit run app.py
```

## Expected behavior change vs. the old rule-based-only version
- Latency per `analyze_prompt()` call goes from ~0.3 ms to roughly
  **0.2–1 second on CPU** (one zero-shot classification forward pass per
  prompt, across 8 candidate labels), since it now does real inference
  instead of only regex matching. This is the real, expected trade-off of
  adding the transformer component your Design Methodology describes —
  it's slower but semantically aware.
- Scores can shift for prompts that satisfy a criterion in wording the
  regex patterns don't cover — e.g. a persona set in a way that
  doesn't match `ROLE_PATTERNS` literally.

## Getting real numbers for your report
Run `python benchmark.py` (the same 18-prompt benchmark used in the
rule-based-only Results section) after `pip install -r requirements.txt`,
then `python summarize_results.py` to aggregate `results.json` into the
tables used in `RESULTS_FOR_PAPER.md`.
