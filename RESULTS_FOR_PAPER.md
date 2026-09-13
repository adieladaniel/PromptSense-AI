# PromptSense AI — Real Experimental Results

All numbers below were produced by actually running the completed PromptSense AI
pipeline (`analyzer.py` + `preprocessing.py` + `semantic_analyzer.py` + `optimizer.py`)
on a fixed 18-prompt benchmark (`benchmark.py`), plus a real before/after LLM
experiment (`llm_eval.py`) using Google Gemini. Nothing here is estimated or
invented — every figure traces back to `results.json`, `summary.json`, or
`llm_eval_results.json` in this repo.

## 1. Methodology recap

- **Benchmark set:** 18 prompts across 3 tiers (6 each): *Vague*, *Moderate*,
  *Well-Structured* — spanning the range from underspecified to fully
  prompt-engineered instructions.
- **Assessment:** each prompt (and each of 7 optimized rewrites of it) is scored
  by the hybrid Prompt Quality Assessment module — a 50/50 blend of rule-based
  pattern matching and DeBERTa-v3 (a BERT-family transformer) zero-shot
  semantic scoring — on 8 core criteria (clarity, context, role, audience,
  constraints, output format, specificity, ambiguity), averaged into a 0–100
  overall score. An earlier revision used DistilBERT-MNLI, which turned out
  badly miscalibrated (see `README_SEMANTIC_UPGRADE.md` for the empirical
  before/after); all numbers below are from the corrected DeBERTa-v3 model.
- **Preprocessing:** real tokenization, sentence segmentation, POS/dependency
  parsing (spaCy `en_core_web_sm`) and Flesch-Kincaid readability (`textstat`)
  feed the ambiguity and complexity metrics.
- **Optimization:** the optimizer consumes the assessment's own weakness
  findings (score < 7 per category) and enriches only those categories —
  it is not an independent second guess.
- **Additional insight metrics** (token usage via `tiktoken`, prompt
  complexity, hallucination risk) are informational, not part of the 0–100
  score, matching the paper's Objective 6.
- **Real LLM comparison:** for a subset of prompts, both the original and the
  best-scoring optimized rewrite were sent to Google Gemini
  (`gemini-2.5-flash-lite`) for actual response generation, then each response
  was scored by an independent Gemini judge call against the specific
  instruction that produced it (see §4 limitations for why this matters).

## 2. Table 1 — Mean Overall Quality Score (0–100) by Tier × Technique

| Tier | Original | Standard | Role-Based | Chain-of-Thought | Few-Shot | Structured | Zero-Shot | Expert |
|---|---|---|---|---|---|---|---|---|
| Vague | 24.8 | 24.8 | 73.7 | 70.8 | 71.3 | 60.0 | 50.5 | **77.5** |
| Moderate | 36.8 | 36.8 | 77.2 | 73.7 | 72.8 | 66.0 | 56.2 | **81.5** |
| Well-Structured | 79.2 | 79.2 | 83.8 | 79.0 | 81.8 | 76.5 | 70.5 | **84.3** |

**Observation:** under the corrected hybrid scorer, **Expert Prompt is now the
single best-scoring technique across all three tiers** — a cleaner, more
reportable finding than the earlier (miscalibrated-model) result, which had
inconsistently split the win between "Expert" and "Structured." Chain-of-Thought
and Few-Shot now score much closer to Role-Based/Structured than before,
since the semantic layer credits their reasoning/example scaffolding as
satisfying clarity and specificity even without explicit role/context
boilerplate. Zero-Shot consistently scores lowest among the optimized
techniques — it deliberately strips the assessment-rewarded boilerplate down
to a bare instruction, so this is expected, not a bug. This remains a
genuine, reportable finding: not every prompting technique is equally
effective at raising the assessed quality score.

## 3. Table 2 — Original vs. the Single Best-Scoring Technique per Tier

The best technique is chosen once per tier (by overall score); every other
metric below is reported for that *same* technique — not re-optimized
independently per metric, which would be an incoherent comparison.

| Tier | Best Technique | Score | Ambiguity (0–10, higher=clearer) | Hallucination Risk (0–10, lower=better) | Complexity (0–10, informational) | Token Usage |
|---|---|---|---|---|---|---|
| Vague | Expert Prompt | 24.8 → 77.5 | 4.45 → 5.82 | 6.70 → 2.28 | 3.21 → 7.01 | 4.7 → 112.7 |
| Moderate | Expert Prompt | 36.8 → 81.5 | 5.28 → 6.56 | 4.72 → 1.91 | 4.55 → 6.57 | 11.2 → 111.0 |
| Well-Structured | Expert Prompt | 79.2 → 84.3 | 5.90 → 5.62 | 1.95 → 1.92 | 3.96 → 5.02 | 44.2 → 97.0 |

**Observations:**
- Hallucination risk (heuristic proxy — see limitations) drops sharply after
  optimization for Vague and Moderate tiers; the Well-Structured tier barely
  moves (1.95 → 1.92) because the original prompt was already well-grounded
  before optimization — there was little risk left to remove.
- Complexity and token usage both rise substantially for Vague/Moderate — a
  real, expected cost of optimization (longer, more structured prompts
  consume more tokens and read at a higher grade level). The Well-Structured
  tier's token increase is smaller in relative terms since its original
  prompt was already fairly long.
- Ambiguity actually *drops slightly* after optimization in the
  Well-Structured tier (5.90 → 5.62) — a genuinely counterintuitive result
  worth flagging rather than smoothing over: the Expert Prompt template's
  extra scaffolding text apparently reads as marginally less semantically
  precise to the assessment even though it adds real content, unlike the
  clear ambiguity improvement seen in the other two tiers.

## 4. Table 3 — Real LLM (Gemini) Response Quality: Original vs. Optimized Prompt

**Sample size: n = 3 prompts** (Vague-1, Vague-2, Vague-3). This is a **pilot**,
not the full 18-prompt set — Google's Gemini free-tier quota for this API key
turned out to be capped at 20 requests/day *per model*, and we exhausted that
across four different free models in one session (each prompt pair costs 4
API calls: 2 generations + 2 judge calls). The methodology is fully built and
tested (`llm_eval.py`, with resume support) and can be extended to the
remaining 15 prompts once more quota is available; the numbers below are 100%
real for the 3 prompts actually run.

| Metric | Original response | Optimized response | Delta |
|---|---|---|---|
| Relevance (1–10) | 10.00 | 10.00 | +0.00 |
| Completeness (1–10) | 9.00 | 9.00 | +0.00 |
| Clarity (1–10) | 9.33 | 9.33 | +0.00 |
| Format adherence (1–10) | 10.00 | 10.00 | +0.00 |
| Unsupported claims (count) | 0.00 | 0.00 | +0.00 |
| **Response length (words)** | **915** | **177** | **−739** |

**Objective (non-LLM-judged) word-limit compliance:** 3/3 optimized responses
stayed within the word limit their own prompt requested; the original,
unconstrained prompts had no limit to violate.

**Honest interpretation:** Gemini is a strong enough model that it produces
subjectively "good" (high judge-score) responses even to vague prompts — the
LLM-judge scores alone don't show a quality gap for this small sample. The
gap that *is* real and measurable is **conciseness and constraint
adherence**: the optimized prompts reliably produced responses ~5× shorter
and matching the explicitly requested word budget, while the original vague
prompts produced long, unconstrained answers with no way to check compliance
because they specified nothing. This is a legitimate, defensible result for
the paper's response-quality claim, but it should be reported as "prompt
optimization measurably improves controllability and constraint-following,"
rather than an overstated "responses are dramatically better" claim the
n=3 judge scores don't actually support.

## 5. Qualitative example (Vague-1)

**Original prompt:** `"Explain about AR VR"`
→ Gemini response: **878 words**, no requested structure (none was specified).

**Optimized prompt (Expert Prompt technique):**
```
You are a world-class expert with 20+ years of experience in this field.
Explain about AR VR

Context: This is intended as a clear, educational explanation for someone
encountering the topic for the first time.
Audience: A general audience with no prior background in this topic.
Constraints: Keep the explanation clear and concise, ideally under 250 words.
Output Format: Use bullet points or short paragraphs, and end with a
one-line summary.

Provide a nuanced, expert-level explanation, but keep it accessible to
someone learning the topic for the first time.
```
→ Gemini response: **171 words**, in bullet-point format, within the
requested 250-word limit.

## 6. Limitations (for the paper's Discussion / Threats to Validity section)

- **Hallucination risk is a heuristic proxy**, computed from
  ambiguity/specificity/context/constraint scores — not a measured rate of
  actual fabricated content in LLM output. Table 3's `unsupported_claims_count`
  from the Gemini judge is the closer-to-empirical signal, but was 0 for both
  conditions in this small sample (n=3), so it does not yet show a
  hallucination-risk difference empirically — only the static heuristic does.
- **LLM comparison sample size is 3, not 18**, due to free-tier API quota
  (20 requests/day/model). Treat Table 3 as a pilot demonstrating the
  methodology works, not a statistically powered result.
- **The LLM judge is itself an LLM** (Gemini judging Gemini), a known source
  of leniency bias in LLM-as-judge setups; the objective, code-computed
  word-count compliance check was added specifically to have at least one
  non-LLM-judged, verifiable metric.
- **"Best technique" is a property of the assessment module's current
  scoring, not an absolute truth about prompting.** Expert Prompt currently
  wins across all three tiers, but this is the model-scored result of one
  specific hybrid scorer (rule-based + DeBERTa-v3); it shifted meaningfully
  when the semantic model was corrected (see §1), so the paper should frame
  it as "best according to this assessment methodology," not a universal
  claim about prompt-engineering technique quality.
- **Reproducibility note:** the semantic scorer runs ~50–100x faster on a
  CUDA-capable GPU than on CPU (verified: ~0.12s/call on an RTX 3050 vs.
  several seconds/call on CPU). `benchmark.py`'s 144 assessment calls take
  well under a minute on GPU vs. ~25–30 minutes on CPU — worth noting if
  reproducing this table on different hardware.
