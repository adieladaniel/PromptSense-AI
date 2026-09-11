"""
semantic_analyzer.py

Transformer-based semantic scoring component for PromptSense AI.

This implements the semantic half of the hybrid design described in the
project's Design Methodology: "the semantic component, implemented using a
transformer model such as DistilBERT, captures the contextual meaning and
intent of the prompt, while the rule-based component evaluates the presence
of essential prompt engineering elements."

Approach
--------
Rather than a plain embedding-similarity trick, this uses a DeBERTa-v3
model (DeBERTa = "Decoding-enhanced BERT with disentangled attention" --
architecturally a direct descendant of BERT) fine-tuned on combined NLI
datasets (MNLI + Fever-NLI + ANLI) and specifically validated for
zero-shot classification, in a zero-shot-classification setup. For each of
the seven quality criteria, we phrase a natural-language "hypothesis"
describing what a prompt satisfying that criterion looks like, and ask the
model how strongly the prompt (as the "premise") entails each hypothesis.
The entailment probability becomes that criterion's semantic score (0-10).

Earlier revision used typeform/distilbert-base-uncased-mnli: a 66M-param
distilled model that turned out badly miscalibrated for this task in
practice (e.g. scoring "role"/"context"/"constraints" near-maximum for a
prompt containing none of them, and near-zero for a fully prompt-engineered
one -- verified empirically, not a hunch). DeBERTa-v3-base (~184M params)
is both larger and, more importantly, was fine-tuned and benchmarked
specifically for zero-shot classification rather than repurposed from a
plain MNLI classifier, which is what actually fixes the miscalibration.

This captures cases the regex rules in analyzer.py miss -- e.g. a prompt
that clearly assigns a persona without using any of the literal phrases in
ROLE_PATTERNS ("act as", "you are a", etc.) can still score well here,
because the model is judging meaning, not surface pattern matches.

Requires: transformers, torch  (pip install transformers torch)
First call downloads the ~370MB DeBERTa-v3-base checkpoint from the
Hugging Face Hub, so an internet connection is required once; the model is
then cached under ~/.cache/huggingface and every later call runs offline.

Design note: the classifier is loaded lazily (only on the first real call
to semantic_scores), and every call is wrapped so that any failure --
missing packages, no internet on first run, OOM, etc. -- returns None
instead of raising. analyzer.py uses that None to fall back to pure
rule-based scoring, so the app never breaks if the transformer backend
isn't available in a given environment.
"""

from functools import lru_cache

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# One natural-language hypothesis per quality criterion, matching the
# categories analyzer.py already scores via regex.
CRITERIA_HYPOTHESES = {
    "clarity": "This text clearly and specifically states what task should be performed.",
    "context": "This text provides background context or situational information.",
    "role": "This text assigns a specific role, persona, or expertise to the AI.",
    "audience": "This text specifies who the response is intended for.",
    "constraints": "This text specifies constraints such as length, tone, or scope.",
    "output": "This text specifies a desired output format, such as bullet points, a table, or headings.",
    "specificity": "This text is specific and concrete rather than vague or generic.",
    "ambiguity": "This text has one single, precise, well-defined meaning.",
}


@lru_cache(maxsize=1)
def _get_classifier():
    """
    Lazily builds the zero-shot-classification pipeline on first use.
    Cached (lru_cache) so the ~260MB model is loaded into memory once per
    process, not on every call. Importing this module never triggers a
    download by itself -- only calling semantic_scores() does.
    """
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=MODEL_NAME)


def semantic_backend_available() -> bool:
    """Cheap check used by the UI to show whether semantic scoring is active."""
    try:
        _get_classifier()
        return True
    except Exception:
        return False


def semantic_scores(prompt: str) -> dict | None:
    """
    Returns {criterion: 0-10 float} semantic scores for the prompt, or
    None if the transformer backend could not be loaded (no internet on
    first run, transformers/torch not installed, etc.) -- callers should
    treat None as "fall back to rule-based scoring only".
    """
    if not prompt or not prompt.strip():
        return {k: 0.0 for k in CRITERIA_HYPOTHESES}

    try:
        classifier = _get_classifier()
    except Exception:
        return None

    try:
        labels = list(CRITERIA_HYPOTHESES.values())
        result = classifier(prompt, candidate_labels=labels, multi_label=True)
        label_to_prob = dict(zip(result["labels"], result["scores"]))
        return {
            criterion: round(label_to_prob.get(hypothesis, 0.0) * 10, 2)
            for criterion, hypothesis in CRITERIA_HYPOTHESES.items()
        }
    except Exception:
        return None
