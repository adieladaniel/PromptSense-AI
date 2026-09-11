"""
analyzer.py

Hybrid Prompt Engineering evaluator: combines the original rule-based
pattern matching (fast, deterministic, zero dependencies) with a
transformer-based semantic component (semantic_analyzer.py, DistilBERT
fine-tuned on MNLI) as described in the project's Design Methodology.

Core quality metrics returned (each 0-10), matching the paper's Objective 2
list (clarity, specificity, contextual completeness, role definition,
constraints, audience specification, output formatting, ambiguity):
    clarity, context, role, audience, constraints, output, specificity,
    ambiguity
Overall "score" (0-100) is the average of those eight metrics.

Additional insight metrics (paper's Objective 6: "token usage, prompt
complexity, and potential hallucination risk") are NOT part of the score --
they are informational only:
    token_usage, complexity, hallucination_risk

Scoring mode:
    - If the semantic backend loads successfully, each core metric is a
      50/50 blend of the rule-based score and the DistilBERT-MNLI semantic
      score, and result["method"] == "hybrid".
    - If the semantic backend is unavailable (no internet on first run,
      transformers/torch not installed, etc.), every core metric falls
      back to the pure rule-based score, and result["method"] ==
      "rule_based_only". The app keeps working either way -- this is a
      graceful degradation, not a failure.
"""

import re

# On Windows, torch and textstat's dependency chain each bundle a
# same-named native runtime DLL (e.g. an OpenMP/MKL runtime); whichever
# loads first "claims" that name in the process, and if textstat claims it
# first, torch's later DLL init fails with WinError 1114. Importing torch
# here, before `preprocessing` pulls in textstat, avoids the conflict. Safe
# no-op if torch isn't installed -- the semantic backend already handles
# that via its own try/except.
try:
    import torch  # noqa: F401
except Exception:
    pass

from preprocessing import preprocess

try:
    from semantic_analyzer import semantic_scores as _semantic_scores
except Exception:
    # semantic_analyzer.py itself failed to import (shouldn't normally
    # happen -- it only imports `transformers` lazily inside a function --
    # but guarded here too so a bad install never takes down the analyzer).
    def _semantic_scores(prompt):
        return None

try:
    import tiktoken
    _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODING = None

# --------------------------------------------------
# PATTERN DEFINITIONS (unchanged rule-based layer)
# --------------------------------------------------

ROLE_PATTERNS = [
    r"\bact as\b",
    r"\byou are (a|an|the)\b",
    r"\bimagine you are\b",
    r"\bas an? [\w\s]{0,25}(expert|specialist|professional|educator|professor|engineer|analyst|consultant)\b",
    r"\bworld[- ]class expert\b",
    r"\brole\s*:",
]

CONTEXT_PATTERNS = [
    r"\bcontext\s*:",
    r"\bi am a\b",
    r"\bi am looking\b",
    r"\bgiven that\b",
    r"\bbackground\s*:",
    r"\bfor context\b",
]

AUDIENCE_PATTERNS = [
    r"\baudience\s*:",
    r"\bbeginner\b",
    r"\bfor a (beginner|student|professional|expert|child|teacher|developer)\b",
    r"(explain|describe|teach)\s+.*\bto a\b",
    r"\bsuitable for\b",
    r"\bno technical background\b",
]

CONSTRAINTS_PATTERNS = [
    r"\bconstraints\s*:",
    r"\blimit\b",
    r"\bunder \d+ words\b",
    r"\bword limit\b",
    r"\bno more than \d+\b",
    r"\bkeep it (short|concise|brief)\b",
    r"\bwithin \d+\b",
]

FORMAT_PATTERNS = [
    r"\bformat\s*:",
    r"\boutput format\b",
    r"\bbullet points?\b",
    r"\bheadings?\b",
    r"\bstructured\b",
    r"\bsummary\b",
    r"\btable\b",
    r"\bnumbered list\b",
]

REASONING_PATTERNS = [
    r"\bstep by step\b",
    r"\bthink through\b",
    r"\breasoning\b",
    r"\bfirst.*then.*finally\b",
]

TASK_VERBS = [
    "explain", "describe", "summarize", "write", "generate", "create",
    "list", "compare", "analyze", "define", "teach", "outline",
]

VAGUE_WORDS = ["something", "stuff", "things", "somehow", "anything", "whatever", "etc"]

# Blend weight: 0.5 means rule-based and semantic contribute equally.
SEMANTIC_WEIGHT = 0.5


def _hits(text, patterns, cap=3):
    """Number of distinct patterns matched, capped to avoid gaming via repetition."""
    return min(sum(1 for p in patterns if re.search(p, text, re.IGNORECASE)), cap)


def _score_from_hits(hit_count):
    """Map a capped hit count (0-3) to a 0-10 score."""
    mapping = {0: 0, 1: 7, 2: 9, 3: 10}
    return mapping[hit_count]


def _clarity_score(prompt, text):
    """Clarity: a clear task verb + a reasonable, non-trivial length."""
    word_count = len(prompt.split())
    has_task_verb = any(re.search(rf"\b{v}\b", text) for v in TASK_VERBS)

    if word_count < 4:
        length_component = 2
    elif word_count < 10:
        length_component = 6
    elif word_count <= 180:
        length_component = 10
    else:
        length_component = 8  # very long prompts lose a bit of clarity

    if not has_task_verb:
        length_component = round(length_component * 0.5)

    return max(0, min(10, length_component))


def _specificity_score(prompt, text):
    """Specificity: concrete details (numbers, named subject) vs. vague filler words."""
    word_count = len(prompt.split())
    has_vague = any(re.search(rf"\b{w}\b", text) for w in VAGUE_WORDS)
    has_number = bool(re.search(r"\d", prompt))

    score = 6  # baseline
    if word_count >= 8:
        score += 2
    if has_number:
        score += 2
    if has_vague:
        score -= 4

    return max(0, min(10, score))


def _ambiguity_score(prompt, text, features) -> float:
    """
    Ambiguity (0-10, higher = LESS ambiguous / clearer -- same "higher is
    better" direction as the other core metrics). Distinct from
    `specificity`: specificity rewards concrete detail being present,
    ambiguity penalizes structural/referential signals that make a prompt
    open to multiple interpretations:
      - unresolved pronoun density (spaCy POS tagging via preprocessing.py)
      - vague filler words
      - too many competing task verbs in one prompt (unclear which task
        takes priority)
    """
    word_count = max(features["word_count"], 1)
    pronoun_ratio = features["pronoun_count"] / word_count
    vague_hits = sum(1 for w in VAGUE_WORDS if re.search(rf"\b{w}\b", text))
    task_verb_hits = sum(1 for v in TASK_VERBS if re.search(rf"\b{v}\b", text))

    score = 10.0
    if pronoun_ratio > 0.15:
        score -= 3
    if vague_hits >= 1:
        score -= min(vague_hits * 2, 6)
    if task_verb_hits >= 3:
        score -= 2  # multiple competing instructions in a single prompt
    if features["sentence_count"] == 0:
        score -= 5

    return max(0.0, min(10.0, score))


def _rule_based_metrics(prompt: str, text: str, features: dict) -> dict:
    role = _score_from_hits(_hits(text, ROLE_PATTERNS))
    context = _score_from_hits(_hits(text, CONTEXT_PATTERNS))
    audience = _score_from_hits(_hits(text, AUDIENCE_PATTERNS))
    constraints = _score_from_hits(_hits(text, CONSTRAINTS_PATTERNS))
    output = _score_from_hits(_hits(text, FORMAT_PATTERNS))
    clarity = _clarity_score(prompt, text)
    specificity = _specificity_score(prompt, text)
    ambiguity = _ambiguity_score(prompt, text, features)

    reasoning_bonus = 1 if _hits(text, REASONING_PATTERNS) > 0 else 0
    clarity = min(10, clarity + reasoning_bonus)

    return {
        "clarity": clarity,
        "context": context,
        "role": role,
        "audience": audience,
        "constraints": constraints,
        "output": output,
        "specificity": specificity,
        "ambiguity": ambiguity,
    }


def _token_usage(prompt: str, features: dict) -> int:
    """
    Approximate LLM token count via tiktoken's cl100k_base encoding (the
    standard proxy used across the industry/literature for
    model-agnostic token estimates). Falls back to the spaCy token count
    from preprocessing.py if tiktoken isn't available.
    """
    if _TOKEN_ENCODING is not None:
        return len(_TOKEN_ENCODING.encode(prompt))
    return features["token_count"]


def _complexity_score(features: dict) -> float:
    """
    Prompt complexity (0-10, informational -- NOT a quality judgment).
    Composite of average sentence length, dependency-tree depth, and
    Flesch-Kincaid grade level, each min-max normalized against thresholds
    typical of prompt-length text before averaging.
    """
    sentence_len_component = min(features["avg_sentence_length"] / 25.0, 1.0) * 10
    depth_component = min(features["avg_dependency_depth"] / 8.0, 1.0) * 10
    grade_component = min(max(features["flesch_kincaid_grade"], 0) / 18.0, 1.0) * 10

    return round((sentence_len_component + depth_component + grade_component) / 3, 2)


def _hallucination_risk(metrics: dict, features: dict) -> float:
    """
    Hallucination risk (0-10, higher = MORE risk -- inverted direction vs.
    the quality metrics, since this represents a risk estimate, not a
    quality score). Heuristic proxy based on prompt-quality signals the
    literature associates with underspecified prompts inviting the model
    to fill gaps with invented content: low grounding context, missing
    constraints, low specificity, and high ambiguity. This is NOT an
    empirical hallucination rate measured from actual model outputs --
    see benchmark.py's optional Gemini-based judge pass for that.
    """
    badness_context = 10 - metrics["context"]
    badness_constraints = 10 - metrics["constraints"]
    badness_specificity = 10 - metrics["specificity"]
    badness_ambiguity = 10 - metrics["ambiguity"]

    risk = (
        0.3 * badness_ambiguity
        + 0.3 * badness_specificity
        + 0.2 * badness_context
        + 0.2 * badness_constraints
    )

    if features["word_count"] < 6:
        risk += 1

    return round(max(0.0, min(10.0, risk)), 2)


def analyze_prompt(prompt: str, use_semantic: bool = True) -> dict:
    """
    Returns a dict with:
        score, clarity, context, role, audience, constraints, output,
        specificity, ambiguity, token_usage, complexity,
        hallucination_risk, strengths, weaknesses, suggestions, method
    """
    if not prompt or not prompt.strip():
        return {
            "score": 0,
            "clarity": 0, "context": 0, "role": 0, "audience": 0,
            "constraints": 0, "output": 0, "specificity": 0, "ambiguity": 0,
            "token_usage": 0, "complexity": 0, "hallucination_risk": 0,
            "features": {},
            "strengths": [], "weaknesses": ["No prompt provided."],
            "suggestions": ["Please enter a prompt to analyze."],
            "method": "n/a",
        }

    text = prompt.lower()
    features = preprocess(prompt)
    rule_metrics = _rule_based_metrics(prompt, text, features)

    semantic_metrics = _semantic_scores(prompt) if use_semantic else None

    if semantic_metrics:
        metrics = {
            k: round(
                (1 - SEMANTIC_WEIGHT) * rule_metrics[k] + SEMANTIC_WEIGHT * semantic_metrics[k],
                2,
            )
            for k in rule_metrics
        }
        method = "hybrid"
    else:
        metrics = rule_metrics
        method = "rule_based_only"

    score = round(sum(metrics.values()) / (10 * len(metrics)) * 100)

    strengths = []
    weaknesses = []
    suggestions = []

    def _flag(pts, strength_msg, weakness_msg, suggestion_msg):
        if pts >= 7:
            strengths.append(strength_msg)
        else:
            weaknesses.append(weakness_msg)
            suggestions.append(suggestion_msg)

    _flag(
        metrics["role"],
        "✅ Clear role/persona assigned to the AI.",
        "❌ No specific role or persona assigned.",
        "💡 Try adding a role, e.g. 'Act as an experienced X.'",
    )
    _flag(
        metrics["context"],
        "✅ Good background context provided.",
        "❌ Missing background context.",
        "💡 Add context, e.g. 'Context: I am a beginner learning X.'",
    )
    _flag(
        metrics["audience"],
        "✅ Target audience clearly specified.",
        "❌ Target audience is not specified.",
        "💡 Mention who the response is for, e.g. 'Audience: complete beginners.'",
    )
    _flag(
        metrics["constraints"],
        "✅ Clear constraints defined (length, tone, scope).",
        "❌ No constraints such as length or tone are defined.",
        "💡 Add constraints, e.g. 'Keep it under 300 words.'",
    )
    _flag(
        metrics["output"],
        "✅ Explicit output format requested.",
        "❌ No specific output format requested.",
        "💡 Specify a format, e.g. 'Use bullet points and end with a summary.'",
    )
    _flag(
        metrics["clarity"],
        "✅ Task is clearly and concretely stated.",
        "❌ The task itself could be stated more clearly.",
        "💡 Use a direct task verb (explain/compare/summarize) with a specific subject.",
    )
    _flag(
        metrics["specificity"],
        "✅ Prompt is specific and concrete rather than vague.",
        "❌ Prompt is a bit vague or generic.",
        "💡 Replace vague words (e.g. 'stuff', 'things') with concrete details.",
    )
    _flag(
        metrics["ambiguity"],
        "✅ Prompt is unambiguous and has a single clear interpretation.",
        "❌ Prompt is ambiguous (unclear referents or competing instructions).",
        "💡 Resolve pronouns ('it', 'this') into concrete nouns and state one task at a time.",
    )

    token_usage = _token_usage(prompt, features)
    complexity = _complexity_score(features)
    hallucination_risk = _hallucination_risk(metrics, features)

    if hallucination_risk >= 7:
        weaknesses.append("❌ High estimated hallucination risk -- the model has too much room to invent details.")
        suggestions.append("💡 Add grounding context, constraints, and specific details to reduce hallucination risk.")
    elif hallucination_risk <= 3:
        strengths.append("✅ Low estimated hallucination risk -- prompt is well-grounded.")

    return {
        "score": score,
        "clarity": metrics["clarity"],
        "context": metrics["context"],
        "role": metrics["role"],
        "audience": metrics["audience"],
        "constraints": metrics["constraints"],
        "output": metrics["output"],
        "specificity": metrics["specificity"],
        "ambiguity": metrics["ambiguity"],
        "token_usage": token_usage,
        "complexity": complexity,
        "hallucination_risk": hallucination_risk,
        "features": features,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "method": method,
    }
