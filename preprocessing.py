"""
preprocessing.py

Real Preprocessing / Feature-Extraction Module, as described in the paper's
Design Methodology: "text normalization, tokenization, sentence
segmentation, and basic linguistic analysis" followed by extraction of
"prompt length, sentence structure, ... " features.

Uses spaCy (en_core_web_sm) for tokenization, sentence segmentation, POS
tagging, and dependency parsing, and textstat for standard readability
formulas (Flesch Reading Ease / Flesch-Kincaid Grade), rather than the
ad-hoc `prompt.split()` word counting the original analyzer.py relied on.

This module is deliberately kept independent of analyzer.py's regex
category rules -- it only extracts objective linguistic structure. Category
detection (role/context/audience/etc.) stays in analyzer.py.
"""

from functools import lru_cache

import textstat

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _dependency_depth(sent) -> int:
    """Max depth of the dependency tree for a spaCy sentence span."""
    def depth(token):
        children = list(token.children)
        if not children:
            return 1
        return 1 + max(depth(c) for c in children)
    return depth(sent.root)


@lru_cache(maxsize=256)
def preprocess(prompt: str) -> dict:
    """
    Runs tokenization, sentence segmentation, POS tagging, dependency
    parsing, and readability scoring on `prompt`. Returns a dict of
    objective linguistic features consumed by analyzer.py.

    Cached (lru_cache) since the same prompt text is often re-analyzed
    multiple times in one session (e.g. Streamlit reruns).
    """
    text = prompt.strip()
    if not text:
        return {
            "token_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "avg_dependency_depth": 0.0,
            "pronoun_count": 0,
            "passive_voice_sentences": 0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "sentences": [],
        }

    nlp = _get_nlp()
    doc = nlp(text)

    sentences = list(doc.sents)
    sentence_count = max(len(sentences), 1)

    words = [t for t in doc if not t.is_punct and not t.is_space]
    word_count = len(words)

    avg_sentence_length = word_count / sentence_count

    depths = [_dependency_depth(s) for s in sentences] or [0]
    avg_dependency_depth = sum(depths) / len(depths)

    pronoun_count = sum(1 for t in doc if t.pos_ == "PRON")

    # Passive voice: presence of an auxpass/nsubjpass dependency (spaCy's
    # standard passive-construction tags).
    passive_voice_sentences = sum(
        1 for s in sentences
        if any(t.dep_ in ("nsubjpass", "auxpass") for t in s)
    )

    try:
        reading_ease = textstat.flesch_reading_ease(text)
        grade_level = textstat.flesch_kincaid_grade(text)
    except Exception:
        reading_ease, grade_level = 0.0, 0.0

    return {
        "token_count": len(doc),
        "word_count": word_count,
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_dependency_depth": round(avg_dependency_depth, 2),
        "pronoun_count": pronoun_count,
        "passive_voice_sentences": passive_voice_sentences,
        "flesch_reading_ease": round(reading_ease, 2),
        "flesch_kincaid_grade": round(grade_level, 2),
        "sentences": [s.text for s in sentences],
    }
