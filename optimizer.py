"""
optimizer.py

Generates multiple Prompt-Engineering-technique versions of a user's prompt
using pure Python string templates -- no external AI model / API call is
made, so there is zero quota, rate-limit, or billing risk.

Analysis-driven design: rather than re-detecting weaknesses with its own
independent regex pass (which used to duplicate, and could silently drift
from, analyzer.py's rules), this module consumes the actual
Prompt Quality Assessment result (analyzer.analyze_prompt's hybrid
rule+DistilBERT scores) and only injects Context / Audience / Constraints /
Output Format boilerplate for the categories that assessment scored below
the "acceptable" threshold (matching the same >=7 cutoff analyzer.py uses
for its strengths/weaknesses split). This is what the paper's Design
Methodology describes: "prompts with low or moderate quality are processed
by the Prompt Optimization Module" based on the assessment's findings, not
an independent second opinion.

Each technique below is meant to *showcase a different prompting style*;
"Standard" is kept as the un-enriched baseline for comparison.
"""

import re

WEAK_THRESHOLD = 7  # matches analyzer.py's strength/weakness cutoff


def _clean_task(user_prompt: str) -> str:
    """Trim and normalize whitespace in the user's raw prompt."""
    task = user_prompt.strip()
    task = re.sub(r"\n{3,}", "\n\n", task)
    return task


def _extract_topic(task: str) -> str:
    """
    Best-effort extraction of the 'subject' of the prompt, so templates can
    refer to it naturally (e.g. 'Explain <topic> to a beginner').
    Falls back to the full task text if no clear pattern is found.
    """
    match = re.search(
        r"(?:explain|describe|summarize|teach|define)\s+(?:about\s+)?(.*?)(?:\.|$)",
        task,
        re.IGNORECASE,
    )
    if match:
        topic = match.group(1).strip()
        if topic:
            return topic
    return task


def _build_enrichment(analysis: dict) -> dict:
    """
    Returns ready-to-insert lines for whichever categories the Prompt
    Quality Assessment (analyzer.analyze_prompt) scored below
    WEAK_THRESHOLD. If a category already scored well, we leave it blank
    here to avoid duplicating what the user already specified.
    """
    enrichment = {"context": "", "audience": "", "constraints": "", "format": ""}

    if analysis.get("context", 0) < WEAK_THRESHOLD:
        enrichment["context"] = (
            "Context: This is intended as a clear, educational explanation "
            "for someone encountering the topic for the first time.\n"
        )

    if analysis.get("audience", 0) < WEAK_THRESHOLD:
        enrichment["audience"] = (
            "Audience: A general audience with no prior background in this topic.\n"
        )

    if analysis.get("constraints", 0) < WEAK_THRESHOLD:
        enrichment["constraints"] = (
            "Constraints: Keep the explanation clear and concise, ideally under 250 words.\n"
        )

    if analysis.get("output", 0) < WEAK_THRESHOLD:
        enrichment["format"] = (
            "Output Format: Use bullet points or short paragraphs, and end with a one-line summary.\n"
        )

    return enrichment


def generate_prompt_versions(user_prompt: str, analysis: dict = None) -> str:
    """
    Takes the user's original prompt and returns a single string containing
    several improved versions, each separated by a '### Title' header so the
    UI can split and display them individually.

    `analysis` should be the dict returned by analyzer.analyze_prompt(); if
    omitted, it is computed here so the function still works standalone
    (e.g. for ad-hoc scripts), but callers that already ran the assessment
    (app.py, benchmark.py) should pass it in to avoid redundant scoring.
    """
    task = _clean_task(user_prompt)
    topic = _extract_topic(task)

    if analysis is None:
        from analyzer import analyze_prompt
        analysis = analyze_prompt(user_prompt)

    enrichment = _build_enrichment(analysis)
    extras = (
        f"{enrichment['context']}"
        f"{enrichment['audience']}"
        f"{enrichment['constraints']}"
        f"{enrichment['format']}"
    ).strip()

    versions = {}

    # Kept as the un-enriched baseline so the person can see how much the
    # other versions improve on it.
    versions["Standard Prompt"] = task

    versions["Role-Based Prompt"] = (
        f"Act as an experienced subject-matter expert and educator.\n"
        f"{task}\n\n"
        f"{extras}\n\n"
        f"Draw on deep domain expertise while keeping the explanation clear "
        f"and approachable."
    ).strip()

    versions["Chain-of-Thought Prompt"] = (
        f"{task}\n\n"
        f"{extras}\n\n"
        f"Think through this step by step before giving your final answer:\n"
        f"1. Identify the core concept of {topic}.\n"
        f"2. Break it down into smaller, logical parts.\n"
        f"3. Explain each part in order, building on the previous one.\n"
        f"4. Summarize the key takeaway at the end."
    ).strip()

    versions["Few-Shot Prompt"] = (
        f"Here is an example of the style of explanation expected:\n\n"
        f"Example Q: Explain how the internet works.\n"
        f"Example A: The internet is a global network of connected computers "
        f"that share information using standard protocols, much like a "
        f"postal system routes letters between addresses.\n\n"
        f"Now, using a similar style:\n{task}\n\n"
        f"{extras}"
    ).strip()

    versions["Structured Prompt"] = (
        f"Context: {task}\n"
        f"{enrichment['audience']}"
        f"{enrichment['constraints']}\n"
        f"Instructions:\n"
        f"- Provide a clear, well-organized explanation of {topic}.\n"
        f"- Use headings or bullet points where helpful.\n"
        f"- Keep the language simple and avoid unnecessary jargon.\n"
        f"- End with a one-line summary.\n\n"
        f"{enrichment['format'] if enrichment['format'] else 'Output Format: Structured response with headings and bullet points.'}"
    ).strip()

    versions["Zero-Shot Prompt"] = (
        f"{task}\n\n"
        f"{extras}\n\n"
        f"Provide a direct, concise answer without using examples."
    ).strip()

    versions["Expert Prompt"] = (
        f"You are a world-class expert with 20+ years of experience in this field.\n"
        f"{task}\n\n"
        f"{extras}\n\n"
        f"Provide a nuanced, expert-level explanation, but keep it accessible "
        f"to someone learning the topic for the first time."
    ).strip()

    output_parts = []
    for title, content in versions.items():
        # Collapse any accidental triple-blank-lines left by empty enrichment fields
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        output_parts.append(f"### {title}\n{content}")

    return "\n\n".join(output_parts).strip()