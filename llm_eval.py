"""
llm_eval.py

Real before/after LLM response-quality experiment, using Google Gemini
(free tier). This is the strongest evidence for the paper's core claim --
"improve response quality" -- because it measures actual LLM output, not
just static prompt-text heuristics like analyzer.py's scores.

For each of the 18 benchmark prompts (results.json, produced by
benchmark.py) this script:
  1. Picks the highest-scoring optimized version for that prompt (per the
     hybrid analyzer's "score" field).
  2. Sends BOTH the original prompt and the optimized prompt to Gemini and
     records the generated response.
  3. Sends each (original_instruction, response) pair to Gemini again as an
     independent LLM-judge call, asking it to rate the response on
     relevance, completeness, clarity, and format adherence (1-10 each),
     plus flag any specific claims in the response that are NOT grounded
     in the prompt (a proxy indicator for hallucination -- not a factual
     ground-truth check, since we have no external knowledge base here).
  4. Saves every prompt, response, and judge verdict to llm_eval_results.json.

Requires: GEMINI_API_KEY environment variable (Google AI Studio free-tier
key: https://aistudio.google.com/apikey).

Usage:
    export GEMINI_API_KEY=...   (or set it in the shell before running)
    python llm_eval.py [--limit N] [--model gemini-2.0-flash]

Rate limiting: free-tier Gemini quotas are modest (commonly ~15 requests/
minute). This script sleeps between calls and retries with backoff on 429s
so a full 18-prompt run (~72 API calls) stays within typical free-tier
limits; expect it to take several minutes.
"""
import argparse
import json
import os
import re
import sys
import time

from google import genai
from google.genai import errors as genai_errors

JUDGE_PROMPT_TEMPLATE = """You are a strict, skeptical evaluator of AI assistant responses. Do not
default to high scores -- most responses have at least minor issues.
Reserve 9-10 for responses with zero flaws in that dimension.

The instruction the response was generated from:
---
{instruction}
---

AI response to evaluate:
---
{response}
---

Rate the response on these criteria, each as an integer 1-10:
- relevance: does it address exactly what the instruction asked for, no more and no less?
- completeness: does it cover the topic adequately given any stated constraints?
- clarity: is it well-organized and easy to understand?
- format_adherence: if the instruction explicitly requested a format, length limit, tone,
  audience level, or structure (e.g. "under 250 words", "bullet points", "for a beginner"),
  score strictly on whether the response actually complies -- a response that ignores an
  explicit length/format/audience requirement must score 3 or below on this dimension,
  regardless of how good the content otherwise is. If the instruction requested nothing
  specific, score on general readability.

Also identify: unsupported_claims_count -- the number of specific, concrete
factual claims (numbers, dates, named entities, statistics) in the response
that go beyond what could be reasonably inferred from general knowledge of
the topic and are stated with unwarranted confidence (a proxy for
hallucination risk, since you cannot verify facts against a live source).
Look actively for these; do not default to 0.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"relevance": <int>, "completeness": <int>, "clarity": <int>, "format_adherence": <int>, "unsupported_claims_count": <int>, "justification": "<one sentence>"}}
"""

WORD_LIMIT_PATTERNS = [
    r"under (\d+) words",
    r"no more than (\d+) words",
    r"within (\d+) words",
    r"word limit[: ]+(\d+)",
    r"limit(?:ed)? to (\d+) words",
]


def extract_word_limit(prompt_text: str):
    """Best-effort extraction of an explicit word-count constraint, so
    adherence can be checked objectively (by counting) rather than relying
    on the LLM judge's self-reported opinion."""
    for pattern in WORD_LIMIT_PATTERNS:
        m = re.search(pattern, prompt_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


class DailyQuotaExhausted(RuntimeError):
    """Raised when the free-tier PER-DAY quota is hit -- backoff/retry
    cannot fix this within the same day, so we fail fast instead of
    burning minutes on a hopeless retry loop."""


def _call_with_retry(fn, *, max_retries=6, base_delay=15):
    for attempt in range(max_retries):
        try:
            return fn()
        except genai_errors.APIError as e:
            msg = str(e)
            status = getattr(e, "code", None)
            if status == 429 or "RESOURCE_EXHAUSTED" in msg:
                if "PerDay" in msg:
                    raise DailyQuotaExhausted(
                        "Free-tier daily request quota for this model is exhausted. "
                        "Either wait for the daily reset or pass --model with a "
                        "different model that still has quota left. Raw error: " + msg[:300]
                    )
                delay = base_delay * (attempt + 1)
                print(f"  rate-limited, retrying in {delay}s...", file=sys.stderr)
                time.sleep(delay)
                continue
            if status == 503 or "UNAVAILABLE" in msg:
                # Transient server-side overload, not a quota issue -- a
                # short retry is usually enough.
                delay = 10 * (attempt + 1)
                print(f"  model overloaded (503), retrying in {delay}s...", file=sys.stderr)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Exceeded max retries calling Gemini API")


def generate_response(client, model, prompt_text):
    def _do():
        r = client.models.generate_content(model=model, contents=prompt_text)
        return r.text or ""
    return _call_with_retry(_do)


def judge_response(client, model, instruction, response_text):
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(instruction=instruction, response=response_text)

    def _do():
        r = client.models.generate_content(model=model, contents=judge_prompt)
        return r.text or ""

    raw = _call_with_retry(_do)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {raw[:200]}")
    return json.loads(match.group(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N prompts (for a quick smoke test).")
    parser.add_argument("--model", default="gemini-flash-latest", help="Gemini model to use for both generation and judging.")
    parser.add_argument("--delay", type=float, default=4.5, help="Seconds to sleep between API calls (free-tier rate limiting).")
    parser.add_argument("--in-file", default="results.json")
    parser.add_argument("--out-file", default="llm_eval_results.json")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set the GEMINI_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    data = json.load(open(args.in_file))
    rows = data["rows"]

    # Group by prompt_id, find the Original text and the best-scoring
    # optimized technique for each.
    by_prompt = {}
    original_text_by_id = {}
    for r in rows:
        pid = r["prompt_id"]
        by_prompt.setdefault(pid, []).append(r)

    # Need the actual prompt text too -- re-derive from benchmark.py's prompt
    # dict. benchmark.py guards its script body behind __main__, so this
    # import is side-effect-free (does not re-run the benchmark).
    from benchmark import prompts as prompt_bank
    id_to_text = {}
    for tier, plist in prompt_bank.items():
        for i, p in enumerate(plist, 1):
            id_to_text[f"{tier}-{i}"] = p

    from optimizer import generate_prompt_versions
    from analyzer import analyze_prompt

    prompt_ids = list(id_to_text.keys())
    if args.limit:
        prompt_ids = prompt_ids[: args.limit]

    # Resume support: skip prompt_ids already present in an existing
    # out-file, so a rate-limit failure partway through doesn't burn quota
    # re-doing prompts that already succeeded.
    results = []
    already_done = set()
    if os.path.exists(args.out_file):
        try:
            results = json.load(open(args.out_file))
            already_done = {r["prompt_id"] for r in results}
            if already_done:
                print(f"Resuming: {len(already_done)} prompt(s) already in {args.out_file}, skipping those.")
        except (json.JSONDecodeError, KeyError):
            results = []

    for idx, pid in enumerate(prompt_ids, 1):
        if pid in already_done:
            continue
        original_text = id_to_text[pid]

        best_row = max(
            (r for r in by_prompt[pid] if r["technique"] != "Original"),
            key=lambda r: r["score"],
        )
        best_technique = best_row["technique"]

        analysis = analyze_prompt(original_text)
        versions_raw = generate_prompt_versions(original_text, analysis)
        versions = {}
        for section in versions_raw.split("###"):
            if not section.strip():
                continue
            lines = section.strip().split("\n")
            versions[lines[0].strip()] = "\n".join(lines[1:]).strip()
        optimized_text = versions[best_technique]

        print(f"[{idx}/{len(prompt_ids)}] {pid}: generating responses (best technique = {best_technique})...")

        original_response = generate_response(client, args.model, original_text)
        time.sleep(args.delay)
        optimized_response = generate_response(client, args.model, optimized_text)
        time.sleep(args.delay)

        # Judge each response against the instruction that actually
        # produced it -- this is what lets format/length/audience
        # requirements added only by the optimizer (e.g. "under 250
        # words", "bullet points") actually get checked, instead of being
        # invisible to a judge that only ever sees the bare original text.
        original_judged = judge_response(client, args.model, original_text, original_response)
        time.sleep(args.delay)
        optimized_judged = judge_response(client, args.model, optimized_text, optimized_response)
        time.sleep(args.delay)

        # Objective (non-LLM) word-count-limit check, when the prompt
        # states one explicitly -- a hard, verifiable signal alongside the
        # judge's subjective scores.
        original_limit = extract_word_limit(original_text)
        optimized_limit = extract_word_limit(optimized_text)
        original_word_count = len(original_response.split())
        optimized_word_count = len(optimized_response.split())

        results.append({
            "prompt_id": pid,
            "original_prompt": original_text,
            "optimized_prompt": optimized_text,
            "optimized_technique": best_technique,
            "original_response": original_response,
            "optimized_response": optimized_response,
            "original_word_count": original_word_count,
            "optimized_word_count": optimized_word_count,
            "original_word_limit": original_limit,
            "optimized_word_limit": optimized_limit,
            "original_within_limit": (original_word_count <= original_limit) if original_limit else None,
            "optimized_within_limit": (optimized_word_count <= optimized_limit) if optimized_limit else None,
            "original_judged": original_judged,
            "optimized_judged": optimized_judged,
        })

        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} evaluated prompt pairs to {args.out_file}")


if __name__ == "__main__":
    main()
