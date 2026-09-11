"""
summarize_results.py

Aggregates results.json (benchmark.py: hybrid analyzer metrics across the
18-prompt x 8-technique matrix) and llm_eval_results.json (llm_eval.py:
real Gemini before/after response-quality comparison) into the tables
needed for the paper's Results section. Prints Markdown tables directly
to stdout and also writes summary.json with the raw numbers.

Design note: "best optimized technique" per tier is chosen ONCE, by mean
overall quality score, and every other metric (ambiguity, hallucination
risk, complexity, token usage) is then reported for that SAME technique --
not independently re-picked per metric. Independently maximizing each
metric would be incoherent (e.g. hallucination risk is "better" when
LOWER, complexity isn't better/worse in either direction, and picking a
different "winning" technique per metric would tell an inconsistent story).

Usage:
    python summarize_results.py
"""
import io
import json
import sys
from collections import defaultdict
from statistics import mean

# Force UTF-8 stdout so table punctuation renders correctly regardless of
# the terminal's default codepage (matters on Windows).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TIERS = ["Vague", "Moderate", "Well-Structured"]
TECHNIQUES = ["Original", "Standard Prompt", "Role-Based Prompt", "Chain-of-Thought Prompt",
              "Few-Shot Prompt", "Structured Prompt", "Zero-Shot Prompt", "Expert Prompt"]
OPTIMIZED_TECHNIQUES = [t for t in TECHNIQUES if t not in ("Original", "Standard Prompt")]

METRICS = ["score", "ambiguity", "hallucination_risk", "complexity", "token_usage"]


def load_results():
    return json.load(open("results.json"))["rows"]


def aggregate(rows):
    """mean[metric][(tier, technique)] -> float"""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in METRICS:
            buckets[m][(r["tier"], r["technique"])].append(r[m])
    return {m: {k: mean(v) for k, v in d.items()} for m, d in buckets.items()}


def print_score_matrix(means):
    print("\n## Table 1 -- Mean Overall Quality Score (0-100) by Tier x Technique\n")
    print("| Tier | " + " | ".join(TECHNIQUES) + " |")
    print("|" + "---|" * (len(TECHNIQUES) + 1))
    for tier in TIERS:
        cells = [f"{means['score'].get((tier, t), float('nan')):.1f}" for t in TECHNIQUES]
        print(f"| {tier} | " + " | ".join(cells) + " |")


def pick_best_technique_per_tier(means):
    best = {}
    for tier in TIERS:
        best_tech = max(OPTIMIZED_TECHNIQUES, key=lambda t: means["score"].get((tier, t), float("-inf")))
        best[tier] = best_tech
    return best


def print_consolidated_table(means, best_by_tier):
    print("\n## Table 2 -- Original vs. Single Best-Scoring Optimized Technique (per tier)\n")
    print("Best technique per tier is the one with the highest mean overall quality "
          "score; all other metrics below are reported for that SAME technique, not "
          "independently re-optimized per metric.\n")
    print("| Tier | Best Technique | Score (orig -> best) | Ambiguity 0-10 higher=clearer (orig -> best) "
          "| Hallucination Risk 0-10 lower=better (orig -> best) | Complexity 0-10 informational (orig -> best) "
          "| Token Usage (orig -> best) |")
    print("|---|---|---|---|---|---|---|")

    out = {}
    for tier in TIERS:
        tech = best_by_tier[tier]
        row = {"best_technique": tech}
        cells = []
        for m in METRICS:
            o = means[m].get((tier, "Original"), float("nan"))
            b = means[m].get((tier, tech), float("nan"))
            row[m] = {"original": round(o, 2), "optimized": round(b, 2), "delta": round(b - o, 2)}
            if m != "score":
                cells.append(f"{o:.2f} -> {b:.2f}")
        print(f"| {tier} | {tech} | {row['score']['original']:.1f} -> {row['score']['optimized']:.1f} | "
              + " | ".join(cells) + " |")
        out[tier] = row
    return out


def summarize_llm_eval():
    try:
        data = json.load(open("llm_eval_results.json"))
    except FileNotFoundError:
        print("\n(llm_eval_results.json not found -- skipping real LLM response-quality section)")
        return None

    print(f"\n## Table 3 -- Real LLM (Gemini) Response Quality: Original vs. Optimized Prompt (n={len(data)} prompts)\n")

    fields = ["relevance", "completeness", "clarity", "format_adherence", "unsupported_claims_count"]
    print("| Metric | Original response | Optimized response | Delta |")
    print("|---|---|---|---|")
    out = {"n": len(data)}
    for f in fields:
        orig_vals = [r["original_judged"][f] for r in data]
        opt_vals = [r["optimized_judged"][f] for r in data]
        om, pm = mean(orig_vals), mean(opt_vals)
        out[f] = {"original": round(om, 2), "optimized": round(pm, 2), "delta": round(pm - om, 2)}
        print(f"| {f} | {om:.2f} | {pm:.2f} | {pm-om:+.2f} |")

    orig_wc = [r["original_word_count"] for r in data]
    opt_wc = [r["optimized_word_count"] for r in data]
    print(f"| response word count | {mean(orig_wc):.0f} | {mean(opt_wc):.0f} | {mean(opt_wc)-mean(orig_wc):+.0f} |")
    out["word_count"] = {"original": round(mean(orig_wc), 1), "optimized": round(mean(opt_wc), 1)}

    constrained = [r for r in data if r.get("optimized_word_limit")]
    if constrained:
        n_within = sum(1 for r in constrained if r["optimized_within_limit"])
        print(f"\nObjective word-limit compliance (non-LLM-judged, directly counted): "
              f"optimized responses stayed within their prompt's explicit word limit in "
              f"{n_within}/{len(constrained)} cases ({100*n_within/len(constrained):.0f}%).")
        out["word_limit_compliance"] = {
            "optimized_within_limit": n_within, "optimized_total_constrained": len(constrained),
        }

    return out


def main():
    rows = load_results()
    means = aggregate(rows)

    print_score_matrix(means)
    best_by_tier = pick_best_technique_per_tier(means)
    consolidated = print_consolidated_table(means, best_by_tier)
    llm_eval = summarize_llm_eval()

    summary = {
        "best_technique_by_tier": best_by_tier,
        "consolidated_by_tier": consolidated,
        "llm_eval": llm_eval,
    }
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved aggregated numbers to summary.json")


if __name__ == "__main__":
    main()
