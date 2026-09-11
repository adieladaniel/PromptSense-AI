"""
benchmark.py

Runs the 18-prompt benchmark (3 tiers x 6 prompts) through the full
PromptSense AI pipeline -- hybrid rule+DistilBERT assessment (all 8 core
metrics plus the 3 additional insight metrics) and the analysis-driven
optimizer -- and saves full results to results.json.

Usage:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python benchmark.py

Then use results.json to compute the Results-section tables/averages for
the paper (see summarize_results.py for a ready-made aggregation).
"""
import time, json
from analyzer import analyze_prompt
from optimizer import generate_prompt_versions

prompts = {
    "Vague": [
        "Explain about AR VR",
        "Write something about climate change",
        "Tell me stuff about the stock market",
        "Describe machine learning",
        "Give info on nutrition",
        "Talk about ancient Rome",
    ],
    "Moderate": [
        "Explain how neural networks work for a student audience.",
        "Summarize the causes of World War 1 in bullet points.",
        "Compare cloud computing and edge computing, keep it concise.",
        "Describe the water cycle for a beginner, under 200 words.",
        "Explain blockchain technology using simple language.",
        "Outline the steps of the scientific method with examples.",
    ],
    "Well-Structured": [
        "Act as an experienced data scientist. Context: I am a beginner learning about regression models. Audience: someone with no statistics background. Explain linear regression step by step. Constraints: keep it under 300 words. Output Format: use bullet points and end with a summary.",
        "You are a professional nutritionist. Context: I am designing a meal plan for a diabetic patient. Audience: healthcare students. Explain how the glycemic index works, step by step, with a table. Limit the response to 250 words.",
        "Act as a senior software engineer. Context: I am mentoring a junior developer. Audience: beginner programmers. Explain the concept of recursion, using a numbered list. Keep it under 200 words.",
        "You are an experienced historian. Context: I am preparing a lecture. Audience: undergraduate students. Describe the causes of the French Revolution in a structured format with headings. Word limit: 300 words.",
        "Act as a financial advisor. Context: I am a first-time investor. Audience: complete beginners. Explain what a mutual fund is, step by step, using bullet points. Keep it concise, under 250 words.",
        "You are a UX design expert. Context: I am onboarding a new designer. Audience: junior designers. Explain the principles of accessible design using a structured summary. Limit to 300 words.",
    ],
}

TECHNIQUES = ["Standard Prompt", "Role-Based Prompt", "Chain-of-Thought Prompt",
              "Few-Shot Prompt", "Structured Prompt", "Zero-Shot Prompt", "Expert Prompt"]

CORE_METRICS = ["score", "clarity", "context", "role", "audience", "constraints",
                "output", "specificity", "ambiguity"]
INSIGHT_METRICS = ["token_usage", "complexity", "hallucination_risk"]


def parse_versions(raw):
    out = {}
    for section in raw.split("###"):
        if not section.strip():
            continue
        lines = section.strip().split("\n")
        out[lines[0].strip()] = "\n".join(lines[1:]).strip()
    return out


def run_benchmark():
    rows, latency_rows = [], []

    for tier, plist in prompts.items():
        for i, p in enumerate(plist, 1):
            t0 = time.perf_counter()
            base = analyze_prompt(p)
            base_latency_ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "tier": tier, "prompt_id": f"{tier}-{i}", "technique": "Original",
                **{k: base[k] for k in CORE_METRICS},
                **{k: base[k] for k in INSIGHT_METRICS},
                "method": base["method"],
            })
            latency_rows.append({"tier": tier, "prompt_id": f"{tier}-{i}", "technique": "Original", "analyze_latency_ms": base_latency_ms})

            # Analysis-driven: pass the already-computed assessment into the
            # optimizer so enrichment reflects the real weaknesses found above,
            # not an independent re-detection pass.
            raw = generate_prompt_versions(p, base)
            versions = parse_versions(raw)
            for tech in TECHNIQUES:
                content = versions.get(tech, "")
                t0 = time.perf_counter()
                res = analyze_prompt(content)
                lat_ms = (time.perf_counter() - t0) * 1000
                rows.append({
                    "tier": tier, "prompt_id": f"{tier}-{i}", "technique": tech,
                    **{k: res[k] for k in CORE_METRICS},
                    **{k: res[k] for k in INSIGHT_METRICS},
                    "method": res["method"],
                })
                latency_rows.append({"tier": tier, "prompt_id": f"{tier}-{i}", "technique": tech, "analyze_latency_ms": lat_ms})
            print(f"done: {tier}-{i}")

    return rows, latency_rows


if __name__ == "__main__":
    rows, latency_rows = run_benchmark()

    with open("results.json", "w") as f:
        json.dump({"rows": rows, "latency_rows": latency_rows}, f, indent=2)

    print(f"\nSaved {len(rows)} rows to results.json")
    print(f"Scoring method used: {rows[0]['method']}")
