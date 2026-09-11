import re
import time
import json

import streamlit as st
import streamlit.components.v1 as components

from analyzer import analyze_prompt
from optimizer import generate_prompt_versions


def copy_button(content, key):
    """Renders a small button that copies `content` to the clipboard."""
    safe_content = json.dumps(content)  # safely escapes quotes/newlines for JS
    components.html(
        f"""
        <button id="copy-btn-{key}"
            style="
                background:linear-gradient(90deg,#2563eb,#06b6d4);
                color:white;
                border:none;
                border-radius:10px;
                padding:8px 16px;
                font-size:14px;
                font-weight:bold;
                cursor:pointer;
                transition:.2s ease;
            "
            onmouseover="this.style.transform='translateY(-1px)'"
            onmouseout="this.style.transform='translateY(0)'"
        >📋 Copy</button>
        <script>
            (function() {{
                const btn = document.getElementById("copy-btn-{key}");
                const textToCopy = {safe_content};
                btn.addEventListener("click", function() {{
                    navigator.clipboard.writeText(textToCopy);
                    btn.innerText = "✅ Copied!";
                    setTimeout(function() {{ btn.innerText = "📋 Copy"; }}, 1500);
                }});
            }})();
        </script>
        """,
        height=45,
    )


# --------------------------------------------------
# HELPERS -- small HTML component renderers used to give the results
# section a richer, hand-designed look than default Streamlit widgets.
# --------------------------------------------------

def _metric_color(value, max_value=10):
    pct = (value / max_value) if max_value else 0
    if pct < 0.4:
        return "#ef4444"
    if pct < 0.7:
        return "#facc15"
    return "#22c55e"


def _score_status(score):
    if score < 40:
        return "#ef4444", "rgba(239,68,68,.15)", "🔴 Needs Improvement"
    if score < 70:
        return "#facc15", "rgba(250,204,21,.15)", "🟡 Average Prompt"
    return "#22c55e", "rgba(34,197,94,.15)", "🟢 Excellent Prompt"


def render_gauge(score):
    color, bg, status = _score_status(score)
    deg = max(0, min(360, round(score / 100 * 360, 1)))
    st.markdown(
        f"""
        <div class="gauge-card">
            <div class="gauge" style="background:conic-gradient({color} {deg}deg, rgba(255,255,255,.08) {deg}deg 360deg);">
                <div class="gauge-inner">
                    <div class="gauge-score" style="color:{color}">{score}</div>
                    <div class="gauge-max">out of 100</div>
                </div>
            </div>
            <div class="status-pill" style="color:{color};background:{bg};border:1px solid {color}44;">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_chips(token_usage, complexity, hallucination_risk):
    risk_color = "#ef4444" if hallucination_risk >= 7 else ("#facc15" if hallucination_risk >= 4 else "#22c55e")
    risk_label = "High" if hallucination_risk >= 7 else ("Medium" if hallucination_risk >= 4 else "Low")
    st.markdown(
        f"""
        <div class="chip-row">
            <div class="chip">
                <div class="chip-icon">🔢</div>
                <div class="chip-label">Token Usage</div>
                <div class="chip-value">{token_usage}</div>
                <div class="chip-sub">tokens (cl100k_base)</div>
            </div>
            <div class="chip">
                <div class="chip-icon">🧩</div>
                <div class="chip-label">Prompt Complexity</div>
                <div class="chip-value">{complexity}<span style="font-size:14px;color:#64748b;">/10</span></div>
                <div class="chip-sub">informational, not a quality score</div>
            </div>
            <div class="chip">
                <div class="chip-icon">⚠️</div>
                <div class="chip-label">Hallucination Risk</div>
                <div class="chip-value" style="color:{risk_color}">{hallucination_risk}<span style="font-size:14px;color:#64748b;">/10</span></div>
                <div class="chip-sub" style="color:{risk_color}">{risk_label} risk</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics):
    """metrics: list of (icon, label, value)."""
    cards = ""
    for icon, label, value in metrics:
        color = _metric_color(value)
        pct = max(0, min(100, value / 10 * 100))
        cards += f"""
            <div class="metric-card">
                <div class="metric-top">
                    <span class="metric-icon">{icon}</span>
                    <span class="metric-name">{label}</span>
                    <span class="metric-value" style="color:{color}">{value}/10</span>
                </div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
            </div>
        """
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def render_list_card(items, kind):
    """kind: 'strength' | 'weakness' | 'suggestion'."""
    icon_map = {"strength": "✅", "weakness": "❌", "suggestion": "💡"}
    class_map = {"strength": "list-strength", "weakness": "list-weak", "suggestion": "list-suggest"}
    empty_map = {
        "strength": "No standout strengths detected yet.",
        "weakness": "No weaknesses found — nicely done!",
        "suggestion": "Your prompt already follows good prompt engineering practices!",
    }

    if not items:
        st.markdown(f'<div class="empty-note">🎉 {empty_map[kind]}</div>', unsafe_allow_html=True)
        return

    rows = ""
    for raw in items:
        clean = re.sub(r"^[^\w]+", "", raw).strip()
        rows += f'<div class="list-item"><span class="li-icon">{icon_map[kind]}</span><span>{clean}</span></div>'

    st.markdown(f'<div class="list-card {class_map[kind]}">{rows}</div>', unsafe_allow_html=True)


TECHNIQUE_META = {
    "Standard Prompt": ("📝", "Your original prompt, unchanged — a baseline to compare the rest against."),
    "Role-Based Prompt": ("🎭", "Assigns the AI a specific persona and expertise before the task."),
    "Chain-of-Thought Prompt": ("🔗", "Asks the AI to reason step by step before giving a final answer."),
    "Few-Shot Prompt": ("📚", "Shows the AI a worked example first, then asks for a similar style."),
    "Structured Prompt": ("🧱", "Organizes context, instructions, and output format into clear sections."),
    "Zero-Shot Prompt": ("⚡", "A direct instruction with no examples — concise and to the point."),
    "Expert Prompt": ("🎓", "Frames the AI as a world-class expert for a nuanced, authoritative answer."),
}

EXAMPLE_PROMPTS = {
    "😕 Vague example": "Explain about AR VR",
    "🙂 Moderate example": "Summarize the causes of World War 1 in bullet points.",
    "🤩 Well-structured example": (
        "Act as an experienced data scientist. Context: I am a beginner learning about "
        "regression models. Audience: someone with no statistics background. Explain "
        "linear regression step by step. Constraints: keep it under 300 words. "
        "Output Format: use bullet points and end with a summary."
    ),
}

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PromptSense AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
}

/* Animated background */
.stApp{
    background: linear-gradient(-45deg,#0f172a,#1e293b,#111827,#0b1220);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}
@keyframes gradientShift{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}
@keyframes fadeIn{
    from{opacity:0;transform:translateY(10px);}
    to{opacity:1;transform:translateY(0);}
}

/* Hide Streamlit chrome */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* Hero header */
.hero{ text-align:center; padding:14px 0 4px 0; }
.hero-badge{
    display:inline-block;
    padding:6px 16px;
    border-radius:999px;
    background:rgba(56,189,248,.12);
    border:1px solid rgba(56,189,248,.35);
    color:#7dd3fc;
    font-size:13px;
    font-weight:600;
    letter-spacing:.3px;
    margin-bottom:16px;
}
.title{
    font-family:'Poppins',sans-serif;
    font-size:52px;
    font-weight:800;
    background:linear-gradient(90deg,#60a5fa,#22d3ee,#a78bfa);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    margin:0;
    line-height:1.15;
}
.subtitle{
    color:#94a3b8;
    font-size:17px;
    margin-top:8px;
}
.chips-row{
    display:flex;
    justify-content:center;
    gap:10px;
    flex-wrap:wrap;
    margin-top:18px;
}
.tech-chip{
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.12);
    color:#cbd5e1;
    padding:6px 14px;
    border-radius:999px;
    font-size:13px;
}

/* Section titles */
.section-title{
    font-family:'Poppins',sans-serif;
    font-weight:700;
    font-size:21px;
    color:#e2e8f0;
    margin:28px 0 14px 0;
}

/* Score gauge */
.gauge-card{
    text-align:center;
    padding:32px 20px;
    border-radius:22px;
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.1);
    animation:fadeIn .6s ease both;
}
.gauge{
    width:200px;height:200px;border-radius:50%;
    margin:0 auto 18px auto;
    display:flex;align-items:center;justify-content:center;
    transition:background 1s ease;
}
.gauge-inner{
    width:156px;height:156px;border-radius:50%;
    background:#0b1220;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    box-shadow: inset 0 0 20px rgba(0,0,0,.45);
}
.gauge-score{
    font-family:'Poppins',sans-serif;
    font-size:46px;
    font-weight:800;
    line-height:1;
}
.gauge-max{ color:#64748b; font-size:13px; margin-top:4px; }
.status-pill{
    display:inline-block;
    margin-top:16px;
    padding:8px 22px;
    border-radius:999px;
    font-weight:700;
    font-size:15px;
}

/* Insight chips */
.chip-row{ display:flex; gap:14px; flex-wrap:wrap; }
.chip{
    flex:1;
    min-width:170px;
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.1);
    border-radius:16px;
    padding:16px 18px;
    text-align:center;
    transition:transform .25s ease, border-color .25s ease;
    animation:fadeIn .5s ease both;
}
.chip:hover{ transform:translateY(-4px); border-color:rgba(56,189,248,.4); }
.chip-icon{ font-size:22px; margin-bottom:6px; }
.chip-label{ color:#94a3b8; font-size:13px; margin-bottom:4px; }
.chip-value{ font-family:'Poppins',sans-serif; font-size:22px; font-weight:700; color:#e2e8f0;}
.chip-sub{ color:#64748b; font-size:12px; margin-top:3px; }

/* Metric cards */
.metric-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }
.metric-card{
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.1);
    border-radius:16px;
    padding:16px 18px;
    transition:transform .25s ease, border-color .25s ease;
    animation:fadeIn .5s ease both;
}
.metric-card:hover{ transform:translateY(-3px); border-color:rgba(255,255,255,.28); }
.metric-top{ display:flex; align-items:center; gap:8px; margin-bottom:10px; }
.metric-icon{ font-size:19px; }
.metric-name{ color:#cbd5e1; font-size:14.5px; font-weight:600; flex:1; }
.metric-value{ font-family:'Poppins',sans-serif; font-weight:700; font-size:14.5px; }
.bar-track{ height:8px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; }
.bar-fill{ height:100%; border-radius:999px; transition:width 1s ease; }

/* Strengths / weaknesses / suggestions lists */
.list-card{ display:flex; flex-direction:column; gap:10px; }
.list-item{
    display:flex;
    gap:10px;
    align-items:flex-start;
    background:rgba(255,255,255,.05);
    border-left:4px solid var(--accent,#38bdf8);
    border-radius:10px;
    padding:12px 14px;
    font-size:14.5px;
    color:#e2e8f0;
    animation:fadeIn .5s ease both;
}
.list-item .li-icon{ font-size:15px; }
.list-strength .list-item{ --accent:#22c55e; }
.list-weak .list-item{ --accent:#ef4444; }
.list-suggest .list-item{ --accent:#facc15; }
.empty-note{ color:#94a3b8; font-style:italic; padding:10px 4px; }

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:14px;
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
    font-size:17px;
    font-weight:700;
    border:none;
    padding:0.6em 1em;
    transition:.25s ease;
    box-shadow:0 4px 14px rgba(37,99,235,.25);
}
.stButton>button:hover{
    transform:translateY(-2px) scale(1.01);
    box-shadow:0 8px 22px rgba(6,182,212,.35);
}

/* Text area */
textarea{
    border-radius:15px !important;
    background:#1e293b !important;
    color:white !important;
    font-size:16px !important;
    border:1px solid rgba(255,255,255,.1) !important;
}
textarea:focus{
    border-color:#38bdf8 !important;
    box-shadow:0 0 0 3px rgba(56,189,248,.2) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{
    font-size:16px;
    font-weight:600;
    padding:10px 18px;
    border-radius:10px 10px 0 0;
}
.stTabs [aria-selected="true"]{
    background:rgba(56,189,248,.12) !important;
    color:#7dd3fc !important;
}

/* Expanders (optimized prompt versions) */
[data-testid="stExpander"]{
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.1);
    border-radius:14px;
    margin-bottom:10px;
}

/* Sidebar */
section[data-testid="stSidebar"]{ background:#020617; }
.side-pill{
    display:flex;
    align-items:center;
    gap:8px;
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.08);
    border-radius:10px;
    padding:9px 12px;
    margin-bottom:8px;
    font-size:14px;
    color:#cbd5e1;
}

::-webkit-scrollbar{ width:10px; }
::-webkit-scrollbar-thumb{ background:#334155; border-radius:10px; }

</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🤖 PromptSense AI")
    st.markdown("---")
    st.success("AI Prompt Engineering Toolkit")

    st.markdown("### 📌 Features")
    for icon, label in [
        ("✅", "Prompt Analysis"),
        ("✨", "Prompt Optimization"),
        ("📊", "Prompt Quality Score"),
        ("💡", "Suggestions"),
        ("📄", "Export Prompt"),
    ]:
        st.markdown(f'<div class="side-pill">{icon} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        """
### About

PromptSense AI analyzes prompts using a hybrid approach: fast rule-based
pattern matching combined with DeBERTa-v3 zero-shot semantic scoring (falls
back to rule-based only if the model can't be loaded).

It evaluates:

- Clarity
- Context
- Constraints
- Audience
- Role
- Output Format
- Specificity
- Ambiguity

and generates multiple improved prompt versions using proven Prompt Engineering techniques (Role Prompting, Chain-of-Thought, Few-Shot, Structured, Zero-Shot, Expert) — generated locally with zero external API calls. Analysis itself uses a local transformer model (no external API calls either, once downloaded).
"""
    )

    st.markdown("---")
    st.caption("Version 1.1")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">⚡ Rule-Based + 🧠 Transformer Hybrid Engine</div>
        <div class="title">🤖 PromptSense AI</div>
        <div class="subtitle">Intelligent Prompt Quality Assessment &amp; Optimization</div>
        <div class="chips-row">
            <div class="tech-chip">🎯 8 Quality Dimensions</div>
            <div class="tech-chip">✨ 7 Prompt Engineering Styles</div>
            <div class="tech-chip">⚠️ Hallucination Risk Estimate</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2 = st.tabs([
    "📊 Prompt Analyzer",
    "✨ Optimized Prompt",
])

# --------------------------------------------------
# TAB 1
# --------------------------------------------------

with tab1:

    st.markdown('<div class="section-title">📝 Enter Your Prompt</div>', unsafe_allow_html=True)

    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = ""

    st.caption("Not sure what to try? Load one of your benchmark tiers:")
    ex_cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, (label, text) in zip(ex_cols, EXAMPLE_PROMPTS.items()):
        if col.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state["prompt_input"] = text
            st.rerun()

    prompt = st.text_area(
        "",
        height=240,
        key="prompt_input",
        placeholder="""Example:

You are an AI Professor.
Explain Artificial Intelligence to a beginner.
Use bullet points.
Limit the response to 300 words.""",
    )

    st.write("")

    analyze = st.button("🚀 Analyze Prompt")

    if analyze:

        if prompt.strip() == "":
            st.warning("Please enter a prompt.")
            st.stop()

        with st.spinner("Analyzing Prompt..."):
            result = analyze_prompt(prompt)

        method_label = (
            "🧠 Hybrid scoring (rule-based + DeBERTa-v3 zero-shot semantic analysis)"
            if result.get("method") == "hybrid"
            else "⚙️ Rule-based scoring only (semantic backend unavailable — check internet connection / `pip install transformers torch`)"
        )
        st.caption(method_label)

        score = result["score"]
        strengths = result["strengths"]
        weaknesses = result["weaknesses"]
        suggestions = result["suggestions"]

        # --------------------------------------------------
        # SCORE SECTION
        # --------------------------------------------------

        st.markdown('<div class="section-title">🎯 Overall Prompt Score</div>', unsafe_allow_html=True)
        render_gauge(score)

        st.write("")

        # --------------------------------------------------
        # ADDITIONAL INSIGHTS
        # --------------------------------------------------

        st.markdown('<div class="section-title">🔍 Additional Prompt Insights</div>', unsafe_allow_html=True)
        render_insight_chips(result["token_usage"], result["complexity"], result["hallucination_risk"])
        st.caption(
            "Hallucination risk is a heuristic estimate derived from ambiguity, "
            "specificity, context, and constraints signals — not a measured rate "
            "from actual LLM outputs."
        )

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        st.markdown('<div class="section-title">📊 Prompt Quality Metrics</div>', unsafe_allow_html=True)
        render_metric_grid([
            ("🎯", "Clarity", result["clarity"]),
            ("🔎", "Specificity", result["specificity"]),
            ("📖", "Context", result["context"]),
            ("🤖", "Role", result["role"]),
            ("⚙️", "Constraints", result["constraints"]),
            ("👤", "Audience", result["audience"]),
            ("📄", "Output Format", result["output"]),
            ("🌀", "Ambiguity", result["ambiguity"]),
        ])

        st.write("")

        # --------------------------------------------------
        # STRENGTHS / WEAKNESSES
        # --------------------------------------------------

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-title">✅ Strengths</div>', unsafe_allow_html=True)
            render_list_card(strengths, "strength")

        with col_right:
            st.markdown('<div class="section-title">❌ Weaknesses</div>', unsafe_allow_html=True)
            render_list_card(weaknesses, "weakness")

        # --------------------------------------------------
        # SUGGESTIONS
        # --------------------------------------------------

        st.markdown('<div class="section-title">💡 Suggestions</div>', unsafe_allow_html=True)
        render_list_card(suggestions, "suggestion")

        st.write("")

        if score >= 80:
            st.balloons()

        # Save for Tab 2
        st.session_state["prompt"] = prompt
        st.session_state["analysis"] = result

# --------------------------------------------------
# TAB 2 - OPTIMIZED PROMPT VERSIONS
# --------------------------------------------------

with tab2:

    st.markdown('<div class="section-title">🚀 AI Recommended Prompt Versions</div>', unsafe_allow_html=True)

    if "prompt" not in st.session_state:

        st.info("Analyze a prompt first to generate optimized versions.")

    else:

        current_prompt = st.session_state["prompt"]

        # Only regenerate if this prompt hasn't been processed yet. Even
        # though generation is now local (no API calls), Streamlit re-runs
        # the whole script on every interaction (opening an expander,
        # clicking a button, etc.), so caching avoids redundant recomputation.
        if st.session_state.get("cached_prompt") != current_prompt:

            with st.spinner("✨ Generating optimized prompt versions..."):

                try:

                    st.session_state["prompt_versions"] = generate_prompt_versions(
                        current_prompt, st.session_state.get("analysis")
                    )

                    st.session_state["cached_prompt"] = current_prompt

                except Exception as e:

                    st.session_state["prompt_versions"] = None
                    st.session_state["cached_prompt"] = None
                    st.error(f"Something went wrong while generating prompt versions: {e}")

        prompt_versions = st.session_state.get("prompt_versions")

        if prompt_versions:

            st.success("Optimization Completed Successfully!")

            # Split the generated text into sections using "###" as the delimiter
            sections = prompt_versions.split("###")

            for section in sections:

                if section.strip() == "":
                    continue

                lines = section.strip().split("\n")
                title = lines[0].strip()
                content = "\n".join(lines[1:]).strip()

                icon, description = TECHNIQUE_META.get(title, ("⭐", ""))

                with st.expander(f"{icon} {title}"):
                    if description:
                        st.caption(description)
                    st.code(content)
                    copy_button(content, key=title.replace(" ", "_"))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;
padding:20px;
font-size:15px;
color:gray;'>

🤖 <b>PromptSense AI</b>
<br>
Prompt Engineering • NLP • Rule-Based Templates
<br><br>
Made using ❤️ with Streamlit

</div>

""",
    unsafe_allow_html=True,
)
