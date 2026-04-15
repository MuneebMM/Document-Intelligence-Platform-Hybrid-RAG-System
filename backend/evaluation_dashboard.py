"""
Streamlit evaluation dashboard for the Document Intelligence Platform.

Displays RAGAS metric results, triggers evaluation runs via the FastAPI
backend, and visualises per-question pass/fail breakdowns.

Run with:
    streamlit run backend/evaluation_dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="🧪",
    layout="wide",
)

_API_BASE = "http://localhost:8000/api/v1/evaluation"
_RESULTS_DIR = Path(__file__).parent / "evaluation_results"
_PASS_THRESHOLD = 0.7
_GOOD_THRESHOLD = 0.8

_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
_METRIC_LABELS = {
    "faithfulness": "🎯 Faithfulness",
    "answer_relevancy": "📊 Answer Relevancy",
    "context_recall": "🔍 Context Recall",
    "context_precision": "⚡ Context Precision",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_latest_results() -> dict | None:
    """Load the most recent evaluation results.

    Tries the API first (works when backend is in Docker), then falls
    back to reading local files on disk.

    Returns:
        Parsed result dict, or None if no result files exist.
    """
    # --- Try API first (reliable when backend runs in Docker) ---------------
    try:
        resp = requests.get(f"{_API_BASE}/results", timeout=5)
        resp.raise_for_status()
        result_list = resp.json().get("results", [])
        if result_list:
            latest = result_list[-1]["filename"]
            detail = requests.get(f"{_API_BASE}/results/{latest}", timeout=5)
            detail.raise_for_status()
            return detail.json()
    except Exception:
        pass  # Fall back to local files

    # --- Fallback: read from local disk -------------------------------------
    if not _RESULTS_DIR.exists():
        return None
    files = sorted(_RESULTS_DIR.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as fh:
        return json.load(fh)


def _score_color(score: float) -> str:
    """Return a CSS colour string based on the score relative to thresholds.

    Args:
        score: Metric score in [0.0, 1.0].

    Returns:
        Hex colour string: green, yellow, or red.
    """
    if score >= _GOOD_THRESHOLD:
        return "#22c55e"   # green-500
    if score >= _PASS_THRESHOLD:
        return "#eab308"   # yellow-500
    return "#ef4444"       # red-500


def _fmt(score: float) -> str:
    """Format a score as a percentage string.

    Args:
        score: Metric score in [0.0, 1.0].

    Returns:
        String like '83.4%'.
    """
    return f"{score * 100:.1f}%"


# ---------------------------------------------------------------------------
# 1. HEADER
# ---------------------------------------------------------------------------

st.title("🧪 Document Intelligence Platform")
st.markdown("#### RAG Evaluation Dashboard — RAGAS Metrics")
st.divider()

# ---------------------------------------------------------------------------
# 2. RUN EVALUATION SECTION
# ---------------------------------------------------------------------------

st.subheader("Run Evaluation")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("▶ Run Full Evaluation (15 test cases)", type="primary")
with col_info:
    st.caption(
        "Calls the FastAPI backend to evaluate all 15 GDPR test cases "
        "using RAGAS metrics. Expect **5–10 minutes** to complete."
    )

if run_clicked:
    with st.spinner("Running RAGAS evaluation... (~5-10 minutes)"):
        try:
            response = requests.post(
                f"{_API_BASE}/run",
                json={"collection_name": "documents", "save_results": True},
                timeout=700,
            )
            response.raise_for_status()
            data = response.json()
            overall = data.get("summary", {}).get("overall_score", 0.0)
            st.success(
                f"Evaluation complete! Overall score: **{_fmt(overall)}** "
                f"— Pass rate: **{data['summary']['pass_rate'] * 100:.0f}%** "
                f"({data['summary']['passed_cases']}/{data['summary']['total_cases']} cases)"
            )
            st.rerun()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The evaluation may still be running — refresh in a minute.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend at http://localhost:8000. Is it running?")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Load results (used by all sections below)
# ---------------------------------------------------------------------------

results = _load_latest_results()

if results is None:
    st.info(
        "No evaluation results found. Upload a GDPR document via the main UI "
        "then click **▶ Run Full Evaluation** above."
    )
    st.stop()

summary = results.get("summary", {})
individual = results.get("individual_results", [])
failed = results.get("failed_cases", [])

# ---------------------------------------------------------------------------
# 3. METRICS OVERVIEW — 4 cards
# ---------------------------------------------------------------------------

st.subheader("Metrics Overview")
st.caption(
    f"Last run: {summary.get('evaluation_timestamp', 'unknown')}  •  "
    f"Overall score: **{_fmt(summary.get('overall_score', 0.0))}**  •  "
    f"Pass rate: **{summary.get('pass_rate', 0.0) * 100:.0f}%** "
    f"({summary.get('passed_cases', 0)}/{summary.get('total_cases', 0)} cases)"
)

card_cols = st.columns(4)
metric_keys = [
    ("avg_faithfulness", "faithfulness", "🎯 Faithfulness"),
    ("avg_answer_relevancy", "answer_relevancy", "📊 Answer Relevancy"),
    ("avg_context_recall", "context_recall", "🔍 Context Recall"),
    ("avg_context_precision", "context_precision", "⚡ Context Precision"),
]

for col, (summary_key, _, label) in zip(card_cols, metric_keys):
    score = float(summary.get(summary_key, 0.0))
    delta = round(score - _PASS_THRESHOLD, 4)
    color = _score_color(score)
    with col:
        st.metric(
            label=label,
            value=_fmt(score),
            delta=f"{delta:+.3f} vs 0.7 target",
            delta_color="normal",
        )
        st.markdown(
            f"<div style='height:6px;border-radius:3px;background:{color};'></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# 4. CHARTS ROW
# ---------------------------------------------------------------------------

st.subheader("Score Visualisation")
chart_left, chart_right = st.columns(2)

# --- Radar / spider chart ---
with chart_left:
    st.markdown("**Metric Radar**")
    radar_scores = [float(summary.get(k, 0.0)) for k, _, _ in metric_keys]
    radar_labels = [lbl for _, _, lbl in metric_keys]
    # Close the polygon
    radar_scores_closed = radar_scores + [radar_scores[0]]
    radar_labels_closed = radar_labels + [radar_labels[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_scores_closed,
            theta=radar_labels_closed,
            fill="toself",
            fillcolor="rgba(59,130,246,0.2)",
            line=dict(color="#3b82f6", width=2),
            name="Scores",
        )
    )
    # Target threshold ring
    fig_radar.add_trace(
        go.Scatterpolar(
            r=[_PASS_THRESHOLD] * len(radar_labels_closed),
            theta=radar_labels_closed,
            line=dict(color="#ef4444", width=1, dash="dot"),
            name="0.7 Target",
            mode="lines",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=380,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- Per-question bar chart ---
with chart_right:
    st.markdown("**Per-Question Overall Score**")
    if individual:
        bar_data = []
        for r in individual:
            scores = [r.get(m, 0.0) for m in _METRICS]
            valid = [s for s in scores if isinstance(s, float)]
            overall_q = sum(valid) / len(valid) if valid else 0.0
            bar_data.append(
                {
                    "question": r["question"][:55] + "…" if len(r["question"]) > 55 else r["question"],
                    "score": round(overall_q, 3),
                    "status": "Pass ✅" if r.get("passed") else "Fail ❌",
                }
            )
        df_bar = pd.DataFrame(bar_data)
        fig_bar = px.bar(
            df_bar,
            x="score",
            y="question",
            color="status",
            color_discrete_map={"Pass ✅": "#22c55e", "Fail ❌": "#ef4444"},
            orientation="h",
            range_x=[0, 1],
            labels={"score": "Overall Score", "question": ""},
        )
        fig_bar.add_vline(
            x=_PASS_THRESHOLD,
            line_dash="dot",
            line_color="#eab308",
            annotation_text="0.7 threshold",
        )
        fig_bar.update_layout(height=380, margin=dict(t=10, b=10), legend_title_text="")
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. RESULTS TABLE
# ---------------------------------------------------------------------------

st.subheader("Individual Results")

if individual:
    rows = []
    for r in individual:
        q = r["question"]
        rows.append(
            {
                "Question": q[:60] + "…" if len(q) > 60 else q,
                "Faithfulness": round(r.get("faithfulness", 0.0), 3),
                "Answer Relevancy": round(r.get("answer_relevancy", 0.0), 3),
                "Context Recall": round(r.get("context_recall", 0.0), 3),
                "Context Precision": round(r.get("context_precision", 0.0), 3),
                "Passed": "✅" if r.get("passed") else "❌",
            }
        )
    df_results = pd.DataFrame(rows)

    def _highlight_failed(row: pd.Series) -> list[str]:
        """Apply a red background to rows where the test case failed."""
        if row["Passed"] == "❌":
            return ["background-color: #fef2f2; color: #991b1b"] * len(row)
        return [""] * len(row)

    styled = df_results.style.apply(_highlight_failed, axis=1).format(
        {
            "Faithfulness": "{:.3f}",
            "Answer Relevancy": "{:.3f}",
            "Context Recall": "{:.3f}",
            "Context Precision": "{:.3f}",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 6. FAILED CASES EXPANDER
# ---------------------------------------------------------------------------

if failed:
    with st.expander(f"❌ Failed Cases — Needs Attention ({len(failed)} case(s))"):
        for i, case in enumerate(failed, start=1):
            st.markdown(f"**Case {i}: {case['question']}**")

            failed_metrics = [
                _METRIC_LABELS.get(m, m)
                for m in _METRICS
                if isinstance(case.get(m), float) and case[m] <= _PASS_THRESHOLD
            ]
            if failed_metrics:
                st.markdown(
                    "**Failed metrics:** " + " · ".join(
                        f"<span style='color:#ef4444'>{m}</span>" for m in failed_metrics
                    ),
                    unsafe_allow_html=True,
                )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Generated Answer**")
                st.info(case.get("generated_answer", "—"))
            with col_b:
                st.markdown("**Ground Truth**")
                st.success(case.get("ground_truth", "—"))

            score_cols = st.columns(4)
            for sc, metric in zip(score_cols, _METRICS):
                val = case.get(metric, 0.0)
                color = _score_color(val) if isinstance(val, float) else "#6b7280"
                sc.markdown(
                    f"<div style='text-align:center'>"
                    f"<span style='font-size:0.75rem;color:#6b7280'>{_METRIC_LABELS[metric]}</span><br>"
                    f"<span style='font-size:1.4rem;font-weight:700;color:{color}'>{_fmt(val) if isinstance(val, float) else '—'}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if i < len(failed):
                st.divider()
else:
    st.success("🎉 All test cases passed!")
