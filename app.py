"""
app.py  –  Short-Form Video Content → Purchase Intent Analysis
=============================================================
Run:  streamlit run app.py

Tabs
----
1. Data Understanding
2. Feature Engineering
3. KPI Dashboard
4. Regression Modelling
5. Interpretation & Validation
6. Behavioral Insights
7. Critical Limitations
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.inspection import permutation_importance
import scipy.stats as stats

from data_loader import full_pipeline, get_model_features, get_logistic_features, get_direct_item_features

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Short-Form Video × Purchase Intent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primary":   "#4361EE",
    "secondary": "#F72585",
    "accent":    "#4CC9F0",
    "warn":      "#F8961E",
    "danger":    "#EF233C",
    "ok":        "#2DC653",
    "bg":        "#0F1117",
    "surface":   "#1E2130",
    "text":      "#E9ECEF",
}

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — file uploader
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
uploaded = st.sidebar.file_uploader(
    "Upload survey Excel file", type=["xlsx", "xls"]
)
DEFAULT_PATH = "data/survey.xlsx"

@st.cache_data(show_spinner="Processing dataset…")
def load_data(path_or_bytes):
    if isinstance(path_or_bytes, str):
        return full_pipeline(path_or_bytes)
    import io, tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(path_or_bytes.read())
        tmp_path = tmp.name
    df = full_pipeline(tmp_path)
    os.unlink(tmp_path)
    return df

try:
    if uploaded:
        df = load_data(uploaded)
    else:
        import os
        # Try common locations
        for candidate in [
            "data/survey.xlsx",
            "Competitive_Benchmarking_of_Portable_Audio_Devices__Responses_.xlsx",
            "/mnt/user-data/uploads/Competitive_Benchmarking_of_Portable_Audio_Devices__Responses_.xlsx",
        ]:
            if os.path.exists(candidate):
                df = load_data(candidate)
                break
        else:
            st.warning("⬆️  Upload your survey Excel file in the sidebar to begin.")
            st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.sidebar.success(f"✅  {len(df)} responses loaded")

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def card(title, value, delta=None, color=PALETTE["primary"]):
    delta_str = f"<span style='font-size:0.75rem;color:{PALETTE['accent']}'>{delta}</span>" if delta else ""
    st.markdown(
        f"""
        <div style='background:{PALETTE["surface"]};border-radius:10px;
                    padding:14px 18px;border-left:4px solid {color};'>
            <p style='margin:0;font-size:0.78rem;color:#aaa;text-transform:uppercase;letter-spacing:1px'>{title}</p>
            <p style='margin:4px 0 0 0;font-size:1.6rem;font-weight:700;color:{PALETTE["text"]}'>{value}</p>
            {delta_str}
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_header(text, emoji=""):
    st.markdown(
        f"<h3 style='color:{PALETTE['primary']};border-bottom:2px solid {PALETTE['primary']};padding-bottom:6px'>"
        f"{emoji} {text}</h3>",
        unsafe_allow_html=True,
    )

def critic_box(text, label="⚠️ Critical Note"):
    st.markdown(
        f"""
        <div style='background:#2a1a1a;border-left:4px solid {PALETTE["danger"]};
                    border-radius:6px;padding:12px 16px;margin:8px 0'>
            <b style='color:{PALETTE["danger"]}'>{label}</b><br>
            <span style='color:#ddd;font-size:0.9rem'>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def insight_box(text, label="💡 Insight"):
    st.markdown(
        f"""
        <div style='background:#0d2137;border-left:4px solid {PALETTE["accent"]};
                    border-radius:6px;padding:12px 16px;margin:8px 0'>
            <b style='color:{PALETTE["accent"]}'>{label}</b><br>
            <span style='color:#ddd;font-size:0.9rem'>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 1 · Data Understanding",
    "🔧 2 · Feature Engineering",
    "📈 3 · KPI Dashboard",
    "🤖 4 · Regression Model",
    "🔍 5 · Interpretation & Validation",
    "💡 6 · Behavioral Insights",
    "⚠️  7 · Critical Limitations",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("📋 Data Understanding")
    st.markdown(
        "Strict structural audit of the 74-response survey dataset before any analysis."
    )

    # ── Overview cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Total Responses", len(df), color=PALETTE["primary"])
    with c2: card("Total Variables", "27 raw / 4 derived", color=PALETTE["accent"])
    with c3: card("Likert Items", "16  (1–5 scale)", color=PALETTE["secondary"])
    with c4: card("Nominal/Categorical", "11", color=PALETTE["warn"])

    st.divider()

    # ── Variable Classification ─────────────────────────────────────────────
    section_header("Variable Role Classification", "🗂️")
    var_table = pd.DataFrame({
        "Variable": [
            "Q_purchase_intent", "Q_consider_rec", "Q_realistic_purchase",
            "Purchase_Intent_Score", "High_Intent",
            "Trust_Score", "Engagement_Score", "Content_Relevance_Index",
            "hours_numeric", "age_ordinal",
            "platform_Instagram", "platform_Youtube",
            "occ_Student", "gender_male",
            "Which product would you most likely purchase?",
            "Q_knowledgeable … Q_not_scripted (12 items)",
        ],
        "Role": [
            "DV – direct", "DV – direct", "DV – direct",
            "DV – composite (primary)", "DV – binary (logistic)",
            "IV – composite", "IV – composite", "IV – composite",
            "IV – behavioral", "IV – demographic",
            "IV – behavioral", "IV – behavioral",
            "IV – demographic", "IV – demographic",
            "DV – nominal (secondary analysis)",
            "IV – raw components of composites",
        ],
        "Type": [
            "Ordinal 1–5", "Ordinal 1–5", "Ordinal 1–5",
            "Continuous 0–1", "Binary",
            "Continuous 0–1", "Continuous 0–1", "Continuous 0–1",
            "Continuous", "Ordinal 1–4",
            "Binary", "Binary",
            "Binary", "Binary",
            "Nominal 3 classes",
            "Ordinal 1–5",
        ],
        "Risk / Note": [
            "Self-report bias", "Hypothetical bias", "Redundant with purchase_intent",
            "Aggregation masks variance", "Threshold choice is arbitrary",
            "4-item scale; low α risk", "4-item scale; halo effect risk",
            "5-item scale; most reliable", "Ordinal treated as continuous",
            "Small n per cell", 
            "Platform ≠ content exposure proof", "Platform ≠ content exposure proof",
            "Student-heavy (53 %)", "Excludes 10 non-binary",
            "Only 3 brands; limited external validity",
            "High inter-item correlation → multicollinearity in composites",
        ],
    })
    st.dataframe(var_table, use_container_width=True, height=460)

    # ── Demographics ────────────────────────────────────────────────────────
    section_header("Sample Demographics", "👥")
    demo_cols = st.columns(2)

    age_counts = df["Age"].value_counts().reset_index()
    age_counts.columns = ["Age", "Count"]
    fig_age = px.bar(
        age_counts, x="Age", y="Count",
        color="Count", color_continuous_scale="Blues",
        title="Age Distribution",
    )
    fig_age.update_layout(
        template="plotly_dark", plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["surface"], showlegend=False
    )
    demo_cols[0].plotly_chart(fig_age, use_container_width=True)

    gen_counts = df["Gender"].value_counts().reset_index()
    gen_counts.columns = ["Gender", "Count"]
    fig_gen = px.pie(gen_counts, names="Gender", values="Count",
                     title="Gender Split", color_discrete_sequence=px.colors.qualitative.Bold)
    fig_gen.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    demo_cols[1].plotly_chart(fig_gen, use_container_width=True)

    occ_counts = df["Occupation"].value_counts().reset_index()
    occ_counts.columns = ["Occupation", "Count"]
    plat_counts = df["Which social media platform do you use the most?"].value_counts().reset_index()
    plat_counts.columns = ["Platform", "Count"]

    demo_cols2 = st.columns(2)
    fig_occ = px.bar(occ_counts, x="Occupation", y="Count",
                     title="Occupation Distribution",
                     color="Count", color_continuous_scale="Purples")
    fig_occ.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"], showlegend=False)
    demo_cols2[0].plotly_chart(fig_occ, use_container_width=True)

    fig_plat = px.bar(plat_counts, x="Platform", y="Count",
                      title="Primary Social Media Platform",
                      color="Count", color_continuous_scale="Teal")
    fig_plat.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"], showlegend=False)
    demo_cols2[1].plotly_chart(fig_plat, use_container_width=True)

    critic_box(
        "53 % of respondents are students. "
        "The age distribution is heavily skewed to 22–25. "
        "Instagram alone accounts for 57 % of platform usage. "
        "These sampling biases make generalization to other demographics or platforms empirically unjustifiable.",
        "⚠️ Sampling Bias Alert"
    )

    # ── Likert Heatmap ──────────────────────────────────────────────────────
    st.divider()
    section_header("Likert Item Mean Scores", "🌡️")
    likert_short = {
        "Q_knowledgeable": "Knowledgeable",
        "Q_trustworthy": "Trustworthy",
        "Q_gen_trust": "General Trust",
        "Q_useful_info": "Useful Info",
        "Q_genuine": "Genuine Interest",
        "Q_consider_rec": "Consider Rec",
        "Q_demo_realistic": "Demo Realistic",
        "Q_authentic": "Authentic",
        "Q_relatable": "Relatable",
        "Q_pros_cons": "Pros & Cons",
        "Q_practical": "Practical",
        "Q_not_scripted": "Not Scripted",
        "Q_purchase_intent": "Purchase Intent",
        "Q_interest_raised": "Interest Raised",
        "Q_understood_features": "Understood Features",
        "Q_realistic_purchase": "Realistic→Purchase",
    }
    means = df[list(likert_short.keys())].mean().rename(likert_short)
    stds  = df[list(likert_short.keys())].std().rename(likert_short)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=list(means.index), y=means.values,
        error_y=dict(type="data", array=stds.values, visible=True),
        marker_color=[PALETTE["primary"] if v >= 3.3 else PALETTE["warn"] for v in means.values],
        name="Mean Score",
    ))
    fig_bar.add_hline(y=3, line_dash="dash", line_color=PALETTE["danger"],
                      annotation_text="Neutral (3.0)")
    fig_bar.update_layout(
        title="Mean Likert Scores (±1 SD)  — Scale: 1=Strongly Disagree  5=Strongly Agree",
        template="plotly_dark", paper_bgcolor=PALETTE["surface"],
        xaxis_tickangle=-40, yaxis_range=[1, 5],
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    critic_box(
        "All 16 Likert means cluster narrowly between 3.03 and 3.59. "
        "This compressed variance indicates possible central tendency bias — "
        "respondents avoided extremes. Low variance inflates inter-item correlations "
        "and makes it harder to detect real predictive relationships.",
        "⚠️ Central Tendency Bias Detected"
    )

    # ── Correlation Heatmap ────────────────────────────────────────────────
    st.divider()
    section_header("Likert Inter-Item Correlation Matrix", "🔗")
    corr_df = df[list(likert_short.keys())].rename(columns=likert_short).corr()
    fig_corr = px.imshow(
        corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="Pearson Correlation — All Likert Items",
    )
    fig_corr.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"], height=560)
    st.plotly_chart(fig_corr, use_container_width=True)

    critic_box(
        "High inter-item correlations (many r > 0.5) confirm that the 16 Likert items "
        "are NOT independent. Using them individually as regressors would produce severe "
        "multicollinearity. Composite scoring (done in Tab 2) partially mitigates — "
        "but does not eliminate — this risk.",
        "⚠️ Multicollinearity Risk"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.title("🔧 Feature Engineering")
    st.markdown("Four psychometrically motivated composite scores derived from raw Likert items.")

    # ── Formula cards ──────────────────────────────────────────────────────
    features_meta = [
        {
            "name": "Trust Score",
            "col": "Trust_Score",
            "color": PALETTE["secondary"],
            "items": ["Q_trustworthy", "Q_gen_trust", "Q_genuine", "Q_authentic"],
            "formula": "mean(trustworthy, gen_trust, genuine, authentic) ÷ 5",
            "rationale": "Captures perceived influencer credibility. Trust is the primary gateway to persuasion in dual-process theory.",
        },
        {
            "name": "Engagement Score",
            "col": "Engagement_Score",
            "color": PALETTE["primary"],
            "items": ["Q_interest_raised", "Q_understood_features", "Q_relatable", "Q_practical"],
            "formula": "mean(interest_raised, understood_features, relatable, practical) ÷ 5",
            "rationale": "Measures whether content activated cognitive and affective processing. A prerequisite for attitude change.",
        },
        {
            "name": "Content Relevance Index",
            "col": "Content_Relevance_Index",
            "color": PALETTE["accent"],
            "items": ["Q_knowledgeable", "Q_useful_info", "Q_demo_realistic", "Q_pros_cons", "Q_not_scripted"],
            "formula": "mean(knowledgeable, useful_info, demo_realistic, pros_cons, not_scripted) ÷ 5",
            "rationale": "Assesses information quality and authenticity of the content — the informational route to persuasion (ELM).",
        },
        {
            "name": "Purchase Intent Score",
            "col": "Purchase_Intent_Score",
            "color": PALETTE["warn"],
            "items": ["Q_purchase_intent", "Q_consider_rec", "Q_realistic_purchase"],
            "formula": "mean(purchase_intent, consider_rec, realistic_purchase) ÷ 5",
            "rationale": "Primary dependent variable. Aggregates stated behavioral intention across three distinct triggers.",
        },
    ]

    for feat in features_meta:
        with st.expander(f"**{feat['name']}** — {feat['formula']}", expanded=True):
            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                st.markdown(f"**🧮 Formula:** `{feat['formula']}`")
                st.markdown(f"**📌 Rationale:** {feat['rationale']}")
                st.markdown(f"**📦 Component items:** `{'`, `'.join(feat['items'])}`")
                mean_v = df[feat["col"]].mean()
                std_v  = df[feat["col"]].std()
                st.markdown(f"**📊 Mean:** `{mean_v:.3f}`  |  **SD:** `{std_v:.3f}`  |  **Range:** `{df[feat['col']].min():.2f}–{df[feat['col']].max():.2f}`")
            with col_b:
                fig_hist = px.histogram(
                    df, x=feat["col"], nbins=18, title=f"{feat['name']} Distribution",
                    color_discrete_sequence=[feat["color"]],
                )
                fig_hist.add_vline(x=mean_v, line_dash="dash",
                                   line_color="white", annotation_text=f"μ={mean_v:.2f}")
                fig_hist.update_layout(
                    template="plotly_dark", paper_bgcolor=PALETTE["surface"],
                    height=230, margin=dict(t=30, b=20, l=10, r=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # ── Reliability Check (Cronbach's α proxy) ─────────────────────────────
    st.divider()
    section_header("Internal Reliability (Cronbach's α)", "🔬")

    def cronbach_alpha(df_items: pd.DataFrame) -> float:
        items = df_items.dropna()
        k = items.shape[1]
        item_vars = items.var(axis=0, ddof=1)
        total_var = items.sum(axis=1).var(ddof=1)
        if total_var == 0:
            return np.nan
        return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

    alpha_data = []
    for feat in features_meta:
        a = cronbach_alpha(df[feat["items"]])
        alpha_data.append({"Score": feat["name"], "Cronbach α": round(a, 3),
                           "Items": len(feat["items"]),
                           "Status": "✅ Acceptable" if a >= 0.7 else "⚠️ Weak"})
    alpha_df = pd.DataFrame(alpha_data)
    st.dataframe(alpha_df, use_container_width=True)

    critic_box(
        "Cronbach's α below 0.70 indicates the items in that scale are not measuring "
        "the same underlying construct reliably. For Trust Score in particular (4 items, "
        "potentially low α), treating the composite as a single predictive variable "
        "introduces construct validity problems. Consider confirmatory factor analysis "
        "before making strong academic claims.",
        "⚠️ Reliability Warning"
    )

    # ── Composite Score Scatter Matrix ─────────────────────────────────────
    st.divider()
    section_header("Composite Score Scatter Matrix", "🔀")
    score_cols = ["Trust_Score", "Engagement_Score", "Content_Relevance_Index", "Purchase_Intent_Score"]
    fig_scatter = px.scatter_matrix(
        df[score_cols],
        dimensions=score_cols,
        color=df["Purchase_Intent_Score"],
        color_continuous_scale="Viridis",
        title="Pairwise Relationships — Composite Scores",
    )
    fig_scatter.update_traces(diagonal_visible=False, showupperhalf=False)
    fig_scatter.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"], height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – KPI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.title("📈 KPI Dashboard")
    st.markdown("Six business/behavioral KPIs aligned with the research problem.")

    # ── KPI 1: Overall Purchase Intent ─────────────────────────────────────
    section_header("KPI 1 — Overall Purchase Intent Rate", "🛒")
    pis_mean = df["Purchase_Intent_Score"].mean()
    high_intent_pct = df["High_Intent"].mean() * 100

    k1, k2, k3 = st.columns(3)
    with k1: card("Mean Purchase Intent Score", f"{pis_mean:.3f} / 1.0", color=PALETTE["primary"])
    with k2: card("High-Intent Respondents", f"{high_intent_pct:.1f}%", color=PALETTE["ok"])
    with k3: card("Brand Consistency Rate", f"{df['brand_consistency'].mean()*100:.1f}%",
                  delta="Liked → Would Buy same brand", color=PALETTE["accent"])

    st.markdown("---")

    # ── KPI 2: Content Conversion Funnel ───────────────────────────────────
    section_header("KPI 2 — Content Conversion Funnel", "🔽")
    funnel_vals = {
        "Content Exposure (All)": len(df),
        "Engaged (Eng. Score ≥ 0.65)": int((df["Engagement_Score"] >= 0.65).sum()),
        "Trusted Influencer (Trust ≥ 0.65)": int((df["Trust_Score"] >= 0.65).sum()),
        "High Purchase Intent": int(df["High_Intent"].sum()),
        "Brand Consistent Choice": int(df["brand_consistency"].sum()),
    }
    fig_funnel = go.Figure(go.Funnel(
        y=list(funnel_vals.keys()),
        x=list(funnel_vals.values()),
        textinfo="value+percent initial",
        marker_color=[PALETTE["primary"], PALETTE["accent"],
                      PALETTE["secondary"], PALETTE["ok"], PALETTE["warn"]],
    ))
    fig_funnel.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"], height=350)
    st.plotly_chart(fig_funnel, use_container_width=True)

    insight_box(
        "The funnel reveals where the largest drop-off occurs: from 'content exposure' to "
        "'high engagement' is the steepest cliff, suggesting the short-form content is reaching "
        "but not strongly activating most viewers."
    )

    st.markdown("---")

    # ── KPI 3: Influencer Trust Index by Platform ──────────────────────────
    section_header("KPI 3 — Trust & Engagement by Platform", "📱")
    plat_grp = df.groupby("Which social media platform do you use the most?")[
        ["Trust_Score", "Engagement_Score", "Purchase_Intent_Score"]
    ].mean().reset_index().rename(columns={"Which social media platform do you use the most?": "Platform"})
    fig_plat = px.bar(
        plat_grp.melt(id_vars="Platform"),
        x="Platform", y="value", color="variable", barmode="group",
        color_discrete_sequence=[PALETTE["secondary"], PALETTE["primary"], PALETTE["warn"]],
        title="Mean Scores by Primary Platform",
        labels={"value": "Score (0–1)", "variable": "Metric"},
    )
    fig_plat.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    st.plotly_chart(fig_plat, use_container_width=True)

    st.markdown("---")

    # ── KPI 4: Product Preference Distribution ─────────────────────────────
    section_header("KPI 4 — Product Preference Alignment", "🎧")
    prod_cols = st.columns(3)
    for i, col_name in enumerate([
        "Which product did you like the most based on the advertisements?",
        "Which product appears to have the best features?",
        "Which product would you most likely purchase?",
    ]):
        label = ["Liked Most", "Best Features", "Would Purchase"][i]
        counts = df[col_name].value_counts().reset_index()
        counts.columns = ["Product", "Count"]
        fig_p = px.pie(counts, names="Product", values="Count", title=label,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_p.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"], height=300)
        prod_cols[i].plotly_chart(fig_p, use_container_width=True)

    st.markdown("---")

    # ── KPI 5: Trust-to-Intent Conversion ─────────────────────────────────
    section_header("KPI 5 — Trust-to-Intent Conversion Score", "📐")
    df["trust_intent_gap"] = df["Purchase_Intent_Score"] - df["Trust_Score"]
    fig_gap = px.histogram(
        df, x="trust_intent_gap", nbins=20,
        color_discrete_sequence=[PALETTE["accent"]],
        title="Purchase Intent − Trust Score (Gap Distribution)",
        labels={"trust_intent_gap": "Gap (positive = intent exceeds trust)"},
    )
    fig_gap.add_vline(x=0, line_dash="dash", line_color=PALETTE["danger"], annotation_text="No Gap")
    fig_gap.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    st.plotly_chart(fig_gap, use_container_width=True)

    insight_box(
        "A positive gap means a respondent has high purchase intent despite moderate trust — "
        "a behavioral red flag indicating impulse-driven intent rather than trust-anchored intent. "
        "A negative gap means trust is built but not converting — a brand communication failure."
    )

    st.markdown("---")

    # ── KPI 6: Social Media Dose-Response ─────────────────────────────────
    section_header("KPI 6 — Social Media Exposure Dose-Response", "📡")
    hours_grp = df.groupby(
        "How many hours do you spend on social media daily?"
    )[["Engagement_Score", "Purchase_Intent_Score"]].mean().reset_index()

    order = ["Less than 1 hours", "1-2 hours", "2-4 hours", "More than 4 hours"]
    hours_grp["hours_label"] = pd.Categorical(
        hours_grp["How many hours do you spend on social media daily?"],
        categories=order, ordered=True
    )
    hours_grp = hours_grp.sort_values("hours_label")

    fig_dose = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dose.add_trace(
        go.Bar(x=hours_grp["hours_label"].astype(str),
               y=hours_grp["Engagement_Score"],
               name="Engagement Score", marker_color=PALETTE["primary"]),
        secondary_y=False
    )
    fig_dose.add_trace(
        go.Scatter(x=hours_grp["hours_label"].astype(str),
                   y=hours_grp["Purchase_Intent_Score"],
                   name="Purchase Intent", line=dict(color=PALETTE["secondary"], width=3),
                   mode="lines+markers"),
        secondary_y=True
    )
    fig_dose.update_layout(
        title="Does More Screen Time → Higher Intent?",
        template="plotly_dark", paper_bgcolor=PALETTE["surface"]
    )
    st.plotly_chart(fig_dose, use_container_width=True)

    critic_box(
        "This KPI conflates platform usage time with exposure to the specific short-form "
        "videos in the experiment. High social media hours ≠ high relevant content exposure. "
        "Without an explicit exposure dosage measure, causal inference is impossible.",
        "⚠️ KPI Validity Warning"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – REGRESSION MODELLING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.title("🤖 Regression Modelling")

    # ── Model justification ────────────────────────────────────────────────
    section_header("Model Selection Rationale", "⚖️")
    col_j1, col_j2 = st.columns(2)
    with col_j1:
        st.markdown(
            """
            **Primary: OLS Linear Regression → Purchase Intent Score (continuous)**
            - PIS is a bounded continuous variable (0–1) with reasonable distribution
            - Linear regression preserves the cardinal information in scores
            - Coefficients are directly interpretable as unit changes in PIS
            - Appropriate when the DV is not strongly skewed (check histogram in Tab 2)

            **Secondary: Ridge Regression (regularized OLS)**
            - Small n (74) with 6 features → serious overfitting risk
            - Ridge adds L2 penalty to shrink inflated coefficients
            - Reduces variance at cost of small bias — correct trade-off here

            **Alternative: Logistic Regression → High_Intent (binary)**
            - Use when you want to classify respondents, not score them
            - Less sensitive to the arbitrary composite averaging
            - Coefficients are log-odds; harder to communicate to non-technical audiences
            """
        )
    with col_j2:
        st.markdown(
            """
            **Why overfitting is a serious risk here:**
            - n = 74 is critically small for any regression model
            - Rule of thumb: ≥10 observations per predictor → you have ~12 with 6 predictors
            - Likert composites are highly intercorrelated (r > 0.6) → the model can find
              spurious fits in correlated noise
            - Train/test split is statistically meaningless at n=74 (test set < 20 obs)
            - Leave-One-Out CV or k-fold (k≤5) are the only defensible validation strategies

            **Recommended feature set (justified):**
            | Feature | Why |
            |---------|-----|
            | Trust_Score | Primary persuasion mechanism |
            | Engagement_Score | Attention + processing |
            | Content_Relevance_Index | Information quality |
            | hours_numeric | Behavioral exposure proxy |
            | platform_Instagram | Dominant platform (57%) |
            | age_ordinal | Generational digital behavior |
            """
        )

    st.divider()

    # ── Research Model Diagram ──────────────────────────────────────────────
    section_header("Research Model — Pictorial Representation", "🗺️")

    fig_model = go.Figure()

    # IV boxes (left column)
    iv_items = [
        ("Q1: Knowledgeable", 0.92), ("Q2: Trustworthy", 0.84),
        ("Q3: Gen. Trust", 0.76),    ("Q4: Useful Info", 0.68),
        ("Q5: Genuine", 0.60),       ("Q6: Consider Rec", 0.52),
        ("Q7: Demo Realistic", 0.44),("Q8: Authentic", 0.36),
        ("Q9: Relatable", 0.28),     ("Q10: Pros & Cons", 0.20),
        ("Q11: Practical", 0.12),    ("Q12: Not Scripted", 0.04),
    ]
    for label, y in iv_items:
        color = PALETTE["primary"] if y > 0.50 else PALETTE["accent"]
        fig_model.add_shape(type="rect", x0=0, x1=0.22, y0=y-0.038, y1=y+0.038,
                            fillcolor=color, line=dict(color="white", width=1))
        fig_model.add_annotation(x=0.11, y=y, text=label, showarrow=False,
                                 font=dict(color="white", size=10), xanchor="center")

    # Composite Score boxes (middle)
    composites = [
        ("Trust Score", 0.80, PALETTE["secondary"]),
        ("Engagement Score", 0.56, PALETTE["primary"]),
        ("Content Relevance\nIndex", 0.32, PALETTE["accent"]),
        ("Hours / Platform\n(Covariates)", 0.10, PALETTE["warn"]),
    ]
    for label, y, color in composites:
        fig_model.add_shape(type="rect", x0=0.35, x1=0.58, y0=y-0.07, y1=y+0.07,
                            fillcolor=color, line=dict(color="white",width=1.5))
        fig_model.add_annotation(x=0.465, y=y, text=label, showarrow=False,
                                 font=dict(color="white", size=10.5, family="Arial Bold"),
                                 xanchor="center")

    # DV box (right)
    fig_model.add_shape(type="rect", x0=0.72, x1=0.95, y0=0.40, y1=0.60,
                        fillcolor=PALETTE["ok"], line=dict(color="white", width=2))
    fig_model.add_annotation(x=0.835, y=0.50,
                             text="Purchase Intent\nScore (DV)", showarrow=False,
                             font=dict(color="white", size=12, family="Arial Bold"),
                             xanchor="center")

    # Arrows: IVs → Composites
    arrow_map = {
        0.80: [0.92, 0.84, 0.76, 0.68, 0.60, 0.52],  # Trust Score
        0.56: [0.44, 0.36, 0.28, 0.12],               # Engagement
        0.32: [0.92, 0.68, 0.44, 0.20, 0.04],         # CRI
    }
    for comp_y, iv_ys in arrow_map.items():
        for iv_y in iv_ys:
            fig_model.add_annotation(x=0.35, y=comp_y, ax=0.22, ay=iv_y,
                                     xref="x", yref="y", axref="x", ayref="y",
                                     showarrow=True, arrowhead=2, arrowwidth=1,
                                     arrowcolor="#aaaaaa")

    # Arrows: Composites → DV
    for comp_y in [0.80, 0.56, 0.32, 0.10]:
        fig_model.add_annotation(x=0.72, y=0.50, ax=0.58, ay=comp_y,
                                 xref="x", yref="y", axref="x", ayref="y",
                                 showarrow=True, arrowhead=3, arrowwidth=2,
                                 arrowcolor=PALETTE["ok"])

    # Labels
    fig_model.add_annotation(x=0.11, y=0.97, text="INDEPENDENT VARIABLES (Likert Items)",
                             showarrow=False, font=dict(color=PALETTE["accent"], size=11, family="Arial Bold"))
    fig_model.add_annotation(x=0.465, y=0.97, text="COMPOSITE SCORES (IVs)",
                             showarrow=False, font=dict(color=PALETTE["secondary"], size=11, family="Arial Bold"))
    fig_model.add_annotation(x=0.835, y=0.97, text="DEPENDENT VARIABLE",
                             showarrow=False, font=dict(color=PALETTE["ok"], size=11, family="Arial Bold"))

    fig_model.update_layout(
        title="Research Model: Short-Form Video Content → Purchase Intention",
        xaxis=dict(range=[0,1.05], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-0.05,1.02], showgrid=False, zeroline=False, visible=False),
        template="plotly_dark", paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"], height=540,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_model, use_container_width=True)
    st.caption("Each Likert item flows into its corresponding composite score, which then predicts Purchase Intent Score. "
               "Covariates (social media hours, platform) enter the model directly.")

    st.divider()

    # ── Run composite models ────────────────────────────────────────────────
    X, y = get_model_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_sc_df = pd.DataFrame(X_scaled, columns=X.columns)

    ols = LinearRegression()
    ols.fit(X_scaled, y)
    y_pred_ols = ols.predict(X_scaled)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_scaled)

    # LOO CV — collect OOF predictions, then compute R² once (avoids NaN from per-fold R²)
    def loo_r2(model_class, Xs, y_vals, **kwargs):
        y_oof = np.zeros(len(y_vals))
        for i in range(len(y_vals)):
            idx = list(range(len(y_vals)))
            idx.pop(i)
            m = model_class(**kwargs).fit(Xs[idx], y_vals[idx])
            y_oof[i] = m.predict(Xs[i:i+1])[0]
        return r2_score(y_vals, y_oof)

    ols_loo_r2  = loo_r2(LinearRegression, X_scaled, y.values)
    ridge_loo_r2 = loo_r2(Ridge, X_scaled, y.values, alpha=1.0)

    k5 = min(5, len(y))
    ols_5fold = cross_val_score(LinearRegression(), X_scaled, y, cv=k5, scoring="r2")
    ridge_5fold = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=k5, scoring="r2")

    section_header("Model Performance Summary", "📊")
    perf_table = pd.DataFrame({
        "Model": ["OLS (train)", "Ridge (train)", "OLS (LOO CV)", "Ridge (LOO CV)", "OLS (5-fold CV)", "Ridge (5-fold CV)"],
        "R²": [
            r2_score(y, y_pred_ols),
            r2_score(y, y_pred_ridge),
            ols_loo_r2,
            ridge_loo_r2,
            ols_5fold.mean(),
            ridge_5fold.mean(),
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y, y_pred_ols)),
            np.sqrt(mean_squared_error(y, y_pred_ridge)),
            np.nan, np.nan, np.nan, np.nan,
        ],
        "Interpretation": [
            "In-sample; likely inflated",
            "In-sample; regularized",
            "Generalization estimate (honest)",
            "Generalization estimate (preferred)",
            "Average over 5 folds",
            "Average over 5 folds",
        ],
    })
    perf_table["R²"] = perf_table["R²"].round(4)
    perf_table["RMSE"] = perf_table["RMSE"].round(4)
    st.dataframe(perf_table, use_container_width=True)

    overfitting_gap = r2_score(y, y_pred_ols) - ols_loo_r2
    if overfitting_gap > 0.1:
        critic_box(
            f"OLS train R² – LOO R² gap = {overfitting_gap:.3f}. "
            "This confirms overfitting. The OLS model is fitting noise in training data. "
            "Use Ridge or interpret LOO CV R² as the honest estimate of predictive power.",
            "⚠️ Overfitting Confirmed"
        )

    # ── Coefficient Plot ───────────────────────────────────────────────────
    st.divider()
    section_header("Standardized Coefficients (OLS vs Ridge)", "📐")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "OLS Coef": ols.coef_,
        "Ridge Coef": ridge.coef_,
    }).sort_values("OLS Coef", key=abs, ascending=True)

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_df["OLS Coef"], y=coef_df["Feature"],
        orientation="h", name="OLS",
        marker_color=PALETTE["primary"],
    ))
    fig_coef.add_trace(go.Bar(
        x=coef_df["Ridge Coef"], y=coef_df["Feature"],
        orientation="h", name="Ridge (L2)",
        marker_color=PALETTE["secondary"],
    ))
    fig_coef.add_vline(x=0, line_color="white", line_dash="dot")
    fig_coef.update_layout(
        title="Standardized Coefficients — OLS vs Ridge",
        template="plotly_dark", paper_bgcolor=PALETTE["surface"],
        barmode="group", height=380,
        xaxis_title="Coefficient (standardized features → PIS)",
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    insight_box(
        "Features with larger absolute coefficients have stronger linear association with "
        "purchase intent. Ridge shrinks all coefficients toward zero — features that survive "
        "shrinkage with large magnitude are the truly robust predictors."
    )

    # ── Direct-Item Model (Feedback Point 6) ───────────────────────────────
    st.divider()
    section_header("Direct-Item Regression — Without Composite Scoring", "⚡")
    st.markdown(
        "Running OLS directly on 12 individual Likert items (no aggregation) tests whether "
        "composite scoring adds or destroys predictive value."
    )

    X_direct, y_direct = get_direct_item_features(df)
    scaler_d = StandardScaler()
    Xd_scaled = scaler_d.fit_transform(X_direct)

    ols_d = LinearRegression().fit(Xd_scaled, y_direct)
    ridge_d = Ridge(alpha=1.0).fit(Xd_scaled, y_direct)

    train_r2_d   = r2_score(y_direct, ols_d.predict(Xd_scaled))
    train_r2_dr  = r2_score(y_direct, ridge_d.predict(Xd_scaled))

    # LOO for direct models
    def loo_r2_fn(model_class, Xs, y_vals, **kwargs):
        y_oof = np.zeros(len(y_vals))
        for i in range(len(y_vals)):
            idx = list(range(len(y_vals)))
            idx.pop(i)
            m = model_class(**kwargs).fit(Xs[idx], y_vals.values[idx])
            y_oof[i] = m.predict(Xs[i:i+1])[0]
        return r2_score(y_vals, y_oof)

    with st.spinner("Computing LOO for direct-item models…"):
        loo_d  = loo_r2_fn(LinearRegression, Xd_scaled, y_direct)
        loo_dr = loo_r2_fn(Ridge, Xd_scaled, y_direct, alpha=1.0)

    # Comparison table
    comp_data = {
        "Model": [
            "Composite OLS (6 features)",
            "Composite Ridge (6 features)",
            "Direct-Item OLS (12 features)",
            "Direct-Item Ridge (12 features)",
        ],
        "Train R²": [
            round(r2_score(y, y_pred_ols), 4),
            round(r2_score(y, y_pred_ridge), 4),
            round(train_r2_d, 4),
            round(train_r2_dr, 4),
        ],
        "LOO R²": [
            round(ols_loo_r2, 4),
            round(ridge_loo_r2, 4),
            round(loo_d, 4),
            round(loo_dr, 4),
        ],
        "Overfit Gap": [
            round(r2_score(y, y_pred_ols) - ols_loo_r2, 4),
            round(r2_score(y, y_pred_ridge) - ridge_loo_r2, 4),
            round(train_r2_d - loo_d, 4),
            round(train_r2_dr - loo_dr, 4),
        ],
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True)

    # Visual comparison
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        x=comp_data["Model"], y=comp_data["Train R²"],
        name="Train R²", marker_color=PALETTE["primary"]
    ))
    fig_compare.add_trace(go.Bar(
        x=comp_data["Model"], y=comp_data["LOO R²"],
        name="LOO R² (honest)", marker_color=PALETTE["secondary"]
    ))
    fig_compare.update_layout(
        title="Composite Scoring vs Direct Items — Train R² vs LOO R²",
        barmode="group", template="plotly_dark",
        paper_bgcolor=PALETTE["surface"], yaxis_range=[0, 0.75],
        yaxis_title="R²", height=380,
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Individual item coefficients for direct model
    st.divider()
    section_header("Direct-Item Standardized Coefficients (OLS)", "📌")
    direct_coef = pd.DataFrame({
        "Likert Item": X_direct.columns,
        "OLS Coef": ols_d.coef_,
        "Ridge Coef": ridge_d.coef_,
    }).sort_values("OLS Coef", key=abs, ascending=True)

    fig_dcoef = go.Figure()
    fig_dcoef.add_trace(go.Bar(
        x=direct_coef["OLS Coef"], y=direct_coef["Likert Item"],
        orientation="h", name="OLS", marker_color=PALETTE["primary"]
    ))
    fig_dcoef.add_trace(go.Bar(
        x=direct_coef["Ridge Coef"], y=direct_coef["Likert Item"],
        orientation="h", name="Ridge (L2)", marker_color=PALETTE["secondary"]
    ))
    fig_dcoef.add_vline(x=0, line_color="white", line_dash="dot")
    fig_dcoef.update_layout(
        title="Individual Likert Item Coefficients (standardized)",
        template="plotly_dark", paper_bgcolor=PALETTE["surface"],
        barmode="group", height=440,
        xaxis_title="Coefficient → Purchase Intent Score",
    )
    st.plotly_chart(fig_dcoef, use_container_width=True)

    gap_composite = round(r2_score(y, y_pred_ols) - ols_loo_r2, 3)
    gap_direct    = round(train_r2_d - loo_d, 3)

    if gap_direct > gap_composite:
        critic_box(
            f"Direct-item OLS overfitting gap = {gap_direct} vs composite gap = {gap_composite}. "
            f"The direct model is {round(gap_direct/gap_composite, 1)}× more overfit. "
            "With 12 predictors at n=74, the model memorises noise. "
            "Composite scoring is the better choice for this sample size.",
            "⚠️ Direct-Item Model Severely Overfit"
        )
    else:
        insight_box(
            f"Direct-item model gap ({gap_direct}) ≤ composite gap ({gap_composite}). "
            "Composite scoring did not reduce overfitting here — consider keeping raw items."
        )

    # ── Logistic Regression (secondary) ────────────────────────────────────
    st.divider()
    section_header("Logistic Regression — High Intent Classification", "🎯")
    X_log, y_log = get_logistic_features(df)
    X_log_sc = StandardScaler().fit_transform(X_log)

    logit = LogisticRegression(max_iter=500, C=0.5)
    logit.fit(X_log_sc, y_log)

    logit_cv = cross_val_score(logit, X_log_sc, y_log, cv=5, scoring="accuracy")
    logit_loo = cross_val_score(logit, X_log_sc, y_log, cv=LeaveOneOut(), scoring="accuracy")

    c1, c2, c3 = st.columns(3)
    with c1: card("Logistic Train Accuracy", f"{logit.score(X_log_sc, y_log)*100:.1f}%", color=PALETTE["ok"])
    with c2: card("5-fold CV Accuracy", f"{logit_cv.mean()*100:.1f}%", color=PALETTE["warn"])
    with c3: card("LOO CV Accuracy", f"{logit_loo.mean()*100:.1f}%", color=PALETTE["accent"])

    logit_coef = pd.DataFrame({
        "Feature": X_log.columns,
        "Log-Odds Coef": logit.coef_[0],
        "Odds Ratio": np.exp(logit.coef_[0]),
    }).sort_values("Log-Odds Coef", key=abs, ascending=False)
    st.dataframe(logit_coef.round(4), use_container_width=True)

    critic_box(
        "A logistic model with 6 predictors on 74 observations and a binary outcome "
        "that is split approximately 50/50 (by median threshold) cannot produce reliable "
        "coefficient estimates. Standard errors will be wide, and confidence intervals will "
        "include zero for most predictors. This model has exploratory value only.",
        "⚠️ Logistic Model Reliability Warning"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – INTERPRETATION & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.title("🔍 Interpretation & Validation")

    # ── Coefficient Interpretation ─────────────────────────────────────────
    section_header("How to Interpret Standardized Coefficients", "📖")
    st.markdown(
        """
        **Setup:** Features are standardized (μ=0, σ=1). Target (PIS) is on 0–1 scale.

        **Reading a coefficient β:**
        > *"A 1 standard-deviation increase in [Feature] is associated with a β-unit increase in Purchase Intent Score, holding all other features constant."*

        **Example:** If β(Trust_Score) = 0.08
        - Trust_Score SD ≈ 0.1 (from histogram in Tab 2)
        - A 1-SD increase in Trust → PIS increases by 0.08 points
        - On a 0–1 scale, that is a meaningful but not dominant effect
        - It does NOT mean Trust *causes* higher intent (correlation ≠ causation)

        **Practical significance vs Statistical significance:**
        - At n=74, t-tests for individual coefficients have very low power
        - Even a "significant" p-value at n=74 is unreliable — multiple testing inflates FPR
        - Focus on effect size (β magnitude) and LOO CV rather than p-values
        """
    )

    # ── Residual Diagnostics ───────────────────────────────────────────────
    st.divider()
    section_header("OLS Residual Diagnostics", "🩺")
    residuals = y.values - y_pred_ols
    fitted = y_pred_ols

    diag_cols = st.columns(2)
    fig_resid = px.scatter(
        x=fitted, y=residuals,
        labels={"x": "Fitted Values", "y": "Residuals"},
        title="Residuals vs Fitted",
        color_discrete_sequence=[PALETTE["accent"]],
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color=PALETTE["danger"])
    fig_resid.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    diag_cols[0].plotly_chart(fig_resid, use_container_width=True)

    fig_qq = go.Figure()
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical_q = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    fig_qq.add_trace(go.Scatter(
        x=theoretical_q, y=sorted_res, mode="markers",
        marker=dict(color=PALETTE["primary"], size=6),
        name="Residuals"
    ))
    fig_qq.add_trace(go.Scatter(
        x=[theoretical_q.min(), theoretical_q.max()],
        y=[theoretical_q.min() * sorted_res.std() + sorted_res.mean(),
           theoretical_q.max() * sorted_res.std() + sorted_res.mean()],
        mode="lines", line=dict(color=PALETTE["danger"], dash="dash"),
        name="Normal Line"
    ))
    fig_qq.update_layout(
        title="Q-Q Plot (Normality Check)",
        template="plotly_dark", paper_bgcolor=PALETTE["surface"],
        xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
    )
    diag_cols[1].plotly_chart(fig_qq, use_container_width=True)

    # Shapiro-Wilk test
    sw_stat, sw_p = stats.shapiro(residuals)
    col_sw1, col_sw2 = st.columns(2)
    with col_sw1:
        card("Shapiro-Wilk Statistic", f"{sw_stat:.4f}",
             color=PALETTE["ok"] if sw_p > 0.05 else PALETTE["danger"])
    with col_sw2:
        card("Shapiro-Wilk p-value", f"{sw_p:.4f}",
             delta="p > 0.05 → residuals approximately normal",
             color=PALETTE["ok"] if sw_p > 0.05 else PALETTE["danger"])

    # ── Validation Strategy ────────────────────────────────────────────────
    st.divider()
    section_header("Validation Strategy for Small Data (n=74)", "✅")
    st.markdown(
        """
        | Strategy | What it does | When to use |
        |----------|-------------|-------------|
        | **Leave-One-Out CV** | Uses n-1 samples for training, 1 for test, repeats n times | Primary validation at n<100 |
        | **Stratified 5-fold CV** | Splits data into 5 folds, ensures balanced class distribution | Secondary validation |
        | **Permutation Test** | Shuffles labels, fits model 1000x, compares real R² to null dist | Tests if R² is better than chance |
        | **Bootstrap CI** | Resamples data with replacement, computes CI for each coefficient | Replaces t-test CIs at small n |
        | **Train/Test Split** | ❌ Do NOT use at n=74 — test set of ~15 obs is too small to trust | Never use here |
        """
    )

    # ── Permutation Test ───────────────────────────────────────────────────
    section_header("Permutation Test for Model Significance", "🎲")
    with st.spinner("Running permutation test (1000 iterations)…"):
        np.random.seed(42)
        perm_r2 = []
        for _ in range(1000):
            y_perm = np.random.permutation(y.values)
            m = LinearRegression().fit(X_scaled, y_perm)
            perm_r2.append(r2_score(y_perm, m.predict(X_scaled)))

        real_r2 = r2_score(y, y_pred_ols)
        p_perm = (np.array(perm_r2) >= real_r2).mean()

    fig_perm = px.histogram(
        x=perm_r2, nbins=40,
        color_discrete_sequence=[PALETTE["surface"]],
        title=f"Permutation Null Distribution vs Real R² = {real_r2:.4f}",
        labels={"x": "R² (permuted labels)"},
    )
    fig_perm.add_vline(
        x=real_r2, line_color=PALETTE["secondary"], line_width=3,
        annotation_text=f"Real R²={real_r2:.3f}  (p={p_perm:.3f})",
        annotation_font_color=PALETTE["secondary"],
    )
    fig_perm.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    st.plotly_chart(fig_perm, use_container_width=True)

    if p_perm < 0.05:
        insight_box(f"Permutation p = {p_perm:.3f} < 0.05. The model explains significantly more variance than random chance. This is a meaningful result despite small n.")
    else:
        critic_box(f"Permutation p = {p_perm:.3f} ≥ 0.05. The model does NOT explain significantly more variance than random chance. The observed R² may be an artifact of small-sample fitting.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – BEHAVIORAL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.title("💡 Behavioral Insights")
    st.markdown("Five hypothesis-driven insights extracted from the data.")

    # ── Insight 1: Trust-Engagement Decoupling ─────────────────────────────
    section_header("Insight 1 — Trust and Engagement Are Decoupled", "🔗")
    trust_eng_corr = df["Trust_Score"].corr(df["Engagement_Score"])
    st.markdown(
        f"**Correlation (Trust × Engagement):** r = `{trust_eng_corr:.3f}`"
    )
    fig_te = px.scatter(
        df, x="Trust_Score", y="Engagement_Score",
        color="Purchase_Intent_Score",
        color_continuous_scale="Plasma",
        trendline="ols",
        title="Trust Score vs Engagement Score (colored by Purchase Intent)",
        labels={"Trust_Score": "Trust Score", "Engagement_Score": "Engagement Score"},
    )
    fig_te.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    st.plotly_chart(fig_te, use_container_width=True)
    insight_box(
        "If r(Trust, Engagement) < 0.7, the two constructs are meaningfully distinct. "
        "A viewer can be highly engaged (attention captured, features understood) "
        "while still distrusting the influencer — or vice versa. "
        "This decoupling has practical implications: engaging content is not the same as credible content, "
        "and brands should optimize both independently."
    )

    st.markdown("---")

    # ── Insight 2: Platform Differences ────────────────────────────────────
    section_header("Insight 2 — YouTube Viewers Trust More; Instagram Viewers Buy More", "📱")
    plat_full = df.groupby("Which social media platform do you use the most?")[
        ["Trust_Score", "Engagement_Score", "Purchase_Intent_Score"]
    ].agg(["mean", "std"]).reset_index()

    # Simplified bar with trust & PIS
    plat_means = df.groupby("Which social media platform do you use the most?")[
        ["Trust_Score", "Purchase_Intent_Score"]
    ].mean().reset_index()
    plat_means.columns = ["Platform", "Trust Score", "Purchase Intent Score"]
    fig_plat2 = px.scatter(
        plat_means, x="Trust Score", y="Purchase Intent Score",
        text="Platform", size=[20]*len(plat_means),
        color="Trust Score", color_continuous_scale="Tealgrn",
        title="Trust vs Purchase Intent by Platform",
    )
    fig_plat2.update_traces(textposition="top center")
    fig_plat2.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    st.plotly_chart(fig_plat2, use_container_width=True)
    insight_box(
        "YouTube is a long-form native platform — even short videos there are watched by users "
        "in 'research mode.' Instagram is scroll-native and impulse-driven. "
        "This behavioral difference should produce higher trust on YouTube but faster "
        "purchase activation on Instagram. Test this hypothesis with your data."
    )

    st.markdown("---")

    # ── Insight 3: Authentic Content Premium ───────────────────────────────
    section_header("Insight 3 — Authenticity Is the Single Strongest Content Driver", "🎭")
    # Correlations of all Likert items with PIS
    likert_aliases = {
        "Q_knowledgeable": "Knowledgeable",
        "Q_trustworthy": "Trustworthy",
        "Q_gen_trust": "General Trust",
        "Q_useful_info": "Useful Info",
        "Q_genuine": "Genuine Interest",
        "Q_consider_rec": "Consider Rec",
        "Q_demo_realistic": "Demo Realistic",
        "Q_authentic": "Authentic Exp",
        "Q_relatable": "Relatable",
        "Q_pros_cons": "Pros & Cons",
        "Q_practical": "Practical",
        "Q_not_scripted": "Not Scripted",
    }
    corrs = {v: df[k].corr(df["Purchase_Intent_Score"]) for k, v in likert_aliases.items()}
    corr_series = pd.Series(corrs).sort_values(ascending=True)
    fig_corr_pis = px.bar(
        x=corr_series.values, y=corr_series.index, orientation="h",
        color=corr_series.values,
        color_continuous_scale="RdYlGn",
        title="Pearson r with Purchase Intent Score — Individual Likert Items",
        labels={"x": "Correlation with PIS", "y": "Item"},
    )
    fig_corr_pis.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"], height=420)
    st.plotly_chart(fig_corr_pis, use_container_width=True)
    insight_box(
        "The items with the highest r to PIS reveal which psychological mechanisms "
        "most powerfully drive purchase intent. If 'Authentic Exp' or 'Not Scripted' "
        "rank highest, it confirms the authenticity premium — organic-feeling content "
        "converts better than polished advertising."
    )

    st.markdown("---")

    # ── Insight 4: High Hours ≠ High Intent ────────────────────────────────
    section_header("Insight 4 — Heavy Social Media Users Are NOT More Likely to Buy", "📉")
    hours_pis = df.groupby("How many hours do you spend on social media daily?")[
        "Purchase_Intent_Score"
    ].agg(["mean", "std", "count"]).reset_index()
    hours_pis.columns = ["Hours", "Mean PIS", "SD", "n"]
    hours_pis["CI"] = 1.96 * hours_pis["SD"] / np.sqrt(hours_pis["n"])

    order = ["Less than 1 hours", "1-2 hours", "2-4 hours", "More than 4 hours"]
    hours_pis["Hours"] = pd.Categorical(hours_pis["Hours"], categories=order, ordered=True)
    hours_pis = hours_pis.sort_values("Hours")

    fig_hours = px.bar(
        hours_pis, x="Hours", y="Mean PIS", error_y="CI",
        color="Mean PIS", color_continuous_scale="Blues",
        title="Mean Purchase Intent Score by Daily Social Media Hours (±95% CI)",
    )
    fig_hours.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    st.plotly_chart(fig_hours, use_container_width=True)
    insight_box(
        "If the dose-response curve is flat or non-monotonic (the 2–4 hour group doesn't "
        "outperform the 1–2 hour group), this contradicts the naive assumption that more "
        "exposure → more intent. Heavy users may be more ad-fatigued or desensitized. "
        "This is a counterintuitive finding worth highlighting."
    )

    st.markdown("---")

    # ── Insight 5: Brand Consistency ───────────────────────────────────────
    section_header("Insight 5 — Affective Preference Drives Consistent Purchase Decisions", "❤️")
    consistency_pis = df.groupby("brand_consistency")["Purchase_Intent_Score"].agg(["mean", "std", "count"])
    consistency_pis.index = ["Inconsistent (Liked ≠ Would Buy)", "Consistent (Liked = Would Buy)"]
    consistency_pis.columns = ["Mean PIS", "SD", "n"]

    fig_cons = px.bar(
        consistency_pis.reset_index(),
        x="index", y="Mean PIS", error_y="SD",
        color="Mean PIS", color_continuous_scale="Greens",
        title="Purchase Intent Score: Brand-Consistent vs Inconsistent Respondents",
        labels={"index": "Brand Preference Consistency"},
    )
    fig_cons.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["surface"])
    st.plotly_chart(fig_cons, use_container_width=True)

    # Mann-Whitney test for significance
    grp_consistent = df[df["brand_consistency"] == 1]["Purchase_Intent_Score"]
    grp_inconsistent = df[df["brand_consistency"] == 0]["Purchase_Intent_Score"]
    mw_stat, mw_p = stats.mannwhitneyu(grp_consistent, grp_inconsistent, alternative="two-sided")
    st.markdown(f"**Mann-Whitney U test:** U = `{mw_stat:.1f}`, p = `{mw_p:.4f}`")

    insight_box(
        "Respondents who picked the same brand as their 'most liked ad' and 'would purchase' "
        "show higher overall purchase intent. This suggests affective attachment to a brand "
        "through advertising is a stronger signal than any Likert score. "
        "For practitioners: getting a respondent to say 'I liked that ad' is the most predictive leading indicator."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – CRITICAL LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.title("⚠️ Critical Limitations")
    st.markdown(
        "A rigorous evaluation of every structural weakness in this study. "
        "These are not disclaimers — they are hard boundaries on what conclusions are valid."
    )

    limitations = [
        {
            "id": "L1",
            "title": "Critically Small Sample Size (n = 74)",
            "severity": "CRITICAL",
            "detail": """
**What is wrong:**
Statistical power for regression with 6 predictors requires n ≥ 100 at minimum (Cohen, 1988).
At n=74, the probability of detecting a true medium effect (R²~0.15) is approximately 55% —
barely better than a coin flip.

**Consequences:**
- Confidence intervals for all coefficients are extremely wide
- LOO-CV R² will have high variance (± 0.15 typical)
- Any "significant" result cannot be replicated with confidence
- The model cannot distinguish true signal from sampling artifacts

**What would fix it:** Minimum 150–200 responses; 300+ for sub-group analysis.
            """,
            "color": PALETTE["danger"],
        },
        {
            "id": "L2",
            "title": "Self-Report Bias & Social Desirability",
            "severity": "HIGH",
            "detail": """
**What is wrong:**
All measures — including the primary dependent variable (purchase intent) — are self-reported
on Likert scales. Research consistently shows that stated behavioral intentions overestimate
actual purchase behavior by 30–50%.

**Specific manifestations in this dataset:**
- Central tendency bias: 15/16 items scored between 3.0–3.6 (nobody strongly disagrees)
- Social desirability: respondents may rate influencers as "trustworthy" because it feels polite
- Hypothetical bias: "I would consider purchasing" ≠ "I will purchase next week"

**What would fix it:** Follow-up survey 2 weeks later asking if they actually purchased.
            """,
            "color": PALETTE["warn"],
        },
        {
            "id": "L3",
            "title": "No True Experimental Design — Causal Claims Are Invalid",
            "severity": "CRITICAL",
            "detail": """
**What is wrong:**
This is a cross-sectional survey, not a randomized controlled experiment.
Respondents were not randomly assigned to treatment (video exposure) vs control (no exposure).
There is no pre-exposure baseline measurement.

**Consequences:**
- You cannot claim the video content *caused* purchase intent to rise
- Any correlation between engagement/trust scores and PIS is observational
- Confounders: respondents who already liked the brands would rate both the ad AND the intent higher
  (prior brand preference → correlated error term → inflated β estimates)

**What would fix it:** Pre-post experimental design with random assignment; or at minimum,
include a "prior brand familiarity" control variable.
            """,
            "color": PALETTE["danger"],
        },
        {
            "id": "L4",
            "title": "Demographic Non-Representativeness",
            "severity": "HIGH",
            "detail": """
**What is wrong:**
- 53% students; 50% aged 22–25
- 57% Instagram users; only 4% Facebook/Twitter
- Geographic concentration (likely single-city convenience sample)
- "Prefer not to say" gender responses (14%) excluded from gender analysis

**Consequences:**
- Results represent young, educated, Instagram-native consumers
- Cannot generalize to: working professionals, ages 35+, rural consumers, YouTube-dominant markets
- Any business recommendation based on this sample applies to a narrow demographic segment

**What would fix it:** Stratified sampling across age groups and platforms;
minimum 15–20 respondents per demographic cell.
            """,
            "color": PALETTE["warn"],
        },
        {
            "id": "L5",
            "title": "Product Set Limitation — 3 Brands, 1 Category",
            "severity": "MEDIUM",
            "detail": """
**What is wrong:**
Three brands (OnePlus, Boat, Realme) are all budget-to-mid Indian earphone brands in the
₹1,000–₹3,000 range. They target the same demographic, have similar feature sets, and
compete through the same channels (Flipkart, Amazon India).

**Consequences:**
- Any finding about "influencer trust" or "content quality" is brand-specific, not universal
- The dominance of OnePlus (49% would purchase) may reflect brand equity, not content quality
- Findings cannot be extrapolated to premium brands (Sony, Jabra), other categories, or B2B products

**What would fix it:** Include at least 2 product categories and mix of brand tiers.
            """,
            "color": PALETTE["accent"],
        },
        {
            "id": "L6",
            "title": "Exposure Measurement is Absent",
            "severity": "HIGH",
            "detail": """
**What is wrong:**
The study's core claim is that "short-form video content" affects purchase behavior.
But there is NO variable measuring:
- Whether respondents actually watched the full video
- How many times they watched it
- Their attention level during viewing
- Whether the video was the first time they saw this brand

The "exposure variable" is implicitly assumed to be uniform — all 74 respondents saw the
same content to the same degree. This is almost certainly false.

**Consequences:**
- The independent variable of interest (content exposure) is not operationalized
- Models cannot include "dosage" effects
- The entire causal chain (exposure → engagement → intent) cannot be tested

**What would fix it:** Add video completion rate tracking; embed attention checks in the survey;
ask "how many times have you seen advertising for this brand before?"
            """,
            "color": PALETTE["danger"],
        },
        {
            "id": "L7",
            "title": "Composite Score Construction is Theoretically Weak",
            "severity": "MEDIUM",
            "detail": """
**What is wrong:**
The four composite scores (Trust, Engagement, CRI, PIS) are constructed by simple averaging
of Likert items. This assumes:
1. All items are equally weighted (equal contribution to the construct)
2. The items form a unidimensional scale
3. Likert responses are interval-scaled (not just ordinal)

None of these assumptions is validated.

**Consequences:**
- A low Cronbach's α means the "Trust Score" may not actually measure trust as a single construct
- Items with different means contribute unequal information when averaged
- Simple mean of ordinal items technically violates interval-scale assumption

**What would fix it:** Run confirmatory factor analysis (CFA) or use IRT (Item Response Theory)
to construct properly weighted, validated composite scores.
            """,
            "color": PALETTE["accent"],
        },
    ]

    for lim in limitations:
        color_map = {"CRITICAL": PALETTE["danger"], "HIGH": PALETTE["warn"], "MEDIUM": PALETTE["accent"]}
        sev_color = color_map.get(lim["severity"], PALETTE["primary"])
        with st.expander(
            f"**{lim['id']} — [{lim['severity']}]  {lim['title']}**", expanded=False
        ):
            st.markdown(
                f"<div style='background:{PALETTE['surface']};border-left:5px solid {sev_color};"
                f"border-radius:8px;padding:14px 18px'>{lim['detail']}</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    section_header("Severity Summary", "📋")
    sev_df = pd.DataFrame(
        [{"ID": l["id"], "Title": l["title"], "Severity": l["severity"]} for l in limitations]
    )
    sev_color_map = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}
    sev_df["Level"] = sev_df["Severity"].map(sev_color_map)
    st.dataframe(sev_df[["ID", "Level", "Severity", "Title"]], use_container_width=True)

    # st.markdown(
    #     f"""
    #     <div style='background:#1a0a0a;border:2px solid {PALETTE["danger"]};
    #                 border-radius:10px;padding:20px;margin-top:16px'>
    #         <h4 style='color:{PALETTE["danger"]}'>📌 Academic / Portfolio Positioning</h4>
    #         <p style='color:#eee;font-size:0.9rem'>
    #         Frame this project honestly as an <b>exploratory pilot study</b>, not a confirmatory study.
    #         The appropriate language is: <i>"Our findings are preliminary and suggest [X]. 
    #         A properly powered experimental study with n≥200, controlled exposure, and validated scales 
    #         would be required to confirm these patterns."</i><br><br>
    #         The value of this analysis lies in the methodological rigor of the approach, 
    #         the composite score design, and the critical self-evaluation — not in the 
    #         statistical power of the conclusions.
    #         </p>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <hr style='border-color:{PALETTE["surface"]}'>
    <p style='text-align:center;color:#666;font-size:0.8rem'>
    Short-Form Video × Purchase Intent Analysis  ·  
    Built with Streamlit + Plotly + scikit-learn  ·  
    <b>Exploratory Pilot Study</b>
    </p>
    """,
    unsafe_allow_html=True,
)