import streamlit as st
import pandas as pd
import joblib
import numpy as np
from ai_coach import generate_advice, generate_chat_reply

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindMeter — Student Burnout Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* ── App shell */
.stApp                         { background: #07090f; color: #dce3f0; }
[data-testid="stSidebar"]      { background: #0b0e17 !important; border-right: 1px solid #151b2a; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
.block-container               { padding-top: 2rem !important; }

/* ── Typography */
h1 { font-size: 1.9rem !important; font-weight: 800 !important; color: #eaf0ff !important; letter-spacing: -0.02em; margin-bottom: 0.1rem !important; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; color: #c8d5f0 !important; }
h3 { font-size: 1rem !important;   font-weight: 600 !important; color: #b0bcd4 !important; }
p  { color: #7a8aa0; }

/* ── Card */
.mm-card {
    background: #0d1121;
    border: 1px solid #161f35;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.9rem;
}
.mm-card-top {
    border-top: 2.5px solid;
    border-radius: 16px;
}

/* ── Score display */
.score-wrap {
    text-align: center;
    padding: 2rem 1rem;
}
.score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 5.5rem;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.03em;
}
.score-denom {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    color: #2e3a50;
    margin-top: -2px;
}
.score-bar-track {
    background: #101623;
    border-radius: 8px;
    height: 8px;
    margin: 1.2rem 0 0.8rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 8px;
}

/* ── Status badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.3rem 0.95rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Horizontal bars */
.hbar { margin-bottom: 0.45rem; }
.hbar-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 3px;
}
.hbar-label { font-size: 0.76rem; color: #8892a8; letter-spacing: 0.02em; }
.hbar-val   { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #3d4e66; }
.hbar-track { background: #101623; border-radius: 3px; height: 5px; overflow: hidden; }
.hbar-fill  { height: 100%; border-radius: 3px; }

/* ── Section heading */
.sh { font-weight: 700; font-size: 1rem; color: #c8d5f0; margin-bottom: 2px; }
.ss { font-size: 0.75rem; color: #3a4a60; margin-bottom: 0.9rem; }

/* ── AI advice block */
.ai-intro {
    font-size: 1rem !important;
    color: #c0ccde !important;
    font-weight: 500;
    line-height: 1.6;
    margin-bottom: 1.1rem;
}
.ai-section-title {
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5a7aa8;
    margin: 1.1rem 0 0.35rem;
}
.ai-wrap p  { font-size: 0.88rem !important; color: #8898b4 !important; line-height: 1.65; }
.ai-wrap ol { padding-left: 1.3rem; }
.ai-wrap li { font-size: 0.87rem; color: #8898b4; line-height: 1.6; margin-bottom: 0.3rem; }
.ai-wrap strong { color: #b8c8e0; }

/* ── Chat bubbles */
.chat-user {
    background: #0f1e38;
    border: 1px solid #1a3050;
    border-radius: 14px 14px 4px 14px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 0.4rem 2.5rem;
    font-size: 0.87rem;
    color: #a8bcd8;
    line-height: 1.55;
}
.chat-ai {
    background: #0d1220;
    border: 1px solid #161f32;
    border-radius: 14px 14px 14px 4px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 2.5rem 0.4rem 0;
    font-size: 0.87rem;
    color: #8898b4;
    line-height: 1.55;
}
.chat-name-user { font-size: 0.65rem; color: #2e4468; text-align: right; margin-bottom: 2px; letter-spacing: 0.08em; text-transform: uppercase; }
.chat-name-ai   { font-size: 0.65rem; color: #2e4468; margin-bottom: 2px; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── XAI bars */
.xai-row {
    background: #0d1121;
    border: 1px solid #161f35;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.35rem;
}
.xai-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.xai-name { font-size: 0.8rem; font-weight: 600; color: #b8c8e0; }
.xai-val  { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #2e3e56; }
.xai-foot { display: flex; justify-content: space-between; margin-top: 4px; }
.xai-dir  { font-size: 0.65rem; color: #2e3e56; }
.xai-wt   { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #2e3e56; }

/* ── Insight boxes */
.insight {
    border-left: 3px solid;
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: #7a8aa4;
    line-height: 1.6;
    background: #0b0f1c;
}
.insight strong { color: #b0c0d8; }

/* ── KPI tiles */
.kpi {
    background: #0d1121;
    border: 1px solid #161f35;
    border-top: 2.5px solid;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.kpi-num   { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 600; line-height: 1; }
.kpi-label { font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: #2e3e56; margin-top: 6px; }
.kpi-sub   { font-size: 0.76rem; color: #3a4e68; margin-top: 3px; }

/* ── Admin table */
.atbl { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.atbl th { background: #0b0f1c; color: #2e3e56; font-size: 0.65rem; letter-spacing: 0.1em;
           text-transform: uppercase; padding: 0.55rem 0.9rem; text-align: left; border-bottom: 1px solid #161f35; }
.atbl td { padding: 0.55rem 0.9rem; border-bottom: 1px solid #0f1520; color: #7a8aa4; }
.atbl tr:hover td { background: #0d1525; }

/* ── Sidebar misc */
.sbar-logo { font-size: 1.1rem; font-weight: 800; color: #eaf0ff; font-family: 'Outfit', sans-serif; }
.sbar-sub  { font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase; color: #1e2a3e; margin-bottom: 1rem; }
.sbar-sec  { font-size: 0.58rem; letter-spacing: 0.16em; text-transform: uppercase; color: #1e2a3e;
             margin-top: 1.3rem; margin-bottom: 0.45rem; border-bottom: 1px solid #131b2c; padding-bottom: 3px; }
.stSlider label { font-size: 0.73rem !important; color: #4a5e7a !important; letter-spacing: 0.02em; }

/* ── Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a3a8a, #3b28cc);
    color: #dce8ff;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    width: 100%;
    transition: opacity .2s;
    margin-top: 4px;
}
.stButton > button:hover { opacity: 0.85; border: none !important; }

/* ── Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0b0f1c; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"]      { border-radius: 8px; font-size: 0.82rem; font-weight: 600;
                                    color: #3a4e68; padding: 0.45rem 1rem; }
.stTabs [aria-selected="true"]    { background: #0f1e38 !important; color: #7ab4ff !important; }

/* ── Text input */
.stTextInput input, .stTextArea textarea {
    background: #0d1121 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 10px !important;
    color: #c8d5f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Spinner */
.stSpinner > div { border-top-color: #3b6bcc !important; }

/* ── Misc */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all():
    model    = joblib.load("model/burnout_model.pkl")
    features = joblib.load("model/features.pkl")
    meta     = joblib.load("model/feature_meta.pkl")
    return model, features, meta

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset/StressLevelDataset.csv")
    neg_f = {"anxiety_level":(0,21),"depression":(0,27),"headache":(0,5),
             "breathing_problem":(0,5),"noise_level":(0,5),"study_load":(0,5),
             "future_career_concerns":(0,5),"peer_pressure":(0,5),
             "extracurricular_activities":(0,5),"bullying":(0,5),
             "blood_pressure":(1,3),"mental_health_history":(0,1)}
    pos_f = {"self_esteem":(0,30),"sleep_quality":(0,5),"living_conditions":(0,5),
             "safety":(0,5),"basic_needs":(0,5),"academic_performance":(0,5),
             "teacher_student_relationship":(0,5),"social_support":(0,3)}
    neg = sum((df[c]-mn)/max(mx-mn,1) for c,(mn,mx) in neg_f.items())
    pos = sum(1-(df[c]-mn)/max(mx-mn,1) for c,(mn,mx) in pos_f.items())
    df["burnout_score"] = np.clip(
        (0.6*(neg/len(neg_f)) + 0.4*(pos/len(pos_f)))*10, 0, 10)
    return df

model, features, feature_meta = load_all()
df_data = load_dataset()


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
LABELS = {
    "anxiety_level":"Anxiety Level","self_esteem":"Self-Esteem",
    "mental_health_history":"Mental Health History","depression":"Depression Level",
    "headache":"Headache Frequency","blood_pressure":"Blood Pressure",
    "sleep_quality":"Sleep Quality","breathing_problem":"Breathing Problems",
    "noise_level":"Environmental Noise","living_conditions":"Living Conditions",
    "safety":"Sense of Safety","basic_needs":"Basic Needs Met",
    "academic_performance":"Academic Performance","study_load":"Study Load",
    "teacher_student_relationship":"Teacher–Student Relationship",
    "future_career_concerns":"Future Career Concerns","social_support":"Social Support",
    "peer_pressure":"Peer Pressure","extracurricular_activities":"Extracurricular Load",
    "bullying":"Bullying Exposure",
}

GROUPS = {
    "😰 Psychological": ["anxiety_level","depression","mental_health_history","self_esteem"],
    "🏥 Physical":      ["headache","blood_pressure","sleep_quality","breathing_problem"],
    "🏠 Environment":   ["noise_level","living_conditions","safety","basic_needs"],
    "📚 Academic":      ["academic_performance","study_load","teacher_student_relationship","future_career_concerns"],
    "👥 Social":        ["social_support","peer_pressure","extracurricular_activities","bullying"],
}

FEAT_IMP = {f: v for f, v in zip(features, model.named_steps["model"].feature_importances_)}

DOMAIN_COLORS = {
    "Psychological":"#8b5cf6","Physical":"#06b6d4",
    "Environment":"#10b981","Academic":"#f59e0b","Social":"#ec4899"
}

XAI_NOTES = {
    "sleep_quality":
        "<strong>Sleep quality is the #1 model factor</strong> at 24.3% importance. Poor sleep amplifies every other stressor simultaneously — it impairs emotional regulation, elevates cortisol, and degrades cognitive performance.",
    "depression":
        "<strong>Depression contributes 15.8%</strong> and correlates strongly with anxiety (+0.69) and bullying (+0.67), creating compounding effects when multiple factors are elevated at once.",
    "teacher_student_relationship":
        "<strong>Highest-importance protective factor</strong> at 14.8%. A supportive educational relationship buffers both academic pressure and future career anxiety simultaneously.",
    "bullying":
        "<strong>10.6% model importance</strong> with the highest cross-correlations: anxiety (+0.71), depression (+0.67), sleep (-0.70). Acts as a multiplier stressor across all domains.",
    "anxiety_level":
        "<strong>8.5% importance</strong>, highest raw burnout correlation. High anxiety combined with low self-esteem (r=-0.67) creates a self-reinforcing negative cycle.",
    "peer_pressure":
        "<strong>8.4% importance</strong>, strongly correlated with anxiety (+0.64) and study load (+0.54), suggesting it frequently compounds academic stressors.",
    "future_career_concerns":
        "<strong>Highest raw burnout correlation (r=0.85)</strong> despite 3.6% model weight. Strong linear signal especially at extremes — a clear anxiety trigger.",
    "self_esteem":
        "<strong>2nd-highest burnout correlation (r=0.84, inverted)</strong>. Acts as a broad resilience buffer — high self-esteem reduces sensitivity to every other stressor.",
}


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for key, default in [
    ("page",       "analyzer"),
    ("chat_history",  []),
    ("advice_cache",  None),
    ("last_score",    None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sbar-logo'>🧠 MindMeter</div>", unsafe_allow_html=True)
    st.markdown("<div class='sbar-sub'>Student Burnout Analyzer</div>", unsafe_allow_html=True)

    # ── Navigation
    for pid, icon, label in [
        ("analyzer","🎯","Burnout Analyzer"),
        ("xai",     "🔬","XAI Dashboard"),
        ("admin",   "⚙️","Admin Analytics"),
    ]:
        t = "primary" if st.session_state.page == pid else "secondary"
        if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True, type=t):
            st.session_state.page = pid
            st.rerun()

    st.markdown("---")

    # ── Gemini API Key
    st.markdown("<div class='sbar-sec'>🤖 AI Coach (Gemini)</div>", unsafe_allow_html=True)
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza…",
        help="Get your free key at aistudio.google.com",
        key="GEMINI_API_KEY",
    )
    if api_key:
        st.markdown("<div style='font-size:0.7rem;color:#1a6a3a;margin-top:-6px;'>✓ API key set</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:0.7rem;color:#3a4e68;margin-top:-6px;'>No key — AI Coach uses fallback tips</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Sliders
    st.markdown("<div style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#1e2a3e;margin-bottom:0.7rem;'>Student Profile</div>", unsafe_allow_html=True)

    user_input = {}
    for group_name, group_feats in GROUPS.items():
        st.markdown(f"<div class='sbar-sec'>{group_name}</div>", unsafe_allow_html=True)
        for f in group_feats:
            if f not in feature_meta:
                continue
            m = feature_meta[f]; mn, mx = m["min"], m["max"]
            hint = "↑ more stress" if m["direction"]=="negative" else ("↑ less stress" if m["direction"]=="positive" else "")
            lbl = LABELS.get(f,f) + (f"  ·  *{hint}*" if hint else "")
            user_input[f] = st.slider(lbl, int(mn), int(mx), int((mn+mx)//2), key=f"sl_{f}")


# ══════════════════════════════════════════════════════════════
#  COMPUTE PREDICTION  (runs on every interaction)
# ══════════════════════════════════════════════════════════════
input_df = pd.DataFrame([user_input])[features]
score = float(np.clip(model.predict(input_df)[0], 0, 10))

if   score < 3.0: STATUS,SC,SG,SE,SB,SD = ("Healthy",      "#22c55e","linear-gradient(90deg,#22c55e,#16a34a)","✅","rgba(34,197,94,.08)",  "Your indicators are within healthy limits. Keep up what's working.")
elif score < 5.5: STATUS,SC,SG,SE,SB,SD = ("Moderate Stress","#eab308","linear-gradient(90deg,#eab308,#ca8a04)","⚠️","rgba(234,179,8,.08)",  "Noticeable stress detected. Small targeted changes can meaningfully improve your score.")
elif score < 7.5: STATUS,SC,SG,SE,SB,SD = ("High Burnout Risk","#f97316","linear-gradient(90deg,#f97316,#ea580c)","🔴","rgba(249,115,22,.08)","Multiple compounding stressors active. Prioritise recovery and address the top contributors.")
else:             STATUS,SC,SG,SE,SB,SD = ("Critical Burnout","#ef4444","linear-gradient(90deg,#ef4444,#b91c1c)","🚨","rgba(239,68,68,.08)", "Severe indicators detected. Please reach out to a counsellor or trusted person immediately.")

# Per-factor breakdown
factor_contribs = []
for f in features:
    m = feature_meta[f]; mn_f,mx_f = m["min"],m["max"]
    norm = (user_input[f]-mn_f)/max(mx_f-mn_f,1)
    if   m["direction"]=="negative": contrib,bc = norm,      f"rgba(239,68,68,{.3+.6*norm:.2f})"
    elif m["direction"]=="positive": contrib,bc = 1-norm,    f"rgba(34,197,94,{.3+.6*(1-norm):.2f})"
    else:                            contrib,bc = norm,      "rgba(100,116,139,.5)"
    factor_contribs.append({
        "feat":f,"name":LABELS.get(f,f),"value":user_input[f],
        "norm":norm,"contrib":contrib,"color":bc,
        "direction":m["direction"],"imp":FEAT_IMP.get(f,0),
        "range":f"{int(mn_f)}–{int(mx_f)}"
    })

fc_by_contrib = sorted(factor_contribs, key=lambda x: x["contrib"], reverse=True)
fc_by_impact  = sorted(factor_contribs, key=lambda x: x["imp"]*x["contrib"], reverse=True)

# Top stress / protective for AI coach
top_stress     = [f["feat"] for f in fc_by_impact[:4]]
top_protective = [f["feat"] for f in sorted(factor_contribs, key=lambda x: x["imp"]*(1-x["contrib"]), reverse=True)[:2]]

# Domain scores
domain_map = {
    "Psychological":["anxiety_level","depression","mental_health_history","self_esteem"],
    "Physical":     ["headache","blood_pressure","sleep_quality","breathing_problem"],
    "Environment":  ["noise_level","living_conditions","safety","basic_needs"],
    "Academic":     ["academic_performance","study_load","teacher_student_relationship","future_career_concerns"],
    "Social":       ["social_support","peer_pressure","extracurricular_activities","bullying"],
}
domain_scores = {}
for domain, d_feats in domain_map.items():
    vals = []
    for f in d_feats:
        m2=feature_meta[f]; rn=max(m2["max"]-m2["min"],1)
        n=(user_input.get(f,m2["min"])-m2["min"])/rn
        if m2["direction"]=="positive": n=1-n
        vals.append(n)
    domain_scores[domain] = round(np.mean(vals)*10,2)


# ══════════════════════════════════════════════════════════════
#  HELPER RENDERERS
# ══════════════════════════════════════════════════════════════
def score_card():
    st.markdown(f"""
    <div class='mm-card mm-card-top' style='border-top-color:{SC};'>
        <div class='score-wrap'>
            <div class='score-num' style='color:{SC};'>{score:.1f}</div>
            <div class='score-denom'>/ 10</div>
            <div style='margin-top:.9rem;'>
                <span class='badge' style='background:{SB};color:{SC};border:1px solid {SC}44;'>
                    {SE} {STATUS}
                </span>
            </div>
            <div class='score-bar-track'>
                <div class='score-bar-fill' style='width:{score*10:.1f}%;background:{SG};'></div>
            </div>
            <p style='font-size:.78rem;color:#3a4e68;margin:0;line-height:1.55;'>{SD}</p>
        </div>
    </div>""", unsafe_allow_html=True)


def domain_bars():
    for domain, ds in domain_scores.items():
        col = DOMAIN_COLORS[domain]
        st.markdown(f"""
        <div class='hbar'>
            <div class='hbar-row'>
                <span class='hbar-label'>{domain}</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:{col};'>{ds:.1f}</span>
            </div>
            <div class='hbar-track'>
                <div class='hbar-fill' style='width:{ds*10:.1f}%;background:{col};opacity:.8;'></div>
            </div>
        </div>""", unsafe_allow_html=True)


def factor_bars(items, limit=14):
    for fc in items[:limit]:
        bar_pct = fc["contrib"]*100
        st.markdown(f"""
        <div class='mm-card' style='padding:.72rem 1rem;margin-bottom:.3rem;'>
            <div class='hbar-row'>
                <span class='hbar-label' style='color:#b0bcd4;font-weight:600;'>{fc['name']}</span>
                <span class='hbar-val'>{fc['value']}/{fc['range'].split('–')[1]}</span>
            </div>
            <div class='hbar-track' style='height:6px;'>
                <div class='hbar-fill' style='width:{bar_pct:.1f}%;background:{fc["color"]};'></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ANALYZER
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "analyzer":

    st.markdown("# 🧠 Burnout Analyzer")
    st.markdown("<p style='margin-top:-.3rem;margin-bottom:1.6rem;color:#2e3e56;'>Real-time assessment powered by Gradient Boosting · Adjust sliders in the sidebar</p>", unsafe_allow_html=True)

    tab_result, tab_chat = st.tabs(["  📊 My Results  ", "  💬 AI Coach Chat  "])

    # ── Results Tab
    with tab_result:
        c1, c2, c3 = st.columns([1.05, 1.15, 1.1], gap="large")

        with c1:
            score_card()
            st.markdown("<div class='sh' style='margin-top:.2rem;'>Domain Scores</div>", unsafe_allow_html=True)
            st.markdown("<div class='ss'>Stress load per life domain</div>", unsafe_allow_html=True)
            domain_bars()

        with c2:
            st.markdown("<div class='sh'>Factor Breakdown</div>", unsafe_allow_html=True)
            st.markdown("<div class='ss'>Sorted by burnout contribution · Red = stress driver, Teal = protective</div>", unsafe_allow_html=True)
            factor_bars(fc_by_contrib)

        with c3:
            st.markdown("<div class='sh'>🤖 AI Coach Insights</div>", unsafe_allow_html=True)
            st.markdown("<div class='ss'>Personalised guidance based on your profile</div>", unsafe_allow_html=True)

            # Cache advice so it doesn't regenerate on every slider move
            if st.button("✨ Generate AI Advice", key="gen_advice", use_container_width=True):
                st.session_state.advice_cache = None
                st.session_state.last_score   = None

            needs_new = (
                st.session_state.advice_cache is None or
                st.session_state.last_score != round(score, 1)
            )

            if needs_new:
                with st.spinner("MindMeter AI is analysing your profile…"):
                    advice = generate_advice(
                            score,
                            top_stress,
                            top_protective,
                            user_input,
                            feature_meta
                    )
                st.session_state.advice_cache = advice
                st.session_state.last_score   = round(score, 1)

            st.markdown(
                f"<div class='mm-card ai-wrap'>{st.session_state.advice_cache}</div>",
                unsafe_allow_html=True
            )

    # ── Chat Tab
    with tab_chat:
        st.markdown("<div class='sh' style='margin-bottom:.2rem;'>💬 Chat with your AI Wellness Coach</div>", unsafe_allow_html=True)
        st.markdown("<div class='ss' style='margin-bottom:1rem;'>Ask anything about your results, stress management, or study habits. Powered by Gemini.</div>", unsafe_allow_html=True)

        if not api_key:
            st.markdown("""
            <div class='insight' style='border-left-color:#1a3a8a;'>
                <strong>API key required</strong><br>
                Enter your Gemini API key in the sidebar to unlock AI chat.
                Get a free key at <code>aistudio.google.com</code>.
            </div>""", unsafe_allow_html=True)
        else:
            # Chat window
            chat_container = st.container()
            with chat_container:
                if not st.session_state.chat_history:
                    st.markdown("""
                    <div class='chat-ai'>
                        <div class='chat-name-ai'>🧠 MindMeter AI</div>
                        Hi! I'm your MindMeter wellness coach. I've reviewed your burnout profile
                        and I'm here to help. Ask me anything — about your results, stress strategies,
                        sleep, academic pressure, or just how you're feeling right now.
                    </div>""", unsafe_allow_html=True)

                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class='chat-user'>
                            <div class='chat-name-user'>You</div>
                            {msg['parts'][0]}
                        </div>""", unsafe_allow_html=True)
                    elif msg["role"] == "model":
                        st.markdown(f"""
                        <div class='chat-ai'>
                            <div class='chat-name-ai'>🧠 MindMeter AI</div>
                            {msg['parts'][0]}
                        </div>""", unsafe_allow_html=True)

            # Input row
            col_input, col_send = st.columns([5, 1])
            with col_input:
                user_msg = st.text_input(
                    "Message",
                    placeholder="Ask about your results, stress tips, study habits…",
                    label_visibility="collapsed",
                    key="chat_input",
                )
            with col_send:
                send_clicked = st.button("Send →", key="chat_send", use_container_width=True)

            col_clear, _ = st.columns([1, 5])
            with col_clear:
                if st.button("Clear chat", key="chat_clear"):
                    st.session_state.chat_history = []
                    st.rerun()

            if send_clicked and user_msg.strip():
                # Append user message
                st.session_state.chat_history.append(
                    {"role": "user", "parts": [user_msg.strip()]}
                )

                with st.spinner("Coach is typing…"):
                    reply = generate_chat_reply(
                        user_msg.strip(), score, user_input,
                        feature_meta, st.session_state.chat_history
                    )

                st.session_state.chat_history.append(
                    {"role": "model", "parts": [reply]}
                )
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE: XAI DASHBOARD
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "xai":

    st.markdown("# 🔬 Explainable AI Dashboard")
    st.markdown("<p style='margin-top:-.3rem;margin-bottom:1.6rem;color:#2e3e56;'>How and why the model makes its predictions — for your profile and globally</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  📌 My Prediction  ", "  🌍 Global Importance  ", "  📉 Partial Dependence  "])

    # ─── Tab 1: Local Explanation
    with tab1:
        st.markdown(f"""
        <div class='mm-card mm-card-top' style='border-top-color:{SC};
             display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-bottom:1.5rem;'>
            <div style='text-align:center;min-width:85px;'>
                <div class='score-num' style='color:{SC};font-size:3rem;'>{score:.1f}</div>
                <div class='score-denom' style='font-size:.8rem;'>/ 10</div>
                <span class='badge' style='background:{SB};color:{SC};border:1px solid {SC}44;
                    display:inline-block;margin-top:6px;font-size:.62rem;'>{STATUS}</span>
            </div>
            <div style='flex:1;min-width:200px;'>
                <div style='font-weight:700;color:#c8d5f0;font-size:.95rem;margin-bottom:4px;'>How this score was computed</div>
                <p style='font-size:.82rem;color:#3a4e68;line-height:1.6;margin:0;'>
                The burnout score blends <strong style='color:#7a9ac8;'>negative drivers</strong>
                (anxiety, depression, bullying…) at 60% weight with
                <strong style='color:#7a9ac8;'>protective factor deficits</strong>
                (low sleep, low self-esteem…) at 40% weight.
                Each feature is normalised to its real dataset range, then a
                <strong style='color:#7a9ac8;'>Gradient Boosting regressor</strong>
                applies learned non-linear interactions from 1,100 student records (R²=0.9945).
                </p>
            </div>
        </div>""", unsafe_allow_html=True)

        xai_c1, xai_c2 = st.columns([1.05, 1], gap="large")

        with xai_c1:
            st.markdown("<div class='sh'>Feature Contributions — Your Profile</div>", unsafe_allow_html=True)
            st.markdown("<div class='ss'>Sorted by weighted impact (contribution × model importance). Direction and weight shown per row.</div>", unsafe_allow_html=True)

            for fc in fc_by_impact:
                pct = fc["contrib"]*100; imp_pct = fc["imp"]*100
                dir_txt = "▲ stress driver" if fc["direction"]=="negative" else ("▼ protective" if fc["direction"]=="positive" else "–")
                st.markdown(f"""
                <div class='xai-row'>
                    <div class='xai-meta'>
                        <span class='xai-name'>{fc['name']}</span>
                        <span class='xai-val'>{fc['value']}&thinsp;/&thinsp;{fc['range'].split('–')[1]}</span>
                    </div>
                    <div class='hbar-track' style='height:6px;'>
                        <div class='hbar-fill' style='width:{pct:.1f}%;background:{fc["color"]};'></div>
                    </div>
                    <div class='xai-foot'>
                        <span class='xai-dir'>{dir_txt}</span>
                        <span class='xai-wt'>model weight {imp_pct:.1f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        with xai_c2:
            st.markdown("<div class='sh'>Factor Deep-Dives</div>", unsafe_allow_html=True)
            st.markdown("<div class='ss'>Evidence-based explanations for the top 8 model factors.</div>", unsafe_allow_html=True)

            top8 = sorted(FEAT_IMP.items(), key=lambda x:x[1], reverse=True)[:8]
            for feat, imp in top8:
                note = XAI_NOTES.get(feat, f"Contributes {imp*100:.1f}% to prediction weight.")
                val  = user_input.get(feat, "—")
                m2   = feature_meta.get(feat, {})
                dir_badge = "▲ stress driver" if m2.get("direction")=="negative" else "▼ protective factor"
                st.markdown(f"""
                <div class='xai-row' style='margin-bottom:.4rem;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>
                        <span style='font-weight:700;font-size:.82rem;color:#c0ccde;'>{LABELS.get(feat,feat)}</span>
                        <span style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:#3b82f6;'>Your: {val}</span>
                    </div>
                    <div style='font-family:JetBrains Mono,monospace;font-size:.68rem;color:#2e3e56;margin-bottom:5px;'>
                        weight {imp*100:.1f}% &nbsp;·&nbsp;
                        <span style='background:#0b0f1c;padding:1px 5px;border-radius:4px;'>{dir_badge}</span>
                    </div>
                    <div style='font-size:.79rem;color:#5a6e8a;line-height:1.58;'>{note}</div>
                </div>""", unsafe_allow_html=True)

    # ─── Tab 2: Global Importance
    with tab2:
        st.markdown("<div class='sh'>Global Feature Importance</div>", unsafe_allow_html=True)
        st.markdown("<div class='ss'>Contribution of each feature to predictions across ALL students (not just your profile)</div>", unsafe_allow_html=True)

        imp_df = (
            pd.DataFrame({"Feature":[LABELS.get(f,f) for f in features],
                          "Importance (%)":[FEAT_IMP[f]*100 for f in features]})
            .sort_values("Importance (%)", ascending=False).set_index("Feature")
        )

        gc1, gc2 = st.columns([1.5,1], gap="large")
        with gc1:
            st.bar_chart(imp_df, color="#3b6bcc", height=460)
        with gc2:
            st.markdown("<div class='sh' style='font-size:.9rem;'>Importance Table</div>", unsafe_allow_html=True)
            rows = ""
            for i,(fname,row) in enumerate(imp_df.iterrows()):
                fk = next((f for f in features if LABELS.get(f,f)==fname), None)
                dir_icon = "🔴" if feature_meta.get(fk,{}).get("direction")=="negative" else ("🟢" if feature_meta.get(fk,{}).get("direction")=="positive" else "⚪")
                rows += f"<tr><td style='color:#2e3e56;font-family:JetBrains Mono,monospace;'>#{i+1}</td><td>{fname}</td><td style='font-family:JetBrains Mono,monospace;color:#3b82f6;'>{row['Importance (%)']:.2f}%</td><td>{dir_icon}</td></tr>"
            st.markdown(f"<table class='atbl'><thead><tr><th>#</th><th>Feature</th><th>Weight</th><th>Dir</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

        st.markdown("<br><div class='sh'>Key Model Insights</div>", unsafe_allow_html=True)
        ic1,ic2,ic3 = st.columns(3)
        with ic1:
            st.markdown("<div class='insight' style='border-left-color:#8b5cf6;'><strong>Sleep Dominates</strong><br>Sleep quality alone = 24.3% of model weight — more than depression + anxiety combined. Sleep interventions yield the highest burnout reduction per unit of change.</div>", unsafe_allow_html=True)
        with ic2:
            st.markdown("<div class='insight' style='border-left-color:#f59e0b;'><strong>Institutional Factors Matter</strong><br>Teacher–student relationship (14.8%) outranks self-esteem and social support. Educational environment quality is a top-3 predictor, not a peripheral factor.</div>", unsafe_allow_html=True)
        with ic3:
            st.markdown("<div class='insight' style='border-left-color:#22c55e;'><strong>Compounding Effects</strong><br>Anxiety, depression, bullying, and peer pressure are intercorrelated (r=0.64–0.71). Students with multiple risk factors face non-linear burnout escalation.</div>", unsafe_allow_html=True)

    # ─── Tab 3: Partial Dependence
    with tab3:
        st.markdown("<div class='sh'>Partial Dependence Plots</div>", unsafe_allow_html=True)
        st.markdown("<div class='ss'>How changing one factor alone shifts the predicted burnout score — all other features held at median values</div>", unsafe_allow_html=True)

        top6 = [f for f,_ in sorted(FEAT_IMP.items(), key=lambda x:x[1], reverse=True)[:6]]
        median_row = {f: float(df_data[f].median()) for f in features}

        pd_c1, pd_c2 = st.columns(2, gap="large")
        for i, feat in enumerate(top6):
            m = feature_meta[feat]; mn_f,mx_f = m["min"],m["max"]
            steps = np.linspace(mn_f, mx_f, 25)
            s_pd = []
            for v in steps:
                r = median_row.copy(); r[feat]=v
                s_pd.append(float(np.clip(model.predict(pd.DataFrame([r])[features])[0],0,10)))
            pd_df = pd.DataFrame({"Burnout Score":s_pd}, index=[round(s,1) for s in steps])
            dir_note = "🔴 Higher → more burnout" if m["direction"]=="negative" else "🟢 Higher → less burnout"
            col = pd_c1 if i%2==0 else pd_c2
            with col:
                st.markdown(f"<div style='font-weight:700;color:#c8d5f0;font-size:.88rem;'>{LABELS.get(feat,feat)} <span style='font-family:JetBrains Mono,monospace;font-size:.7rem;color:#2e3e56;'>· weight {FEAT_IMP[feat]*100:.1f}%</span></div>", unsafe_allow_html=True)
                st.caption(dir_note)
                st.line_chart(pd_df, color="#6366f1", height=195)


# ══════════════════════════════════════════════════════════════
#  PAGE: ADMIN ANALYTICS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "admin":

    st.markdown("# ⚙️ Admin Analytics Panel")
    st.markdown("<p style='margin-top:-.3rem;margin-bottom:1.6rem;color:#2e3e56;'>Dataset statistics, model performance metrics, and population-level burnout insights</p>", unsafe_allow_html=True)

    X_data    = df_data[features]
    all_preds = np.clip(model.predict(X_data), 0, 10)

    # ── KPIs
    kc = st.columns(5)
    kpis = [
        ("1,100",  "Total Students",   "#3b82f6", "Dataset size"),
        (f"{all_preds.mean():.2f}", "Avg Burnout", "#f97316", "Population mean /10"),
        (f"{(all_preds>=7.5).sum()}", "Critical Cases", "#ef4444", f"{(all_preds>=7.5).mean()*100:.0f}% of cohort"),
        (f"{(all_preds<3.0).sum()}",  "Healthy", "#22c55e", f"{(all_preds<3.0).mean()*100:.0f}% of cohort"),
        ("99.45%", "Model R²", "#8b5cf6", "Goodness of fit"),
    ]
    for col,(val,label,color,sub) in zip(kc,kpis):
        with col:
            st.markdown(f"""
            <div class='kpi' style='border-top-color:{color};'>
                <div class='kpi-num' style='color:{color};'>{val}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribution + Risk Tiers
    dc1, dc2 = st.columns([1.6,1], gap="large")
    with dc1:
        st.markdown("<div class='sh'>Score Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='ss'>Predicted burnout scores across all 1,100 students</div>", unsafe_allow_html=True)
        bins = np.arange(0, 10.6, 0.5)
        hv, he = np.histogram(all_preds, bins=bins)
        st.bar_chart(pd.DataFrame({"Students":hv}, index=[f"{e:.1f}" for e in he[:-1]]), color="#3b6bcc", height=260)

    with dc2:
        st.markdown("<div class='sh'>Risk Tier Breakdown</div>", unsafe_allow_html=True)
        st.markdown("<div class='ss'>Distribution across burnout severity levels</div>", unsafe_allow_html=True)
        total = len(all_preds)
        for label,mask,color in [
            ("🟢 Healthy",         all_preds<3.0,                  "#22c55e"),
            ("🟡 Moderate Stress", (all_preds>=3.0)&(all_preds<5.5),"#eab308"),
            ("🟠 High Risk",       (all_preds>=5.5)&(all_preds<7.5),"#f97316"),
            ("🔴 Critical",        all_preds>=7.5,                  "#ef4444"),
        ]:
            count=mask.sum(); pct=count/total*100
            st.markdown(f"""
            <div class='mm-card' style='padding:.8rem 1rem;margin-bottom:.35rem;
                 display:flex;align-items:center;gap:.8rem;'>
                <div style='font-weight:700;color:{color};font-size:.8rem;min-width:140px;'>{label}</div>
                <div style='flex:1;background:#0a0d14;border-radius:4px;height:7px;overflow:hidden;'>
                    <div style='height:100%;width:{pct:.1f}%;background:{color};opacity:.7;border-radius:4px;'></div>
                </div>
                <div style='font-family:JetBrains Mono,monospace;font-size:.78rem;color:{color};min-width:72px;text-align:right;'>
                    {count} <span style='color:#2e3e56;'>({pct:.0f}%)</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='insight' style='border-left-color:#ef4444;margin-top:.6rem;'>
            <strong>{(all_preds>=7.5).sum()} students ({(all_preds>=7.5).mean()*100:.0f}%)</strong>
            are in Critical Burnout range and may benefit from immediate counselling referral.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Stats + Correlation
    fc1, fc2 = st.columns([1.4,1], gap="large")
    corrs = X_data.corrwith(df_data["burnout_score"]).abs()

    with fc1:
        st.markdown("<div class='sh'>Feature Statistics</div>", unsafe_allow_html=True)
        st.markdown("<div class='ss'>Mean, std, range, |correlation with burnout|, direction, and model weight for all 20 features</div>", unsafe_allow_html=True)

        stat_list = sorted([{
            "f":f,"name":LABELS.get(f,f),
            "mean":f"{X_data[f].mean():.2f}","std":f"{X_data[f].std():.2f}",
            "range":f"{int(X_data[f].min())}–{int(X_data[f].max())}",
            "corr":corrs[f],"direction":feature_meta[f]["direction"],
            "imp":FEAT_IMP[f]*100,
        } for f in features], key=lambda x:x["corr"], reverse=True)

        rows=""
        for r in stat_list:
            di = "🔴" if r["direction"]=="negative" else ("🟢" if r["direction"]=="positive" else "⚪")
            rows += f"<tr><td>{r['name']}</td><td style='font-family:JetBrains Mono,monospace;'>{r['mean']}</td><td style='font-family:JetBrains Mono,monospace;'>{r['std']}</td><td style='font-family:JetBrains Mono,monospace;'>{r['range']}</td><td style='font-family:JetBrains Mono,monospace;color:#3b82f6;'>{r['corr']:.3f}</td><td>{di}</td><td style='font-family:JetBrains Mono,monospace;color:#8b5cf6;'>{r['imp']:.1f}%</td></tr>"

        st.markdown(f"""
        <div style='max-height:420px;overflow-y:auto;border:1px solid #161f35;border-radius:12px;'>
        <table class='atbl'><thead><tr>
            <th>Feature</th><th>Mean</th><th>Std</th><th>Range</th><th>|Corr|</th><th>Dir</th><th>Wt</th>
        </tr></thead><tbody>{rows}</tbody></table></div>""", unsafe_allow_html=True)

    with fc2:
        st.markdown("<div class='sh'>Top Burnout Correlations</div>", unsafe_allow_html=True)
        st.markdown("<div class='ss'>|r| with burnout score — higher = stronger linear relationship</div>", unsafe_allow_html=True)

        top_c = corrs.sort_values(ascending=False).head(10)
        st.bar_chart(
            pd.DataFrame({"Correlation":top_c.values}, index=[LABELS.get(f,f) for f in top_c.index]),
            color="#8b5cf6", height=280
        )

        st.markdown("<div class='sh' style='margin-top:.8rem;'>Model Performance</div>", unsafe_allow_html=True)
        for metric,val,color,desc in [
            ("R² Score",      "0.9945",        "#22c55e", "Variance explained"),
            ("MAE",           "0.096",          "#3b82f6", "Mean absolute error"),
            ("Algorithm",     "Gradient Boost", "#8b5cf6", "GradientBoostingRegressor"),
            ("Estimators",    "500 trees",      "#f59e0b", "Ensemble size"),
            ("Train / Test",  "80 / 20",        "#ec4899", "1,100 student records"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                 background:#0d1121;border:1px solid #161f35;border-radius:8px;
                 padding:.55rem .9rem;margin-bottom:.3rem;'>
                <div>
                    <div style='font-size:.76rem;font-weight:600;color:#6a7a94;'>{metric}</div>
                    <div style='font-size:.66rem;color:#2e3e56;'>{desc}</div>
                </div>
                <div style='font-family:JetBrains Mono,monospace;font-weight:600;color:{color};font-size:.9rem;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;color:#151e2e;font-size:.7rem;margin-top:2.5rem;
         padding-top:1rem;border-top:1px solid #111827;'>
        MindMeter Admin · Gradient Boosting Regressor · 1,100 students · 20 biopsychosocial features · R²=0.9945
    </div>""", unsafe_allow_html=True)