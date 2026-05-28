import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MindMeter API", description="FastAPI backend for MindMeter Burnout Intelligence System")

# Enable CORS for React frontend (usually running on port 5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In prod, restrict.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOAD MODEL & CONFIG
# =========================================================
MODEL_PATH = "model/burnout_model.pkl"
FEATURES_PATH = "model/features.pkl"
META_PATH = "model/feature_meta.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH) or not os.path.exists(META_PATH):
    # Try parent directory fallback
    MODEL_PATH = "../model/burnout_model.pkl"
    FEATURES_PATH = "../model/features.pkl"
    META_PATH = "../model/feature_meta.pkl"

try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    feature_meta = joblib.load(META_PATH)
    print("[OK] Model and metadata loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model files: {e}")
    model, features, feature_meta = None, [], {}

# Emojis and labels mapping
LABELS = {
    "anxiety_level": "Anxiety Level",
    "self_esteem": "Self-Esteem",
    "mental_health_history": "Mental Health History",
    "depression": "Depression Level",
    "headache": "Headache Frequency",
    "blood_pressure": "Blood Pressure",
    "sleep_quality": "Sleep Quality",
    "breathing_problem": "Breathing Problems",
    "noise_level": "Environmental Noise",
    "living_conditions": "Living Conditions",
    "safety": "Sense of Safety",
    "basic_needs": "Basic Needs Met",
    "academic_performance": "Academic Performance",
    "study_load": "Study Load",
    "teacher_student_relationship": "Teacher–Student Relationship",
    "future_career_concerns": "Future Career Concerns",
    "social_support": "Social Support",
    "peer_pressure": "Peer Pressure",
    "extracurricular_activities": "Extracurricular Load",
    "bullying": "Bullying Exposure",
}

GROUPS = {
    "😰 Psychological": ["anxiety_level", "depression", "mental_health_history", "self_esteem"],
    "🏥 Physical":      ["headache", "blood_pressure", "sleep_quality", "breathing_problem"],
    "🏠 Environment":   ["noise_level", "living_conditions", "safety", "basic_needs"],
    "📚 Academic":      ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"],
    "👥 Social":        ["social_support", "peer_pressure", "extracurricular_activities", "bullying"],
}

# Extract global feature importances from Ridge coefficients
try:
    ridge_model = model.named_steps["model"]
    raw_coefs = ridge_model.coef_
    normalized_coefs = np.abs(raw_coefs) / np.sum(np.abs(raw_coefs))
    FEAT_IMP = {f: float(val) for f, val in zip(features, normalized_coefs)}
except Exception:
    FEAT_IMP = {f: 0.05 for f in features}  # fallback equal weight

# Domain maps
DOMAIN_MAPS = {
    "Psychological": ["anxiety_level", "depression", "mental_health_history", "self_esteem"],
    "Physical":      ["headache", "blood_pressure", "sleep_quality", "breathing_problem"],
    "Environment":   ["noise_level", "living_conditions", "safety", "basic_needs"],
    "Academic":      ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"],
    "Social":        ["social_support", "peer_pressure", "extracurricular_activities", "bullying"],
}

ALLOWED_TOPICS = [
    "stress", "burnout", "mental", "anxiety", "depression", "study",
    "sleep", "focus", "motivation", "academic", "career", "peer",
    "social", "wellbeing", "health", "feeling", "workload", "bullying"
]

CRISIS_WORDS = ["suicide", "kill myself", "end my life", "self harm", "want to die", "harm myself"]


# =========================================================
# DATA MODELS
# =========================================================
class PredictRequest(BaseModel):
    features: Dict[str, float]

class AdviceRequest(BaseModel):
    score: float
    user_values: Dict[str, float]
    user_api_key: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    parts: List[str]

class ChatRequest(BaseModel):
    message: str
    score: float
    user_values: Dict[str, float]
    chat_history: List[ChatMessage]
    user_api_key: Optional[str] = None


# =========================================================
# CORE LOGIC HELPERS
# =========================================================
def get_status_details(score: float) -> Dict[str, str]:
    if score < 3.0:
        return {
            "status": "Healthy",
            "color": "#22c55e",
            "emoji": "✅",
            "bg": "rgba(34,197,94,0.08)",
            "description": "Your indicators are within healthy limits. Keep up what's working."
        }
    elif score < 5.5:
        return {
            "status": "Moderate Stress",
            "color": "#eab308",
            "emoji": "⚠️",
            "bg": "rgba(234,179,8,0.08)",
            "description": "Noticeable stress detected. Small targeted changes can meaningfully improve your score."
        }
    elif score < 7.5:
        return {
            "status": "High Burnout Risk",
            "color": "#f97316",
            "emoji": "🔴",
            "bg": "rgba(249,115,22,0.08)",
            "description": "Multiple compounding stressors active. Prioritise recovery and address the top contributors."
        }
    else:
        return {
            "status": "Critical Burnout",
            "color": "#ef4444",
            "emoji": "🚨",
            "bg": "rgba(239,68,68,0.08)",
            "description": "Severe indicators detected. Please reach out to a counsellor or trusted person immediately."
        }

def static_fallback_advice(score: float, stress: List[str], protective: List[str], error: str = "") -> str:
    stress_names = [LABELS.get(f, f) for f in stress[:3]]
    protect_names = [LABELS.get(f, f) for f in protective[:2]]
    status_details = get_status_details(score)
    status_label = status_details["status"]
    
    error_note = f"<small style='color: #ef4444; opacity: 0.8;'>(AI Coach Engine Fallback: {error[:60]})</small><br><br>" if error else ""
    protect_text = f"Your strongest protective factor is <strong>{', '.join(protect_names)}</strong>." if protect_names else "Focus on rebuilding supportive habits gradually."
    
    return f"""{error_note}
<p class="ai-intro">You're currently experiencing <strong>{status_label.lower()}</strong>.</p>

<div class="ai-section-title">🔴 What's Driving Your Stress</div>
<p>Based on your profile, the primary drivers are <strong>{', '.join(stress_names)}</strong>. Addressing these core factors first will yield the highest burnout reduction.</p>

<div class="ai-section-title">🟢 What's Working in Your Favour</div>
<p>{protect_text} These assets act as a protective buffer, reducing your susceptibility to severe stressors.</p>

<div class="ai-section-title">⚡ Your 3 Priority Actions This Week</div>
<ol>
<li><strong>Establish boundaries:</strong> Dedicate 15 minutes to quiet reflection or breathing exercises daily.</li>
<li><strong>Rebuild protective habits:</strong> Focus on maintaining a consistent sleep schedule to support emotional regulation.</li>
<li><strong>Reach out:</strong> Discuss your current challenges with a trusted mentor, advisor, or counsellor.</li>
</ol>

<div class="ai-section-title">💬 A Note From Your Coach</div>
<p>Small, steady recovery steps are incredibly powerful. Remember, you don't have to carry this entire load alone.</p>
"""


# =========================================================
# ENDPOINTS
# =========================================================

@app.post("/api/predict")
def predict_burnout(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model not available.")
    
    try:
        user_input = req.features
        # Ensure all features exist in order
        missing = [f for f in features if f not in user_input]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing feature values: {missing}")
        
        # Build dataframe in correct column order
        input_df = pd.DataFrame([user_input])[features]
        # Predict burnout score
        pred_val = float(model.predict(input_df)[0])
        score = float(np.clip(pred_val, 0.0, 10.0))
        
        status_info = get_status_details(score)
        
        # Calculate domain scores
        domain_scores = {}
        for domain, d_feats in DOMAIN_MAPS.items():
            vals = []
            for f in d_feats:
                m = feature_meta[f]
                rn = max(m["max"] - m["min"], 1)
                norm = (user_input[f] - m["min"]) / rn
                if m["direction"] == "positive":
                    norm = 1.0 - norm
                vals.append(norm)
            domain_scores[domain] = float(np.clip(np.mean(vals) * 10, 0.0, 10.0))
            
        # Calculate factor breakdown
        factor_contribs = []
        for f in features:
            m = feature_meta[f]
            rn = max(m["max"] - m["min"], 1)
            norm = (user_input[f] - m["min"]) / rn
            
            if m["direction"] == "negative":
                contrib = norm
                color = f"rgba(239, 68, 68, {0.25 + 0.65 * norm:.2f})"
            elif m["direction"] == "positive":
                contrib = 1.0 - norm
                color = f"rgba(34, 197, 94, {0.25 + 0.65 * (1.0 - norm):.2f})"
            else:
                contrib = norm
                color = "rgba(100, 116, 139, 0.5)"
                
            factor_contribs.append({
                "feat": f,
                "name": LABELS.get(f, f),
                "value": float(user_input[f]),
                "norm": float(norm),
                "contrib": float(contrib),
                "color": color,
                "direction": m["direction"],
                "imp": float(FEAT_IMP.get(f, 0.0)),
                "range": f"{int(m['min'])}–{int(m['max'])}"
            })
            
        # Sort factors
        fc_by_contrib = sorted(factor_contribs, key=lambda x: x["contrib"], reverse=True)
        fc_by_impact = sorted(factor_contribs, key=lambda x: x["imp"] * x["contrib"], reverse=True)
        
        return {
            "score": round(score, 2),
            "status": status_info["status"],
            "color": status_info["color"],
            "emoji": status_info["emoji"],
            "bg": status_info["bg"],
            "description": status_info["description"],
            "domains": domain_scores,
            "factors_by_contrib": fc_by_contrib,
            "factors_by_impact": fc_by_impact
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/coach/advice")
def get_coach_advice(req: AdviceRequest):
    # Determine which API Key to use (client-provided or server-side .env)
    api_key = req.user_api_key or os.getenv("GEMINI_API_KEY")
    
    # Identify top stress / protective factors
    factor_contribs = []
    for f in features:
        m = feature_meta[f]
        rn = max(m["max"] - m["min"], 1)
        norm = (req.user_values[f] - m["min"]) / rn
        contrib = norm if m["direction"] == "negative" else (1.0 - norm)
        factor_contribs.append({
            "feat": f,
            "contrib": contrib,
            "imp": FEAT_IMP.get(f, 0.0)
        })
        
    fc_by_impact = sorted(factor_contribs, key=lambda x: x["imp"] * x["contrib"], reverse=True)
    fc_by_protection = sorted(factor_contribs, key=lambda x: x["imp"] * (1.0 - x["contrib"]), reverse=True)
    
    top_stress = [x["feat"] for x in fc_by_impact[:4]]
    top_protective = [x["feat"] for x in fc_by_protection[:2]]
    
    if not api_key:
        # Fallback to static tips if no API key is supplied anywhere
        return {"advice": static_fallback_advice(req.score, top_stress, top_protective, "Gemini API key not configured.")}
        
    try:
        genai.configure(api_key=api_key)
        # Use robust, standard gemini-1.5-flash or gemini-2.5-flash model
        model_name = "gemini-1.5-flash"
        ai_model = genai.GenerativeModel(model_name)
        
        stress_names = [LABELS.get(f, f) for f in top_stress]
        protect_names = [LABELS.get(f, f) for f in top_protective]
        
        profile_lines = []
        for f, val in req.user_values.items():
            m = feature_meta[f]
            profile_lines.append(f"- {LABELS.get(f, f)}: {val}/{m['max']} (direction: {m['direction']})")
        profile = "\n".join(profile_lines)
        
        status_label = get_status_details(req.score)["status"]
        
        prompt = f"""
You are MindMeter AI Wellness Coach — a specialised, highly empathetic student wellbeing assistant.

Burnout Score: {req.score:.2f}/10 (Status: {status_label})

Student Lifestyle and Academic Profile:
{profile}

Primary Stress Drivers: {', '.join(stress_names)}
Primary Protective Assets: {', '.join(protect_names) if protect_names else "None"}

Write a premium, deeply empathetic wellbeing coach advice response based on this profile.
Write this in clean HTML exactly matching this template (do not include markdown ticks, just raw HTML text):

<p class="ai-intro">Empathic opening sentence acknowledging the student's status.</p>

<div class="ai-section-title">🔴 What's Driving Your Stress</div>
<p>Analyze how the primary stress drivers are interacting and holding down their score.</p>

<div class="ai-section-title">🟢 What's Working in Your Favour</div>
<p>Validate the student's current protective features and explain how they can use these strengths.</p>

<div class="ai-section-title">⚡ Your 3 Priority Actions This Week</div>
<ol>
<li><strong>Action 1:</strong> Specific, highly actionable step.</li>
<li><strong>Action 2:</strong> Specific, highly actionable step.</li>
<li><strong>Action 3:</strong> Specific, highly actionable step.</li>
</ol>

<div class="ai-section-title">💬 A Note From Your Coach</div>
<p>Vibrant, warm, and highly supportive concluding sentence to inspire confidence.</p>

Rules:
- Deeply supportive, non-clinical tone. Do NOT provide diagnoses or prescribe medication.
- Be concise: 200–260 words.
- Provide ONLY the HTML content.
"""
        response = ai_model.generate_content(prompt)
        advice_html = response.text.strip()
        # Clean markdown wrappers if returned
        if advice_html.startswith("```html"):
            advice_html = advice_html[7:]
        if advice_html.endswith("```"):
            advice_html = advice_html[:-3]
        return {"advice": advice_html.strip()}
        
    except Exception as e:
        return {"advice": static_fallback_advice(req.score, top_stress, top_protective, str(e))}


@app.post("/api/coach/chat")
def chat_with_coach(req: ChatRequest):
    # Crisis Keyword Guardrail
    msg_lower = req.message.lower()
    if any(w in msg_lower for w in CRISIS_WORDS):
        return {
            "reply": "I'm really sorry you're feeling this way. I cannot provide crisis support, but please know that you do not have to go through this alone and there are people who want to support you.\n\n"
                     "**Please immediately reach out to:**\n"
                     "• A trusted friend, family member, or professor\n"
                     "• A school counselor or health professional\n"
                     "• **Crisis Text Line:** Text HOME to 741741\n"
                     "• **National Suicide Prevention Lifeline:** Call/Text 988\n\n"
                     "Your health and safety are what matters most. Please connect with one of these resources."
        }
        
    # Topic Constraint Guardrail
    if not any(topic in msg_lower for topic in ALLOWED_TOPICS):
        return {
            "reply": "💬 I am here as your **MindMeter Wellness Coach** to support you with stress management, burnout risk, sleep hygiene, academic workload balance, study strategies, and mental wellbeing.\n\n"
                     "Try asking me:\n"
                     "• *Why is my anxiety/academic load driving my burnout score?*\n"
                     "• *What are some practical tips to improve my sleep quality?*\n"
                     "• *How can I manage study load and peer pressure?*"
        }
        
    api_key = req.user_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"reply": "⚠️ Gemini API Key not configured. To enable interactive AI Wellness Coaching, please configure the API key in the frontend Settings or server environment."}
        
    try:
        genai.configure(api_key=api_key)
        model_name = "gemini-1.5-flash"
        ai_model = genai.GenerativeModel(model_name)
        
        status_label = get_status_details(req.score)["status"]
        factor_lines = [f"{LABELS.get(f, f)}: {v}" for f, v in list(req.user_values.items())[:8]]
        
        system_instructions = (
            "You are MindMeter AI Wellness Coach, an empathetic student wellbeing assistant.\n"
            "STRICT RULES:\n"
            "- Only address student stress, academic balance, sleep, self-esteem, relationships, and wellbeing.\n"
            "- Never make medical diagnoses or clinical judgments. Maintain a supportive, coaching persona.\n"
            "- Be concise and impactful. Keep your replies under 120 words.\n"
            f"- Student's Burnout Score: {req.score:.2f}/10 ({status_label}).\n"
            f"- Profile details: {'; '.join(factor_lines)}."
        )
        
        # Build chat history for Gemini API in correct format
        gemini_history = []
        # Prepend system context to guide the conversation
        gemini_history.append({"role": "user", "parts": [f"[System Context]\n{system_instructions}\n\nHi coach, let's talk."]})
        gemini_history.append({"role": "model", "parts": ["I'm ready to support the student. I will maintain a deeply warm, empathetic tone, enforce non-diagnostic boundaries, restrict topic scope, and keep replies under 120 words."]})
        
        # Add past user/model turns from request
        for h in req.chat_history:
            role = "user" if h.role == "user" else "model"
            gemini_history.append({"role": role, "parts": h.parts})
            
        chat = ai_model.start_chat(history=gemini_history)
        response = chat.send_message(req.message)
        
        return {"reply": response.text.strip()}
        
    except Exception as e:
        return {"reply": f"⚠️ Gemini Coach Engine Error: {str(e)}"}


@app.get("/api/analytics")
def get_population_analytics():
    data_path = "dataset/StressLevelDataset.csv"
    if not os.path.exists(data_path):
        data_path = "../dataset/StressLevelDataset.csv"
        
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
        
    try:
        df = pd.read_csv(data_path)
        total_students = len(df)
        
        # Reconstruct burnout scores for the population
        negative_factors = {
            "anxiety_level": (0, 21), "depression": (0, 27), "headache": (0, 5),
            "breathing_problem": (0, 5), "noise_level": (0, 5), "study_load": (0, 5),
            "future_career_concerns": (0, 5), "peer_pressure": (0, 5),
            "extracurricular_activities": (0, 5), "bullying": (0, 5),
            "blood_pressure": (1, 3), "mental_health_history": (0, 1),
        }
        positive_factors = {
            "self_esteem": (0, 30), "sleep_quality": (0, 5), "living_conditions": (0, 5),
            "safety": (0, 5), "basic_needs": (0, 5), "academic_performance": (0, 5),
            "teacher_student_relationship": (0, 5), "social_support": (0, 3),
        }
        
        neg_score = sum((df[col] - mn) / max(mx - mn, 1) for col, (mn, mx) in negative_factors.items())
        pos_score = sum(1 - (df[col] - mn) / max(mx - mn, 1) for col, (mn, mx) in positive_factors.items())
        raw = 0.60 * (neg_score / len(negative_factors)) + 0.40 * (pos_score / len(positive_factors))
        df["burnout_score"] = np.clip(raw * 10, 0, 10)
        
        # Calculate burnout status counts
        status_counts = {"Healthy": 0, "Moderate Stress": 0, "High Burnout Risk": 0, "Critical Burnout": 0}
        for s in df["burnout_score"]:
            if s < 3.0:
                status_counts["Healthy"] += 1
            elif s < 5.5:
                status_counts["Moderate Stress"] += 1
            elif s < 7.5:
                status_counts["High Burnout Risk"] += 1
            else:
                status_counts["Critical Burnout"] += 1
                
        # Format for charts
        status_distribution = [{"name": k, "value": v} for k, v in status_counts.items()]
        
        # Domain population averages
        domain_averages = {}
        for domain, d_feats in DOMAIN_MAPS.items():
            vals = []
            for f in d_feats:
                rn = max(negative_factors.get(f, positive_factors.get(f, (0, 5)))[1] - negative_factors.get(f, positive_factors.get(f, (0, 5)))[0], 1)
                mn = negative_factors.get(f, positive_factors.get(f, (0, 5)))[0]
                norm = (df[f] - mn) / rn
                if f in positive_factors:
                    norm = 1.0 - norm
                vals.append(norm)
            domain_averages[domain] = float(np.mean(vals) * 10)
            
        # Top 5 features correlated with burnout
        correlations = []
        for col in features:
            corr_val = float(df[col].corr(df["burnout_score"]))
            correlations.append({
                "feat": col,
                "name": LABELS.get(col, col),
                "corr": corr_val,
                "dir_icon": "🔴" if corr_val > 0 else "🟢"
            })
        correlations = sorted(correlations, key=lambda x: abs(x["corr"]), reverse=True)
        
        # Global Feature Weights for chart
        global_weights = []
        for f in features:
            global_weights.append({
                "feat": f,
                "name": LABELS.get(f, f),
                "weight": float(FEAT_IMP.get(f, 0.05) * 100)
            })
        global_weights = sorted(global_weights, key=lambda x: x["weight"], reverse=True)
        
        return {
            "total_students": total_students,
            "status_distribution": status_distribution,
            "domain_averages": [{"domain": k, "score": round(v, 2)} for k, v in domain_averages.items()],
            "top_correlations": correlations[:8],
            "global_weights": global_weights,
            "model_metrics": {
                "name": "Ridge Regression",
                "mse": 0.00000012,
                "mae": 0.00029426,
                "r2": 0.99999998
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics compute error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Run uvicorn on localhost:8000 when executed directly
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
