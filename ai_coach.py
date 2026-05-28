"""
ai_coach.py — Gemini-powered AI burnout coach for MindMeter
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai


# =========================================================
# LOAD API KEY
# =========================================================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-3-flash-preview"


# =========================================================
# HUMAN READABLE LABELS
# =========================================================
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


# =========================================================
# ALLOWED CHAT DOMAINS (GUARDRAIL)
# =========================================================
ALLOWED_TOPICS = [
    "stress",
    "burnout",
    "mental",
    "anxiety",
    "depression",
    "study",
    "sleep",
    "focus",
    "motivation",
    "academic",
    "career",
    "peer",
    "social",
    "wellbeing",
    "health",
]


CRISIS_WORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "want to die",
]


def _is_allowed_topic(message: str) -> bool:
    msg = message.lower()
    return any(t in msg for t in ALLOWED_TOPICS)


def _is_crisis(message: str) -> bool:
    msg = message.lower()
    return any(w in msg for w in CRISIS_WORDS)


# =========================================================
# STATUS LABEL
# =========================================================
def _get_status_label(score: float) -> str:
    if score < 3:
        return "Healthy"
    if score < 5.5:
        return "Moderate Stress"
    if score < 7.5:
        return "High Burnout Risk"
    return "Critical Burnout"


# =========================================================
# GENERATE AI ADVICE
# =========================================================
def generate_advice(
    score,
    top_stress_factors,
    top_protective_factors,
    user_values,
    feature_meta,
):

    if not API_KEY:
        return _static_fallback(score, top_stress_factors, top_protective_factors)

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        stress_names = [LABELS.get(f, f) for f in top_stress_factors]
        protect_names = [LABELS.get(f, f) for f in top_protective_factors]

        factor_lines = []

        for f, val in user_values.items():
            m = feature_meta.get(f, {})
            mn = m.get("min", 0)
            mx = m.get("max", 5)
            direction = m.get("direction", "neutral")

            pct = round((val - mn) / max(mx - mn, 1) * 100)

            factor_lines.append(
                f"- {LABELS.get(f,f)}: {val}/{mx} ({direction}, {pct}% range)"
            )

        profile = "\n".join(factor_lines)
        status = _get_status_label(score)

        prompt = f"""
You are MindMeter AI Coach — a specialised student wellbeing assistant.

Burnout Score: {score:.1f}/10 → {status}

Student Profile:
{profile}

Stress Drivers: {', '.join(stress_names)}
Protective Factors: {', '.join(protect_names) if protect_names else "None"}

Write an empathetic coaching response.

FORMAT EXACTLY IN HTML:

<p class="ai-intro">Warm opening sentence.</p>

<div class="ai-section-title">🔴 What's Driving Your Stress</div>
<p>Explain main stress causes clearly.</p>

<div class="ai-section-title">🟢 What's Working in Your Favour</div>
<p>Highlight strengths.</p>

<div class="ai-section-title">⚡ Your 3 Priority Actions This Week</div>
<ol>
<li>Specific action</li>
<li>Specific action</li>
<li>Specific action</li>
</ol>

<div class="ai-section-title">💬 A Note From Your Coach</div>
<p>Encouraging closing message.</p>

Rules:
- supportive tone
- no diagnosis
- 200–280 words
- no extra text outside HTML
"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return _static_fallback(
            score,
            top_stress_factors,
            top_protective_factors,
            error=str(e),
        )


# =========================================================
# CHAT COACH (GUARDED CHATBOT)
# =========================================================
def generate_chat_reply(
    message,
    score,
    user_values,
    feature_meta,
    chat_history,
):

    if not API_KEY:
        return "⚠️ Gemini API key not configured."

    # ---------- Crisis Guard ----------
    if _is_crisis(message):
        return (
            "I'm really sorry you're feeling this way. "
            "I can't provide crisis support, but you deserve real help.\n\n"
            "Please contact:\n"
            "• A trusted friend or family member\n"
            "• A counselor or mental health professional\n"
            "• Local emergency or mental health helpline\n\n"
            "You don't have to handle this alone."
        )

    # ---------- Topic Guard ----------
    if not _is_allowed_topic(message):
        return (
            "💬 I'm here to help with **student stress, burnout, wellbeing, "
            "sleep, and academic balance**.\n\n"
            "Try asking:\n"
            "• Why is my burnout score high?\n"
            "• How can I reduce study stress?\n"
            "• How do I improve sleep quality?"
        )

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        status = _get_status_label(score)

        factor_lines = [
            f"{LABELS.get(f,f)}: {v}"
            for f, v in list(user_values.items())[:10]
        ]

        system_context = (
            "You are MindMeter AI Coach.\n"
            "STRICT RULES:\n"
            "- Only discuss student wellbeing topics.\n"
            "- Never answer unrelated questions.\n"
            "- No medical diagnosis.\n"
            "- Be empathetic and concise (<120 words).\n"
            f"Student burnout score: {score:.1f}/10 ({status}).\n"
            f"Profile: {'; '.join(factor_lines)}."
        )

        history = (
            chat_history
            if chat_history
            else [
                {"role": "user", "parts": [system_context]},
                {"role": "model", "parts": ["Ready to support the student."]},
            ]
        )

        chat = model.start_chat(history=history)
        response = chat.send_message(message)

        return response.text.strip()

    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"


# =========================================================
# STATIC FALLBACK
# =========================================================
def _static_fallback(score, stress, protective, error=""):

    stress_names = [LABELS.get(f, f) for f in stress[:3]]
    protect_names = [LABELS.get(f, f) for f in protective[:2]]

    status = _get_status_label(score)

    error_note = (
        f"<small>(Gemini unavailable: {error[:60]})</small><br><br>"
        if error
        else ""
    )

    protect_text = (
        f"Your strongest supports are <strong>{', '.join(protect_names)}</strong>."
        if protect_names
        else "Focus on rebuilding supportive habits gradually."
    )

    return f"""{error_note}
<p class="ai-intro">You're currently experiencing <strong>{status.lower()}</strong>.</p>

<div class="ai-section-title">🔴 What's Driving Your Stress</div>
<p>Main contributors: <strong>{', '.join(stress_names)}</strong>.</p>

<div class="ai-section-title">🟢 What's Working in Your Favour</div>
<p>{protect_text}</p>

<div class="ai-section-title">⚡ Your 3 Priority Actions This Week</div>
<ol>
<li>Maintain a consistent sleep schedule.</li>
<li>Take daily 10-minute recovery breaks.</li>
<li>Reach out to one supportive person.</li>
</ol>

<div class="ai-section-title">💬 A Note From Your Coach</div>
<p>Small consistent steps lead to recovery.</p>
"""