# 🧠 MindMeter --- AI Burnout Intelligence System

MindMeter is an **Explainable AI + Machine Learning system** that
predicts student burnout risk using lifestyle, academic, and mental
wellness indicators.

Instead of assigning a simple label, MindMeter generates a **Burnout
Intelligence Score (0--10)** and combines:

-   Machine Learning prediction\
-   Explainable AI reasoning\
-   Guardrailed Generative AI coaching

to deliver **personalized, human-centered wellbeing insights**.

------------------------------------------------------------------------

## 🚀 Project Overview

Student burnout rarely comes from one cause. It emerges from the
interaction between:

-   academic workload\
-   sleep quality\
-   social environment\
-   lifestyle balance\
-   mental health factors

MindMeter models these relationships using Machine Learning and
translates predictions into **actionable guidance**, not just analytics.

------------------------------------------------------------------------

## 🎯 Core Features

-   ✅ Burnout Intelligence Score (0--100)
-   ✅ Random Forest Regression Model
-   ✅ Explainable AI factor analysis
-   ✅ Gemini-powered AI Wellness Coach
-   ✅ Domain-restricted AI chatbot
-   ✅ Crisis-aware safety guardrails
-   ✅ Personalized AI-generated recommendations
-   ✅ Secure API key management via `.env`
-   ✅ Interactive Streamlit dashboard

------------------------------------------------------------------------

## 🧩 System Architecture

    Student Inputs
          ↓
    Data Preprocessing
          ↓
    Random Forest Regressor
          ↓
    Burnout Intelligence Score
          ↓
    Explainable AI Analysis
          ↓
    Gemini AI Coach (Guardrailed)
          ↓
    Personalized Guidance + Chat Support

------------------------------------------------------------------------

## 🧠 Machine Learning Approach

### Problem Type

Regression

### Model Used

Random Forest Regressor

### Why Random Forest?

-   Handles nonlinear behavioral relationships
-   Robust to noisy survey data
-   Works well on small datasets
-   Stable predictions with minimal tuning
-   Compatible with Explainable AI methods

------------------------------------------------------------------------

## 📊 Dataset

MindMeter uses a **student stress / mental wellbeing dataset from
Kaggle** (\~1100 samples).

Example features:

-   Anxiety level\
-   Depression indicators\
-   Sleep quality\
-   Academic pressure\
-   Social support\
-   Environmental stressors\
-   Lifestyle habits

The original target is transformed into:

    MindMeter Score → Burnout Risk (0–100)

⚠️ **Disclaimer:**\
This project is educational and analytical. It does **not** provide
medical diagnosis.

------------------------------------------------------------------------

## 🤖 AI Wellness Coach

MindMeter integrates **Google Gemini API** to convert predictions into
meaningful human guidance.

### Capabilities

-   Uses ML reasoning as context
-   Generates personalized coaching responses
-   Maintains empathetic tone
-   Refuses unrelated questions
-   Detects crisis language
-   Encourages professional help when needed

------------------------------------------------------------------------

## 🛡️ AI Safety & Guardrails

The AI coach includes:

-   Domain restriction (wellbeing topics only)
-   Role locking (MindMeter Coach persona)
-   Crisis keyword detection
-   Non-diagnostic responses
-   Safe fallback responses
-   Environment-based API key security

------------------------------------------------------------------------

## ⚙️ Tech Stack

### Machine Learning

-   Python
-   Pandas
-   NumPy
-   Scikit-learn

### Explainable AI & LLM

-   SHAP Explainability
-   Google Gemini API

### Application Layer

-   Streamlit
-   Joblib
-   python-dotenv

------------------------------------------------------------------------

## 📁 Project Structure

    MindMeter/
    │
    ├── dataset/
    │   └── students.csv
    │
    ├── notebook/
    │   └── training.ipynb
    │
    ├── model/
    │   ├── burnout_model.pkl
    │   ├── features.pkl
    │   └── feature_meta.pkl
    │
    ├── ai_coach.py
    ├── train.py
    ├── app.py
    ├── requirements.txt
    ├── .env
    └── README.md

------------------------------------------------------------------------

## 🏗️ Installation & Setup

### 1️⃣ Clone Repository

``` bash
git clone <repo-url>
cd MindMeter
```

### 2️⃣ Create Virtual Environment

``` bash
python -m venv venv
```

Activate:

Windows:

    venv\Scripts\activate

Mac/Linux:

    source venv/bin/activate

### 3️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file:

    GEMINI_API_KEY=your_api_key_here

### 5️⃣ Train Model

``` bash
python train.py
```

### 6️⃣ Run Application

``` bash
streamlit run app.py
```

Open:

    http://localhost:8501

------------------------------------------------------------------------

## 📈 Model Evaluation

Metrics used:

-   Mean Absolute Error (MAE)
-   R² Score
-   Cross-validation testing

------------------------------------------------------------------------

## 💡 Example Output

    MindMeter Score: 74 / 100
    Status: High Burnout Risk

AI Coach Example:

> "Your elevated academic load combined with reduced sleep recovery
> appears to be driving sustained fatigue."

------------------------------------------------------------------------

## 🌍 Real-World Applications

-   University wellness monitoring systems
-   Student self-assessment platforms
-   Preventive mental health tools
-   Productivity & lifestyle coaching apps

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Longitudinal burnout tracking
-   Intervention simulation
-   FastAPI backend architecture
-   User authentication & history tracking
-   Model monitoring & drift detection
-   Cloud deployment

------------------------------------------------------------------------

## ⚠️ Disclaimer

MindMeter is an educational AI project and **not a medical diagnostic
tool**. It provides wellbeing insights, not clinical advice.

------------------------------------------------------------------------

⭐ If you find MindMeter interesting, consider giving the repository a
star!
