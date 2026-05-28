# =========================================================
# MINDMETER - PRODUCTION GRADE TRAINING PIPELINE
# =========================================================

"""
FEATURES:
- Proper burnout classification
- Reduced leakage
- XGBoost classification
- SMOTE balancing
- Hyperparameter tuning
- Cross validation
- SHAP explainability
- Confidence scoring
- Metrics export
- Feature importance export
- Production-ready artifacts
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import shap


# =========================================================
# WARNINGS
# =========================================================

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "dataset/StressLevelDataset.csv"

MODEL_DIR = "model"

RANDOM_STATE = 42

TEST_SIZE = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================

print("\n==============================")
print("LOADING DATASET")
print("==============================")

df = pd.read_csv(DATA_PATH)

print(f"\nDataset Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# NORMALIZED BURNOUT SCORE
# =========================================================

print("\n==============================")
print("CREATING BURNOUT LABELS")
print("==============================")

# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

depression_norm = df["depression"] / 27

anxiety_norm = df["anxiety_level"] / 21

sleep_norm = df["sleep_quality"] / 5

study_load_norm = df["study_load"] / 5

peer_pressure_norm = df["peer_pressure"] / 5

social_support_norm = df["social_support"] / 3


# ---------------------------------------------------------
# WEIGHTED SCORE
# ---------------------------------------------------------

burnout_score = (

    0.35 * depression_norm +

    0.30 * anxiety_norm +

    0.15 * study_load_norm +

    0.10 * peer_pressure_norm -

    0.07 * sleep_norm -

    0.03 * social_support_norm
)


# =========================================================
# CLASSIFICATION LABELS
# =========================================================

def classify_burnout(score):

    if score >= 0.72:
        return "Critical"

    elif score >= 0.55:
        return "High"

    elif score >= 0.38:
        return "Moderate"

    else:
        return "Healthy"


df["burnout_risk"] = burnout_score.apply(classify_burnout)


print("\nBurnout Distribution:")

print(df["burnout_risk"].value_counts())


# =========================================================
# FEATURES
# =========================================================

"""
IMPORTANT:
Avoid direct leakage by excluding:
- anxiety_level
- depression

Keep related contextual features.
"""

FEATURES = [

    "self_esteem",

    "mental_health_history",

    "headache",

    "sleep_quality",

    "breathing_problem",

    "noise_level",

    "living_conditions",

    "safety",

    "basic_needs",

    "academic_performance",

    "study_load",

    "teacher_student_relationship",

    "future_career_concerns",

    "social_support",

    "peer_pressure",

    "extracurricular_activities",

    "bullying"
]


TARGET = "burnout_risk"


X = df[FEATURES]

y = df[TARGET]


# =========================================================
# LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nEncoded Classes:")

for i, cls in enumerate(label_encoder.classes_):

    print(f"{i} -> {cls}")


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=TEST_SIZE,

    stratify=y_encoded,

    random_state=RANDOM_STATE
)


print("\nTrain Shape:", X_train.shape)

print("Test Shape:", X_test.shape)


# =========================================================
# SMOTE BALANCING
# =========================================================

print("\n==============================")
print("APPLYING SMOTE")
print("==============================")

smote = SMOTE(random_state=RANDOM_STATE)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)

print("\nBalanced Training Shape:")

print(X_train_balanced.shape)


# =========================================================
# XGBOOST MODEL
# =========================================================

base_model = XGBClassifier(

    objective="multi:softprob",

    num_class=len(label_encoder.classes_),

    eval_metric="mlogloss",

    random_state=RANDOM_STATE
)


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

print("\n==============================")
print("HYPERPARAMETER TUNING")
print("==============================")

param_grid = {

    "n_estimators": [100, 200],

    "max_depth": [3, 5, 7],

    "learning_rate": [0.01, 0.05, 0.1],

    "subsample": [0.8, 1.0],

    "colsample_bytree": [0.8, 1.0]
}


grid_search = GridSearchCV(

    estimator=base_model,

    param_grid=param_grid,

    scoring="f1_weighted",

    cv=5,

    verbose=1,

    n_jobs=-1
)


print("\nTraining Model...\n")

grid_search.fit(
    X_train_balanced,
    y_train_balanced
)


best_model = grid_search.best_estimator_


print("\nBest Parameters:")

print(grid_search.best_params_)


# =========================================================
# CALIBRATION
# =========================================================

print("\n==============================")
print("CALIBRATING MODEL")
print("==============================")

calibrated_model = CalibratedClassifierCV(
    best_model,
    method="isotonic",
    cv=5
)

calibrated_model.fit(
    X_train_balanced,
    y_train_balanced
)


# =========================================================
# PREDICTIONS
# =========================================================

y_pred = calibrated_model.predict(X_test)

y_probs = calibrated_model.predict_proba(X_test)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall: {recall:.4f}")

print(f"Weighted F1: {f1:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

print(report)


# =========================================================
# CROSS VALIDATION
# =========================================================

print("\n==============================")
print("CROSS VALIDATION")
print("==============================")

cv_scores = cross_val_score(

    best_model,

    X,

    y_encoded,

    cv=5,

    scoring="f1_weighted"
)

print("\nCV Scores:")

print(cv_scores)

print(f"\nMean CV Score: {cv_scores.mean():.4f}")


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = best_model.feature_importances_

feature_importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(feature_importance_df)


# ---------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# ---------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.barh(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.show()


# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

print("\n==============================")
print("GENERATING SHAP EXPLANATIONS")
print("==============================")

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURES
)


# =========================================================
# SAMPLE PREDICTION
# =========================================================

print("\n==============================")
print("SAMPLE PREDICTION")
print("==============================")

sample_input = X_test.iloc[[0]]

sample_probs = calibrated_model.predict_proba(sample_input)[0]

sample_prediction = calibrated_model.predict(sample_input)[0]

predicted_class = label_encoder.inverse_transform(
    [sample_prediction]
)[0]

confidence = np.max(sample_probs)


print(f"\nPrediction: {predicted_class}")

print(f"Confidence: {confidence:.2%}")

print("\nProbabilities:")

for cls, prob in zip(
    label_encoder.classes_,
    sample_probs
):

    print(f"{cls}: {prob:.4f}")


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "cross_validation_mean": float(
        cv_scores.mean()
    ),

    "best_parameters": grid_search.best_params_
}


with open(
    os.path.join(MODEL_DIR, "metrics.json"),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

feature_importance_df.to_csv(

    os.path.join(
        MODEL_DIR,
        "feature_importance.csv"
    ),

    index=False
)


# =========================================================
# SAVE MODEL ARTIFACTS
# =========================================================

joblib.dump(

    calibrated_model,

    os.path.join(
        MODEL_DIR,
        "burnout_classifier.pkl"
    )
)


joblib.dump(

    label_encoder,

    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)


joblib.dump(

    FEATURES,

    os.path.join(
        MODEL_DIR,
        "features.pkl"
    )
)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

pipeline_artifacts = {

    "model": calibrated_model,

    "label_encoder": label_encoder,

    "features": FEATURES
}


joblib.dump(

    pipeline_artifacts,

    os.path.join(
        MODEL_DIR,
        "complete_pipeline.pkl"
    )
)


# =========================================================
# SUCCESS MESSAGE
# =========================================================

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print("\nSaved Files:")

print("- burnout_classifier.pkl")

print("- label_encoder.pkl")

print("- features.pkl")

print("- metrics.json")

print("- feature_importance.csv")

print("- complete_pipeline.pkl")

