# import pandas as pd

# ADVICE_MAP = {

#     "anxiety_level":
#         "Practice short breathing or journaling exercises.",

#     "sleep_quality":
#         "Maintain consistent sleep and reduce night screen time.",

#     "study_load":
#         "Break study sessions into smaller focused blocks.",

#     "peer_pressure":
#         "Focus on personal progress rather than comparison.",

#     "social_support":
#         "Connect regularly with supportive friends or mentors.",

#     "depression":
#         "Consider talking to someone you trust or a counselor."
# }


# def get_top_factors(shap_values, X, index, top_n=5):

#     contrib = pd.Series(
#         shap_values[index],
#         index=X.columns
#     )

#     return contrib.sort_values(
#         key=abs,
#         ascending=False
#     ).head(top_n)


# def generate_advice(shap_values, X, index):

#     factors = get_top_factors(shap_values, X, index)

#     advice = []

#     for feature, impact in factors.items():

#         if impact > 0 and feature in ADVICE_MAP:
#             advice.append(
#                 f"{feature.replace('_',' ').title()}: "
#                 + ADVICE_MAP[feature]
#             )

#     return advice


# def comfort_message(score):

#     if score > 75:
#         return "You're not failing — you're carrying a heavy load."

#     elif score > 50:
#         return "You're managing many pressures. Small adjustments help."

#     return "You're maintaining a healthy balance. Keep going."