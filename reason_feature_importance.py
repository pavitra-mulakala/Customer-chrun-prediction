# -----------------------------------------------
# IDENTIFY FEATURES THAT AFFECT CHURN
# -----------------------------------------------

import os
import joblib
import pandas as pd
import numpy as np

print("\n--- IDENTIFYING FEATURES THAT AFFECT CHURN ---")

# -----------------------------------------------
# Load Model + Feature Names
# -----------------------------------------------
# NOTE: churn_pipeline.py saves a single dict to "final_pipeline.pkl"
# containing {"pipeline": ..., "threshold": ...} — it never writes
# "final_model.pkl" or "feature_names.pkl" separately. This previously
# caused a FileNotFoundError every time this script was run.
# Both the model and the feature names are pulled from that one file.

if not os.path.exists("final_pipeline.pkl"):
    raise FileNotFoundError(
        "final_pipeline.pkl not found — run churn_pipeline.py first."
    )

data = joblib.load("final_pipeline.pkl")
pipeline = data["pipeline"]

model = pipeline.named_steps["model"]
feature_names = pipeline.named_steps["prep"].get_feature_names_out()

model_name = type(model).__name__

# -----------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# -----------------------------------------------

if model_name == "RandomForestClassifier":

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

# -----------------------------------------------
# LOGISTIC REGRESSION COEFFICIENTS
# -----------------------------------------------

elif model_name == "LogisticRegression":

    coefficients = model.coef_[0]

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    feature_importance_df["importance"] = np.abs(feature_importance_df["coefficient"])

    feature_importance_df = feature_importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

else:

    print("Model does not support feature importance")
    exit()


# -----------------------------------------------
# CLEAN FEATURE NAMES FOR READABILITY
# -----------------------------------------------

feature_importance_df["feature_clean"] = (
    feature_importance_df["feature"]
    .str.replace("_", " ")
    .str.title()
)

# -----------------------------------------------
# DISPLAY TOP CHURN DRIVERS
# -----------------------------------------------

print("\nTop 10 Features Affecting Churn:\n")

top_features = feature_importance_df.head(10)

for i, row in top_features.iterrows():

    print(
        f"{row['feature_clean']}  → Importance Score: {round(row['importance'],4)}"
    )


# -----------------------------------------------
# SAVE FULL RESULTS
# -----------------------------------------------

feature_importance_df.to_csv("feature_importance.csv", index=False)

print("\nFeature importance file saved as feature_importance.csv")