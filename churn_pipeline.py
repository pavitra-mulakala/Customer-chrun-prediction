# ===========================================
# FINAL CORRECT PIPELINE (STRONG VERSION)
# ===========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

print("Fixed Pipeline Started...")

# ----------------------------------
# LOAD DATA
# ----------------------------------
df = pd.read_csv("telco_churn_dataset.csv")

drop_cols = ["customer_id", "churn_category", "churn_reason"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
df = df.dropna()

df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

# ----------------------------------
# SPLIT
# ----------------------------------
X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ----------------------------------
# COLUMN TYPES (FIXED)
# ----------------------------------
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object","string"]).columns

# ----------------------------------
# CLASS WEIGHTS (IMPORTANT)
# ----------------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(zip(np.unique(y_train), weights))

# ----------------------------------
# PREPROCESSOR
# ----------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ----------------------------------
# MODELS
# ----------------------------------
log_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)

rf_model = RandomForestClassifier(
    random_state=42,
    class_weight=class_weight_dict
)

# ----------------------------------
# PIPELINES
# ----------------------------------
pipe_log = Pipeline([
    ("prep", preprocessor),
    ("model", log_model)
])

pipe_rf = Pipeline([
    ("prep", preprocessor),
    ("model", rf_model)
])

# ----------------------------------
# GRID SEARCH
# ----------------------------------
grid_log = GridSearchCV(pipe_log, {
    "model__C": [0.01, 0.1, 1, 10]
}, cv=5, scoring="roc_auc")

grid_rf = GridSearchCV(pipe_rf, {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20]
}, cv=5, scoring="roc_auc")

grid_log.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

best_log = grid_log.best_estimator_
best_rf = grid_rf.best_estimator_

# ----------------------------------
# BUSINESS EVALUATION (BETTER)
# ----------------------------------
def evaluate(model, X_test, y_test, name):

    probs = model.predict_proba(X_test)[:, 1]

    best = None

    for t in np.arange(0.3, 0.7, 0.05):

        preds = (probs >= t).astype(int)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)

        if rec >= 0.80 and prec >= 0.45:

            if best is None or acc > best["acc"]:
                best = {
                    "model": name,
                    "threshold": t,
                    "acc": acc,
                    "rec": rec,
                    "prec": prec
                }

    return best

log_result = evaluate(best_log, X_test, y_test, "Logistic")
rf_result = evaluate(best_rf, X_test, y_test, "RF")

candidates = [r for r in [log_result, rf_result] if r]

if not candidates:
    raise Exception("No model meets business criteria")

final = max(candidates, key=lambda x: x["acc"])

final_model = best_log if final["model"] == "Logistic" else best_rf

# ----------------------------------
# CROSS-VALIDATION REPORTING
# ----------------------------------
# GridSearchCV already runs 5-fold CV internally (cv=5, scoring="roc_auc") to
# pick hyperparameters, but the fold-level results were never surfaced before.
# cv_results_ at best_index_ gives the mean/std ROC-AUC across those 5 folds
# for the winning hyperparameter combination of each model.
log_cv_mean = grid_log.cv_results_["mean_test_score"][grid_log.best_index_]
log_cv_std = grid_log.cv_results_["std_test_score"][grid_log.best_index_]
rf_cv_mean = grid_rf.cv_results_["mean_test_score"][grid_rf.best_index_]
rf_cv_std = grid_rf.cv_results_["std_test_score"][grid_rf.best_index_]

# ----------------------------------
# FULL METRIC COMPARISON (at each model's own best threshold)
# ----------------------------------
def full_metrics(model, X_test, y_test, threshold):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round((2 * precision_score(y_test, preds) * recall_score(y_test, preds))
                    / (precision_score(y_test, preds) + recall_score(y_test, preds) + 1e-9), 4),
        "ROC_AUC": round(roc_auc_score(y_test, probs), 4),
    }

comparison_rows = []
for name, model, result, cv_mean, cv_std in [
    ("Logistic", best_log, log_result, log_cv_mean, log_cv_std),
    ("RandomForest", best_rf, rf_result, rf_cv_mean, rf_cv_std),
]:
    row = {"Model": name, "CV_ROC_AUC_Mean": round(cv_mean, 4), "CV_ROC_AUC_Std": round(cv_std, 4)}
    if result:
        row["Threshold_Used"] = result["threshold"]
        row.update(full_metrics(model, X_test, y_test, result["threshold"]))
        row["Meets_Business_Criteria"] = True
    else:
        row["Threshold_Used"] = None
        row["Meets_Business_Criteria"] = False
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("model_comparison.csv", index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON (Accuracy / Precision / Recall / F1 / ROC-AUC)")
print("=" * 60)
print(comparison_df.to_string(index=False))

print("\n" + "=" * 60)
print("SELECTED MODEL:", final["model"])
print("=" * 60)
winner_row = comparison_df[comparison_df["Model"] == ("Logistic" if final["model"] == "Logistic" else "RandomForest")].iloc[0]
loser_name = "RandomForest" if final["model"] == "Logistic" else "Logistic"
loser_row_match = comparison_df[comparison_df["Model"] == loser_name]

print(f"Reason: Among models meeting the business floor (Recall >= 0.80, Precision >= 0.45),")
print(f"'{final['model']}' had the highest test-set Accuracy ({final['acc']:.4f}) at its")
print(f"business-optimal threshold ({final['threshold']:.2f}), with a 5-fold CV ROC-AUC of")
print(f"{winner_row['CV_ROC_AUC_Mean']:.4f} (+/- {winner_row['CV_ROC_AUC_Std']:.4f}) during hyperparameter search.")
if not loser_row_match.empty:
    loser_row = loser_row_match.iloc[0]
    if not loser_row["Meets_Business_Criteria"]:
        print(f"'{loser_name}' was rejected because no threshold in its sweep satisfied the")
        print(f"Recall >= 0.80 / Precision >= 0.45 business floor.")
    else:
        print(f"'{loser_name}' met the business floor too, but with lower test Accuracy "
              f"({loser_row['Accuracy']:.4f} vs {winner_row['Accuracy']:.4f}).")
print("\nFull comparison saved to model_comparison.csv")

# ----------------------------------
# SAVE PIPELINE + THRESHOLD
# ----------------------------------
joblib.dump({
    "pipeline": final_model,
    "threshold": final["threshold"]
}, "final_pipeline.pkl")

print("✅ Final pipeline saved with threshold")