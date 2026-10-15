# =====================================================
# SHAP ANALYSIS + BUSINESS INSIGHTS (MERGED, FINAL)
# =====================================================
# Single entry point: python shap_analysis.py
#
# Flow:
#   Load model -> Load dataset -> Predict -> Probability -> Prediction
#   -> Dataset Reason -> SHAP Reason -> AI Suggestion -> powerbi_data.csv
#   -> Global SHAP plots -> Local SHAP plots -> Business Insights
#   -> business_insights_summary.csv -> Finished
#
# NOTE ON SPEED: computing SHAP values for a Random Forest with many trees
# is CPU-intensive and can take several minutes with NO console output in
# between — this is normal, not a freeze. This version prints a progress
# message before every slow step so it's always clear what's running.
# The SHAP "interaction values" plot is the single slowest part by far
# (can take 10-30+ minutes on a large forest) and is OFF by default —
# set GENERATE_INTERACTION_PLOT = True below if you want it and are
# willing to wait.

import os
import time
import numpy as np
import pandas as pd
import joblib
from suggestions import get_suggestion
from core.shap_engine import build_explainer, SHAP_AVAILABLE

try:
    import shap
    import matplotlib.pyplot as plt
except ImportError:
    pass

PIPELINE_PATH = "final_pipeline.pkl"
DATASET_PATH = "telco_churn_dataset.csv"
SHAP_PLOTS_DIR = "shap_plots"
POWERBI_OUTPUT = "powerbi_data.csv"
INSIGHTS_OUTPUT = "business_insights_summary.csv"
N_LOCAL_CUSTOMERS = 5
SAMPLE_SIZE = 500
RANDOM_SEED = 42

# The interaction plot is by far the slowest part of this script for large
# Random Forests. Leave this False for a fast run; set True only if you
# specifically need that plot and can wait.
GENERATE_INTERACTION_PLOT = False


def _log(msg):
    """Timestamped progress print so long steps never look frozen."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# =====================================================
# SHARED LOADING
# =====================================================
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"{PIPELINE_PATH} not found — run churn_pipeline.py first.")
    _log(f"Loading {PIPELINE_PATH} (this can take 10-30 seconds, it's a large file)...")
    data = joblib.load(PIPELINE_PATH)
    if "pipeline" not in data or "threshold" not in data:
        raise ValueError(f"{PIPELINE_PATH} is missing expected 'pipeline'/'threshold' keys.")
    _log("Pipeline loaded.")
    return data["pipeline"], data["threshold"]


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found in the working directory.")

    _log(f"Loading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    required_cols = {"total_charges", "churn", "churn_reason", "service_provider"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df = df.dropna()
    df["churn"] = df["churn"].map({"No": 0, "Yes": 1})
    _log(f"Dataset loaded: {len(df):,} rows.")
    return df


def prepare_sample(pipeline, df):
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    _log("Preprocessing data (transform + sampling)...")
    drop_cols = ["customer_id", "churn_category", "churn_reason", "churn"]
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_processed = np.array(preprocessor.transform(X_raw))
    feature_names = preprocessor.get_feature_names_out()

    sample_idx = np.random.RandomState(RANDOM_SEED).choice(len(X_processed), SAMPLE_SIZE, replace=False)
    X_sample = X_processed[sample_idx]
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    dataset_reasons_sample = df["churn_reason"].iloc[sample_idx].reset_index(drop=True)
    service_providers_sample = df["service_provider"].iloc[sample_idx].reset_index(drop=True)
    _log(f"Sample ready: {SAMPLE_SIZE} rows, {len(feature_names)} features. Model type: {type(model).__name__}")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "sample_idx": sample_idx,
        "X_sample": X_sample,
        "X_sample_df": X_sample_df,
        "dataset_reasons_sample": dataset_reasons_sample,
        "service_providers_sample": service_providers_sample,
    }


# =====================================================
# PART 1: PREDICTION + REASON ATTRIBUTION + AI SUGGESTION
# =====================================================
def run_prediction_and_suggestions(ctx, sv_class1, threshold, top_n=N_LOCAL_CUSTOMERS):
    model = ctx["model"]
    feature_names = ctx["feature_names"]
    sample_idx = ctx["sample_idx"]
    X_sample = ctx["X_sample"]
    dataset_reasons_sample = ctx["dataset_reasons_sample"]
    service_providers_sample = ctx["service_providers_sample"]

    y_prob = model.predict_proba(X_sample)[:, 1]
    indices = np.argsort(y_prob)[-top_n:]

    print("\n----- RESULTS -----\n")
    results = []

    for i, idx in enumerate(indices, start=1):
        dataset_reason = dataset_reasons_sample.iloc[idx]
        provider = service_providers_sample.iloc[idx]

        vals = sv_class1.values[idx]
        shap_df = pd.DataFrame({"feature": feature_names, "shap": vals})
        shap_df = shap_df[shap_df["shap"] > 0].sort_values(by="shap", ascending=False).head(3)
        shap_reason = ", ".join(shap_df["feature"].values)

        probability = y_prob[idx]
        prediction = "Will Churn" if probability >= threshold else "Will Not Churn"

        _log(f"Getting AI suggestion for customer {i}/{top_n}...")
        try:
            suggestion = get_suggestion(provider, dataset_reason, shap_reason)
        except Exception as e:
            suggestion = f"Error generating suggestion: {e}"

        print(f"Service Provider: {provider}")
        print(f"Dataset Reason: {dataset_reason}")
        print(f"SHAP Reason: {shap_reason}")
        print(f"Churn Probability: {round(probability*100,2)}%")
        print(f"Prediction: {prediction}")
        print(f"Suggestion: {suggestion}\n")
        print("-" * 70)

        results.append({
            "Customer_ID": sample_idx[idx],
            "Service_Provider": provider,
            "Churn_Probability": round(probability * 100, 2),
            "Prediction": prediction,
            "Dataset_Reason": dataset_reason,
            "SHAP_Reason": shap_reason,
            "Suggestion": suggestion,
        })

    return pd.DataFrame(results), indices, y_prob


def export_powerbi_csv(results_df):
    try:
        results_df.to_csv(POWERBI_OUTPUT, index=False)
        print(f"\nPower BI file saved as {POWERBI_OUTPUT}")
        print("CSV Location:", os.path.abspath(POWERBI_OUTPUT))
    except Exception as e:
        print(f"[ERROR] Failed to export {POWERBI_OUTPUT}: {e}")


# =====================================================
# PART 2: GLOBAL SHAP
# =====================================================
def generate_global_shap(explainer, model_type, sv_class1, X_sample_df, feature_names):
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print(f"GLOBAL SHAP  (model: {model_type})")
    print("=" * 60)

    _log("Generating summary plot...")
    plt.figure()
    shap.summary_plot(sv_class1, X_sample_df, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_PLOTS_DIR}/global_summary_plot.png", dpi=150)
    plt.close()
    print(f"Saved: {SHAP_PLOTS_DIR}/global_summary_plot.png")

    _log("Generating bar plot...")
    plt.figure()
    shap.plots.bar(sv_class1, show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_PLOTS_DIR}/global_bar_plot.png", dpi=150)
    plt.close()
    print(f"Saved: {SHAP_PLOTS_DIR}/global_bar_plot.png")

    _log("Generating dependence plot...")
    mean_abs_shap = np.abs(sv_class1.values).mean(axis=0)
    top_feature_idx = int(np.argmax(mean_abs_shap))
    top_feature_name = feature_names[top_feature_idx]

    plt.figure()
    shap.dependence_plot(top_feature_idx, sv_class1.values, X_sample_df, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_PLOTS_DIR}/global_dependence_plot.png", dpi=150)
    plt.close()
    print(f"Saved: {SHAP_PLOTS_DIR}/global_dependence_plot.png  (top feature: {top_feature_name})")

    if not GENERATE_INTERACTION_PLOT:
        print("Interaction plot skipped (GENERATE_INTERACTION_PLOT = False — this step is very slow "
              "on large Random Forests; set it to True in this file if you want it).")
        return

    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier"):
        try:
            _log("Computing interaction values — this is the slow step, can take 10-30+ minutes...")
            interaction_values = explainer.shap_interaction_values(X_sample_df)
            inter_class1 = interaction_values[1] if isinstance(interaction_values, list) else interaction_values
            plt.figure()
            shap.summary_plot(inter_class1, X_sample_df, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f"{SHAP_PLOTS_DIR}/interaction_plot.png", dpi=150)
            plt.close()
            print(f"Saved: {SHAP_PLOTS_DIR}/interaction_plot.png")
        except Exception as e:
            print(f"Interaction plot skipped (error): {e}")
    else:
        print("Interaction plot skipped — only supported for tree-based models.")


# =====================================================
# PART 3: LOCAL SHAP
# =====================================================
def generate_local_shap(sv_class1, X_sample_df, feature_names, indices, y_prob, sample_idx):
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("LOCAL SHAP (top 5 high-risk customers)")
    print("=" * 60)

    for rank, idx in enumerate(indices, start=1):
        customer_ref = sample_idx[idx]
        probability = y_prob[idx]
        vals = sv_class1.values[idx]

        _log(f"Generating local SHAP plots for customer {rank}/{len(indices)}...")

        shap_df = pd.DataFrame({"feature": feature_names, "shap": vals})
        top_negative = shap_df[shap_df["shap"] < 0].sort_values("shap", ascending=True).head(3)

        print(f"\n--- Customer {rank} (index {customer_ref}) | Churn probability: {round(probability*100,2)}% ---")
        print("Top factors DECREASING churn risk:")
        for _, r in top_negative.iterrows():
            print(f"  - {r['feature']}  (impact: {round(r['shap'], 4)})")

        base_val = sv_class1.base_values[idx] if np.ndim(sv_class1.base_values) > 0 else sv_class1.base_values
        explanation = shap.Explanation(
            values=vals,
            base_values=base_val,
            data=X_sample_df.iloc[idx].values,
            feature_names=feature_names,
        )

        try:
            plt.figure()
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            plt.savefig(f"{SHAP_PLOTS_DIR}/waterfall_customer_{rank}.png", dpi=150)
            plt.close()
            print(f"Saved: {SHAP_PLOTS_DIR}/waterfall_customer_{rank}.png")
        except Exception as e:
            print(f"[WARNING] Waterfall plot failed for customer {rank}, continuing anyway: {e}")

        try:
            shap.force_plot(base_val, vals, X_sample_df.iloc[idx], feature_names=feature_names,
                             matplotlib=True, show=False)
            plt.tight_layout()
            plt.savefig(f"{SHAP_PLOTS_DIR}/force_customer_{rank}.png", dpi=150)
            plt.close()
            print(f"Saved: {SHAP_PLOTS_DIR}/force_customer_{rank}.png")
        except Exception as e:
            print(f"[WARNING] Force plot skipped for customer {rank}: {e}")


# =====================================================
# PART 5: BUSINESS INSIGHTS
# =====================================================
def generate_business_insights(df):
    print("\n============= BUSINESS INSIGHTS =============")

    d = df.copy()
    d["churn_flag"] = d["churn"]

    rows = []

    provider_churn = d.groupby("service_provider")["churn_flag"].agg(["mean", "count"])
    provider_churn["mean"] *= 100
    worst_provider = provider_churn["mean"].idxmax()
    print("\n--- Provider-wise churn ---")
    print(provider_churn.rename(columns={"mean": "churn_rate_%", "count": "customers"}))
    rows.append(("Provider-wise churn", f"{worst_provider} churns most at {provider_churn.loc[worst_provider,'mean']:.2f}%"))

    contract_churn = d.groupby("contract")["churn_flag"].agg(["mean", "count"])
    contract_churn["mean"] *= 100
    worst_contract = contract_churn["mean"].idxmax()
    print("\n--- Contract-wise churn ---")
    print(contract_churn.rename(columns={"mean": "churn_rate_%", "count": "customers"}))
    rows.append(("Contract-wise churn", f"{worst_contract} churns most at {contract_churn.loc[worst_contract,'mean']:.2f}%"))

    corr_sat = d["satisfaction_score"].corr(d["churn_flag"])
    print(f"\n--- Satisfaction correlation with churn: {corr_sat:.3f} ---")
    rows.append(("Satisfaction correlation", f"corr = {corr_sat:.3f}"))

    charge_bins = pd.qcut(d["monthly_charge"], q=4, duplicates="drop")
    charge_churn = d.groupby(charge_bins, observed=True)["churn_flag"].mean() * 100
    highest_charge_band = charge_churn.idxmax()
    print(f"\n--- Highest-churn monthly charge band: {highest_charge_band} at {charge_churn.max():.2f}% ---")
    rows.append(("Monthly charge impact", f"highest churn band: {highest_charge_band} at {charge_churn.max():.2f}%"))

    tenure_bins = pd.cut(d["tenure"], bins=5)
    tenure_churn = d.groupby(tenure_bins, observed=True)["churn_flag"].mean() * 100
    rows.append(("Tenure impact", f"earliest band {tenure_churn.iloc[0]:.2f}% vs longest band {tenure_churn.iloc[-1]:.2f}%"))
    print(f"\n--- Tenure: earliest band {tenure_churn.iloc[0]:.2f}% vs longest band {tenure_churn.iloc[-1]:.2f}% ---")

    churned_revenue = d.loc[d["churn_flag"] == 1, "monthly_charge"].sum()
    total_revenue = d["monthly_charge"].sum()
    revenue_at_risk_pct = churned_revenue / total_revenue * 100
    annualized_loss = churned_revenue * 12
    print(f"\n--- Revenue: {revenue_at_risk_pct:.2f}% of monthly revenue lost to churn, {annualized_loss:,.0f} annualized ---")
    rows.append(("Revenue impact", f"{revenue_at_risk_pct:.2f}% of monthly revenue, {annualized_loss:,.0f} annualized"))

    value_median = d["monthly_charge"].median()
    d["value_segment"] = np.where(d["monthly_charge"] >= value_median, "High Value", "Low Value")
    d["risk_segment"] = np.where(d["satisfaction_score"] <= d["satisfaction_score"].median(), "High Risk", "Low Risk")
    segment_table = d.groupby(["value_segment", "risk_segment"])["churn_flag"].agg(["mean", "count"])
    segment_table["mean"] *= 100
    priority_segment = segment_table["mean"].idxmax()
    print(f"\n--- Priority segment: {priority_segment} at {segment_table['mean'].max():.2f}% churn ---")
    rows.append(("Priority segment", f"{priority_segment} at {segment_table['mean'].max():.2f}% churn"))

    high_risk_mask = (
        (d["contract"] == "Month-to-month")
        & (d["satisfaction_score"] <= d["satisfaction_score"].quantile(0.25))
        & (d["churn_flag"] == 0)
    )
    high_risk_customers = d[high_risk_mask]
    print(f"\n--- High-risk active customers: {len(high_risk_customers):,} "
          f"({high_risk_customers['monthly_charge'].sum():,.2f} monthly revenue at risk) ---")
    rows.append(("High-risk active customers",
                  f"{len(high_risk_customers):,} customers, {high_risk_customers['monthly_charge'].sum():,.0f} monthly revenue at risk"))

    reason_counts = d.loc[d["churn_flag"] == 1, "churn_reason"].value_counts().head(5)
    top_reason = reason_counts.idxmax()
    print(f"\n--- Top churn reason: '{top_reason}' ({reason_counts.max()} customers) ---")
    rows.append(("Top churn reason", f"{top_reason}: {reason_counts.max()} customers"))

    summary_df = pd.DataFrame(rows, columns=["Insight_Area", "Finding"])
    try:
        summary_df.to_csv(INSIGHTS_OUTPUT, index=False)
        print(f"\nSaved: {INSIGHTS_OUTPUT}")
    except Exception as e:
        print(f"[ERROR] Failed to save {INSIGHTS_OUTPUT}: {e}")


# =====================================================
# MAIN
# =====================================================
def main():
    start_time = time.time()
    print("\n==============================")
    print("DATASET + SHAP (FINAL)")
    print("==============================\n")

    pipeline, threshold = load_pipeline()
    df = load_dataset()
    ctx = prepare_sample(pipeline, df)

    if not SHAP_AVAILABLE:
        raise ImportError(
            "The 'shap' package is required for this script. Install it with: pip install shap"
        )

    _log("Building SHAP explainer and computing SHAP values for the sample "
         "(this is the main slow step — can take a few minutes)...")
    explainer, model_type, sv_class1 = build_explainer(ctx["model"], ctx["X_sample_df"])
    _log("SHAP values computed.")

    results_df, indices, y_prob = run_prediction_and_suggestions(ctx, sv_class1, threshold)
    export_powerbi_csv(results_df)

    generate_global_shap(explainer, model_type, sv_class1, ctx["X_sample_df"], ctx["feature_names"])
    generate_local_shap(sv_class1, ctx["X_sample_df"], ctx["feature_names"], indices, y_prob, ctx["sample_idx"])
    generate_business_insights(df)

    elapsed = time.time() - start_time
    print("\n==============================")
    print(f"DONE  (total time: {elapsed/60:.1f} minutes)")
    print("==============================")


if __name__ == "__main__":
    main()
