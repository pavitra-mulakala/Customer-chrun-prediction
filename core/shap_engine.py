# =====================================================
# SHARED SHAP EXPLAINER LOGIC
# =====================================================
# This is the SAME build_explainer() function that lives in shap_analysis.py,
# extracted here so both the batch backend script (shap_analysis.py) and the
# live Streamlit app import ONE version instead of maintaining two copies
# that could silently drift apart.
#
# shap_analysis.py imports this instead of defining it locally:
#   from core.shap_engine import build_explainer
#
# Nothing about the explainer logic itself has changed from the original.

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def build_explainer(model, X_sample_df):
    """
    Model-aware SHAP explainer.

    Picks TreeExplainer for tree-based models (RandomForest, GradientBoosting)
    and LinearExplainer for LogisticRegression, instead of assuming a single
    model type. Returns SHAP values already sliced to the "churn = Yes" class
    so callers don't need to know about the underlying 2D/3D shape difference
    between explainer types.

    Returns:
        explainer: the raw shap explainer object (needed for interaction plots)
        model_type: string name of the model class, e.g. "RandomForestClassifier"
        sv_class1: shap.Explanation sliced to class 1 (churn = Yes)
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "The 'shap' package is required. Install it with: pip install shap"
        )

    model_type = type(model).__name__

    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model)
        raw = explainer(X_sample_df, check_additivity=False)
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_sample_df)
        raw = explainer(X_sample_df)
    else:
        raise ValueError(f"No SHAP explainer configured for model type '{model_type}'.")

    if raw.values.ndim == 3:
        sv_class1 = shap.Explanation(
            values=raw.values[:, :, 1],
            base_values=raw.base_values[:, 1] if np.ndim(raw.base_values) > 1 else raw.base_values,
            data=raw.data,
            feature_names=X_sample_df.columns.tolist(),
        )
    else:
        sv_class1 = raw

    return explainer, model_type, sv_class1


def explain_single_customer(model, X_customer_df):
    """
    Live, on-demand SHAP explanation for ONE new customer submitted through
    the Streamlit prediction form — this is what powers the interactive
    Churn Prediction page, as opposed to the batch 500-row sample used for
    the pre-generated Global/Local SHAP plots.

    X_customer_df must be a single-row DataFrame, already preprocessed the
    same way as training data (via the pipeline's own preprocessor).
    """
    explainer, model_type, sv_class1 = build_explainer(model, X_customer_df)
    return sv_class1.values[0], sv_class1.base_values if np.ndim(sv_class1.base_values) == 0 else sv_class1.base_values[0]
