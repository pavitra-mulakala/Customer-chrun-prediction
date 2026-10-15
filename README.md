# Customer Churn Prediction — Backend (Complete)

## What's in this folder

| File | Purpose |
|---|---|
| `churn_pipeline.py` | Trains the model (only needed if you want to retrain — `final_pipeline.pkl` is already included) |
| `shap_analysis.py` | **Main script.** Predicts, explains with SHAP, gets AI suggestions, generates all plots and reports |
| `suggestions.py` | AI suggestion function only (needs your Groq API key) |
| `core/shap_engine.py` | Shared SHAP explainer logic used by `shap_analysis.py` |
| `final_pipeline.pkl` | Your already-trained model — included so you don't have to retrain |
| `telco_churn_dataset.csv` | Your 50,000-row dataset |
| `feature.py`, `reason_feature_importance.py` | Supporting scripts, not required for the main flow |
| `business_insights_summary.csv`, `model_comparison.csv` | Already-generated reports (will be overwritten when you run `shap_analysis.py`) |

## Setup (one time only)

**1. Install everything needed:**
```powershell
pip install pandas numpy scikit-learn shap matplotlib joblib python-dotenv groq
```

**2. Create your `.env` file** in this same folder, containing exactly one line:
```
GROQ_API_KEY=gsk_your_actual_key_here
```
Get a key at https://console.groq.com/keys — use a **fresh** key, not any key that appeared in a chat or shared document before.

## Running it

```powershell
python shap_analysis.py
```

**This will take several minutes — that's expected, not a bug.** You'll now see timestamped progress messages the whole way through (e.g. `[14:32:10] Loading final_pipeline.pkl...`), so it should never look frozen. Rough timing on a typical laptop:
- Loading the model + dataset: under 1 minute
- Computing SHAP values for the sample: 1-5 minutes
- Global + local plots: 1-3 minutes
- Business insights: a few seconds

**By default, the slowest possible step (the SHAP interaction plot) is turned OFF**, because it can take 10-30+ minutes on a Random Forest this size and isn't required for your project's core deliverables. If you specifically want it, open `shap_analysis.py`, find this line near the top:
```python
GENERATE_INTERACTION_PLOT = False
```
and change it to `True`. Only do this if you're prepared to wait.

## What you'll get when it finishes

- Console output ending in `DONE (total time: X.X minutes)`
- `powerbi_data.csv` — updated
- `business_insights_summary.csv` — updated, with real live-computed numbers
- `shap_plots/` folder containing:
  - `global_summary_plot.png`, `global_bar_plot.png`, `global_dependence_plot.png`
  - `waterfall_customer_1.png` through `5.png`
  - `force_customer_1.png` through `5.png`
  - `interaction_plot.png` (only if you enabled it above)

## If it still errors

Copy the **entire** error message (not just the last line) and share it — the exact traceback tells you exactly which line failed and why.
