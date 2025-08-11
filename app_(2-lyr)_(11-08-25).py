import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

# ----- Header (fix title + caption) -----
st.title("SPR Performance Evaluation Tool")
st.caption("Predict λres and FWHM with dual models; compute Sensitivity, Q‑factor, and FOM")

# ----- Model metadata (explicit, edit as needed) -----
MODEL_CONFIG = {
    "lambda": {
        "path": "models/xgb_lambda.json",   # or .pkl if using joblib
        "target_transform": "log10",        # one of: None | 'log10' | 'log1p' | 'ln'
        "unit": "nm",
        "feature_order": [
            # put exact training feature names here in order
            # e.g., "n_core", "n_analyte", "thickness_nm", "wavelength_start_nm", ...
        ],
        "training_ranges": {  # optional: used for domain warnings
            # "n_analyte": (1.33, 1.39),
            # "thickness_nm": (0, 50),
        },
        "version": "lambda_v1.0"
    },
    "fwhm": {
        "path": "models/xgb_fwhm.json",
        "target_transform": None,
        "unit": "nm",
        "feature_order": [
            # same ordering discipline as above
        ],
        "training_ranges": {
            # ...
        },
        "version": "fwhm_v1.0"
    },
    "fwhm_floor_nm": 1e-6  # numerical safety for Q and FOM
}

# ----- Utilities -----
def invert_transform(y_pred, kind):
    if kind is None:
        return y_pred
    if kind == "log10":
        return np.power(10.0, y_pred)
    if kind == "log1p":
        return np.expm1(y_pred)
    if kind == "ln":
        return np.exp(y_pred)
    raise ValueError(f"Unknown target_transform: {kind}")

def align_features(df_in: pd.DataFrame, expected_order: list, defaults: dict | None = None):
    defaults = defaults or {}
    df = df_in.copy()
    # add missing
    for col in expected_order:
        if col not in df.columns:
            df[col] = defaults.get(col, 0.0)
    # drop extras (but keep a note)
    extras = [c for c in df.columns if c not in expected_order]
    if extras:
        st.warning(f"Dropping unexpected features: {extras}")
        df = df.drop(columns=extras)
    # reorder
    df = df[expected_order]
    return df

def compute_metrics(lambda_nm_1, lambda_nm_2, n1, n2, fwhm_nm, fwhm_floor=1e-6):
    d_n = float(n2) - float(n1)
    if d_n == 0:
        return np.nan, np.nan, np.nan, "Δn is zero; cannot compute sensitivity."
    d_lambda = float(lambda_nm_2) - float(lambda_nm_1)
    S = d_lambda / d_n  # nm/RIU
    fwhm_safe = max(float(fwhm_nm), fwhm_floor)
    Q = float(lambda_nm_1) / fwhm_safe
    FOM = S / fwhm_safe  # RIU^-1
    return S, Q, FOM, None

def warn_out_of_domain(x: dict, ranges: dict):
    msgs = []
    for k, (lo, hi) in (ranges or {}).items():
        if k in x and not (lo <= x[k] <= hi):
            msgs.append(f"{k}={x[k]} outside training range [{lo}, {hi}]")
    if msgs:
        st.warning("Extrapolation warning: " + "; ".join(msgs))

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    path = Path(path)
    if path.suffix in {".pkl", ".joblib"}:
        return joblib.load(path)
    # XGBoost JSON: adjust your loader if needed
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster

def predict_with_model(model, X_df, is_booster: bool):
    if is_booster:
        import xgboost as xgb
        dmat = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
        return model.predict(dmat)
    return model.predict(X_df)

# ----- Example inference block (call inside your UI events) -----
def run_inference(inputs_n1: dict, inputs_n2: dict):
    # 1) Align features
    is_booster_lambda = MODEL_CONFIG["lambda"]["path"].endswith(".json")
    is_booster_fwhm = MODEL_CONFIG["fwhm"]["path"].endswith(".json")

    X1 = pd.DataFrame([inputs_n1])
    X2 = pd.DataFrame([inputs_n2])

    X1_l = align_features(X1, MODEL_CONFIG["lambda"]["feature_order"])
    X2_l = align_features(X2, MODEL_CONFIG["lambda"]["feature_order"])
    X1_f = align_features(X1, MODEL_CONFIG["fwhm"]["feature_order"])
    # FWHM assumed independent of n; if it depends on n, prepare X2_f similarly

    # 2) Load models
    model_lambda = load_model(MODEL_CONFIG["lambda"]["path"])
    model_fwhm = load_model(MODEL_CONFIG["fwhm"]["path"])

    # 3) Predict
    y1_l = predict_with_model(model_lambda, X1_l, is_booster_lambda)[0]
    y2_l = predict_with_model(model_lambda, X2_l, is_booster_lambda)[0]
    y_f = predict_with_model(model_fwhm, X1_f, is_booster_fwhm)[0]

    # 4) Inverse transforms
    lambda1 = float(invert_transform(y1_l, MODEL_CONFIG["lambda"]["target_transform"]))
    lambda2 = float(invert_transform(y2_l, MODEL_CONFIG["lambda"]["target_transform"]))
    fwhm = float(invert_transform(y_f, MODEL_CONFIG["fwhm"]["target_transform"]))

    # 5) Numeric validity checks
    if not np.isfinite(lambda1) or lambda1 <= 0:
        st.error("Invalid λres prediction for n1 after inverse transform.")
    if not np.isfinite(lambda2) or lambda2 <= 0:
        st.error("Invalid λres prediction for n2 after inverse transform.")
    if not np.isfinite(fwhm) or fwhm <= 0:
        st.warning("FWHM prediction ≤ 0; applying safety floor.")
        fwhm = MODEL_CONFIG["fwhm_floor_nm"]

    # 6) Metrics
    S, Q, FOM, msg = compute_metrics(lambda1, lambda2, inputs_n1["n_analyte"], inputs_n2["n_analyte"], fwhm, MODEL_CONFIG["fwhm_floor_nm"])
    if msg:
        st.warning(msg)

    # 7) Domain warnings
    warn_out_of_domain(inputs_n1, MODEL_CONFIG["lambda"]["training_ranges"])
    warn_out_of_domain(inputs_n2, MODEL_CONFIG["lambda"]["training_ranges"])
    warn_out_of_domain(inputs_n1, MODEL_CONFIG["fwhm"]["training_ranges"])

    return {
        "lambda_nm_n1": lambda1,
        "lambda_nm_n2": lambda2,
        "fwhm_nm": fwhm,
        "S_nm_per_RIU": S,
        "Q": Q,
        "FOM_per_RIU": FOM
    }
