# app.py
from __future__ import annotations

import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "SPR Predictor: Resonance Wavelength (λres) and FWHM"
APP_DESC = (
    "Two-model inference with correct preprocessing: "
    "log-transform for λres features and target inversion, raw for FWHM."
)
EPS = 1e-9  # to stabilize log transform

# --------- Cache helpers ---------
@st.cache_resource(show_spinner=False)
def load_pickle_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def read_any_table(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        raise ValueError("Please upload a .csv or .xlsx file.")

# --------- Utility: feature alignment & validation ---------
def get_model_feature_names(model) -> list[str] | None:
    # Prefer sklearn attribute when available
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Fallback: try XGBoost booster feature_names (less reliable for pandas)
    if hasattr(model, "get_booster"):
        try:
            fn = model.get_booster().feature_names
            if fn:
                return list(fn)
        except Exception:
            pass
    return None

def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}. "
            f"Your data must include exactly the training features."
        )
    if extra:
        # Keep only the required columns, ignore extras safely
        df = df[feature_names]
    return df[feature_names]

def validate_nonnegative_for_log(df: pd.DataFrame, eps: float = EPS):
    # Training used log(X + 1e-9). If any value <= -1e-9, log is undefined.
    bad_mask = (df <= -eps).any(axis=1)
    if bad_mask.any():
        bad_rows = np.where(bad_mask.values)[0][:5].tolist()
        raise ValueError(
            "Log-transform requires all features > -1e-9. "
            f"Found invalid values at rows (0-based): {bad_rows}..."
        )

# --------- Model prediction wrappers ---------
def predict_lambda_um(model_lam, X_raw: pd.DataFrame) -> np.ndarray:
    # Preprocess X: log-transform as in training, then invert y: exp
    validate_nonnegative_for_log(X_raw, EPS)
    X_log = np.log(X_raw + EPS)
    y_log_pred = model_lam.predict(X_log)
    lam_um = np.exp(y_log_pred)
    return lam_um

def predict_fwhm_um(model_fwhm, X_raw: pd.DataFrame) -> np.ndarray:
    # Raw-to-raw mapping
    fwhm_um = model_fwhm.predict(X_raw)
    # Safety: avoid zero/negatives for Q/FOM
    fwhm_um = np.maximum(fwhm_um, EPS)
    return fwhm_um

def estimate_sensitivity_um_per_RIU(model_lam, X_row: pd.Series, ri_feature: str, delta_ri: float) -> float:
    """
    Finite-difference sensitivity using λres predictions around the selected RI feature.
    Returns S in µm/RIU.
    """
    x0 = X_row.copy()
    x1 = X_row.copy()
    x0[ri_feature] = x0[ri_feature]
    x1[ri_feature] = x1[ri_feature] + delta_ri

    lam0 = float(predict_lambda_um(model_lam, pd.DataFrame([x0]))[0])
    lam1 = float(predict_lambda_um(model_lam, pd.DataFrame([x1]))[0])

    dlam = lam1 - lam0
    S_um_per_RIU = dlam / delta_ri
    return S_um_per_RIU

# --------- UI ---------
st.set_page_config(page_title="SPR Dual-Model Predictor", page_icon="🔬", layout="wide")

st.title(APP_TITLE)
st.caption(APP_DESC)

with st.expander("About this app", expanded=False):
    st.markdown(
        "- λres model: trained on log-transformed features and predicts log(λ). Inference applies log(X+1e-9) and exp(y).\n"
        "- FWHM model: trained on raw features and predicts FWHM in µm directly.\n"
        "- Q = λres / FWHM (dimensionless)\n"
        "- FOM = S / FWHM (conventional; units cancel). We also display values in nm-based units for clarity."
    )

# --------- Load models ---------
models_col1, models_col2 = st.columns(2)

with models_col1:
    lam_path = st.text_input(
        "Path to λres model (.pkl)",
        value=os.environ.get("LAM_MODEL_PATH", "models/best_xgboost_model_wl.pkl"),
    )
with models_col2:
    fwhm_path = st.text_input(
        "Path to FWHM model (.pkl)",
        value=os.environ.get("FWHM_MODEL_PATH", "models/best_xgb_model_fwhm.pkl"),
    )

load_btn = st.button("Load models")

if load_btn:
    try:
        lam_model = load_pickle_model(lam_path)
        fwhm_model = load_pickle_model(fwhm_path)
        st.success("Models loaded.")
        st.session_state["lam_model"] = lam_model
        st.session_state["fwhm_model"] = fwhm_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")

lam_model = st.session_state.get("lam_model")
fwhm_model = st.session_state.get("fwhm_model")

if not (lam_model and fwhm_model):
    st.info("Load your models to proceed.")
    st.stop()

# --------- Feature schema ---------
lam_features = get_model_feature_names(lam_model)
fwhm_features = get_model_feature_names(fwhm_model)

if not lam_features and not fwhm_features:
    st.error(
        "Could not infer feature names from models. "
        "Please re-train saving sklearn 'feature_names_in_' or upload a template dataset below."
    )

# Reconcile feature sets
feature_names = None
if lam_features and fwhm_features:
    if lam_features != fwhm_features:
        st.warning(
            "λres and FWHM models list different feature sets. "
            "Using the intersection to proceed; ensure consistency."
        )
        feature_names = [c for c in lam_features if c in set(fwhm_features)]
    else:
        feature_names = lam_features
elif lam_features:
    feature_names = lam_features
elif fwhm_features:
    feature_names = fwhm_features

# Optional: allow user to upload a template dataset to confirm/override columns
with st.expander("Optional: Upload a template dataset to confirm feature columns", expanded=False):
    tmp_file = st.file_uploader("Upload CSV/Excel with training features only", type=["csv", "xlsx"])
    if tmp_file is not None:
        try:
            tmp_df = read_any_table(tmp_file)
            if feature_names:
                # Align and adopt exact order
                tmp_df = align_features(tmp_df, feature_names)
            feature_names = list(tmp_df.columns)
            st.success(f"Feature schema set from uploaded file: {feature_names}")
        except Exception as e:
            st.error(f"Template parsing error: {e}")

if not feature_names:
    st.error("No feature schema available. Please provide models with feature_names_in_ or upload a template dataset.")
    st.stop()

# --------- Prediction mode ---------
st.markdown("---")
mode = st.radio("Prediction mode", ["Single input", "Batch file"], horizontal=True)

# Sensitivity options
with st.expander("Sensitivity (S) and FOM options", expanded=False):
    S_mode = st.radio(
        "How to compute Sensitivity S?",
        ["None (skip S & FOM)", "Finite difference on a chosen RI feature", "Manual S input"],
        index=1,
        help="S is typically dλ/dn. You can estimate it via small ΔRI or enter your own."
    )
    ri_feature = None
    delta_ri = None
    S_manual_nm_per_RIU = None

    if S_mode == "Finite difference on a chosen RI feature":
        ri_feature = st.selectbox("Select the RI feature to perturb for S", feature_names)
        delta_ri = st.number_input("ΔRI for sensitivity (typ. 1e-3 to 1e-4)", value=1e-3, min_value=1e-6, max_value=1e-1, step=1e-4, format="%.6f")
    elif S_mode == "Manual S input":
        S_manual_nm_per_RIU = st.number_input("Enter S (nm/RIU)", value=10000.0, min_value=0.0, step=100.0)

# --------- Single input ---------
if mode == "Single input":
    st.subheader("Enter feature values")
    cols = st.columns(min(4, len(feature_names)) or 1)

    # Build single-row DataFrame from user inputs
    single_vals = {}
    for i, feat in enumerate(feature_names):
        with cols[i % len(cols)]:
            # Conservative defaults; adjust as needed
            val = st.number_input(
                feat,
                value=1.0 if "n" in feat.lower() else 100.0,
                step=0.001 if "n" in feat.lower() else 1.0,
                format="%.6f" if "n" in feat.lower() else "%.3f",
            )
            single_vals[feat] = val

    run_btn = st.button("Predict")

    if run_btn:
        try:
            X_single = pd.DataFrame([single_vals], columns=feature_names)

            lam_um = predict_lambda_um(lam_model, X_single)
            fwhm_um = predict_fwhm_um(fwhm_model, X_single)

            lam_val = float(lam_um[0])
            fwhm_val = float(fwhm_um[0])

            # Core metrics
            Q = lam_val / max(fwhm_val, EPS)
            # Sensitivity and FOM
            S_um_per_RIU = None
            S_nm_per_RIU = None
            FOM = None

            if S_mode == "Finite difference on a chosen RI feature" and ri_feature and delta_ri:
                S_um_per_RIU = estimate_sensitivity_um_per_RIU(lam_model, X_single.iloc[0], ri_feature, delta_ri)
                S_nm_per_RIU = S_um_per_RIU * 1000.0
                FOM = S_nm_per_RIU / (fwhm_val * 1000.0)  # same as S_um / FWHM_um
            elif S_mode == "Manual S input" and S_manual_nm_per_RIU is not None:
                S_nm_per_RIU = float(S_manual_nm_per_RIU)
                FOM = S_nm_per_RIU / (fwhm_val * 1000.0)

            # Display
            st.markdown("### Results")
            res_cols = st.columns(3)
            res_cols[0].metric("λres (µm)", f"{lam_val:.6f}")
            res_cols[1].metric("FWHM (µm)", f"{fwhm_val:.6f}")
            res_cols[2].metric("Q-factor (λ/FWHM)", f"{Q:.3f}")

            if S_nm_per_RIU is not None:
                st.markdown("---")
                mcols = st.columns(3)
                mcols[0].metric("Sensitivity S (nm/RIU)", f"{S_nm_per_RIU:.2f}")
                mcols[1].metric("FOM (S/FWHM)", f"{FOM:.3f}" if FOM is not None else "—")
                mcols[2].metric("Sensitivity S (µm/RIU)", f"{(S_nm_per_RIU/1000.0):.6f}")

        except Exception as e:
            st.error(f"Inference error: {e}")

# --------- Batch file ---------
else:
    st.subheader("Upload a batch file with features")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if up is not None:
        try:
            df = read_any_table(up)
            # Align with training features
            df = align_features(df, feature_names)

            st.success(f"Loaded {df.shape[0]} rows. Running predictions...")
            lam_um = predict_lambda_um(lam_model, df)
            fwhm_um = predict_fwhm_um(fwhm_model, df)

            out = df.copy()
            out["λres (µm)"] = lam_um
            out["FWHM (µm)"] = fwhm_um
            out["Q-factor"] = out["λres (µm)"] / np.maximum(out["FWHM (µm)"], EPS)

            # Optional S and FOM per-row
            if S_mode == "Finite difference on a chosen RI feature" and ri_feature and delta_ri:
                if ri_feature not in out.columns:
                    st.warning(f"Chosen RI feature '{ri_feature}' not in uploaded data; skipping S/FOM.")
                else:
                    S_vals_um = []
                    for idx, row in out[feature_names].iterrows():
                        try:
                            S_um = estimate_sensitivity_um_per_RIU(lam_model, row, ri_feature, delta_ri)
                        except Exception:
                            S_um = np.nan
                        S_vals_um.append(S_um)
                    out["S (nm/RIU)"] = np.array(S_vals_um) * 1000.0
                    out["FOM (S/FWHM)"] = out["S (nm/RIU)"] / (out["FWHM (µm)"] * 1000.0)

            elif S_mode == "Manual S input" and S_manual_nm_per_RIU is not None:
                out["S (nm/RIU)"] = float(S_manual_nm_per_RIU)
                out["FOM (S/FWHM)"] = out["S (nm/RIU)"] / (out["FWHM (µm)"] * 1000.0)

            st.markdown("### Preview")
            st.dataframe(out.head(20), use_container_width=True)

            # Download
            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button(
                "Download results (CSV)",
                data=csv_buf.getvalue(),
                file_name="spr_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Failed to process file: {e}")

st.markdown("---")
st.caption("Remember: λres model uses log-transformed features and log-target inversion; FWHM model uses raw features.")