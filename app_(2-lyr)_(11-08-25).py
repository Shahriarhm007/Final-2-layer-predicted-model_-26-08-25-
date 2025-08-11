# app.py
# SPR Performance Evaluation Tool — dual-model (λres + FWHM)
# Copy-paste ready. Adjust MODEL_CONFIG paths and feature_order to match your training.

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependency if you use .pkl/.joblib models
try:
    import joblib
except Exception:
    joblib = None

# Optional dependency for plotting
try:
    import altair as alt
except Exception:
    alt = None

# ----- App header -----
st.set_page_config(page_title="SPR Performance Evaluation Tool", layout="wide")
st.title("SPR Performance Evaluation Tool")
st.caption("Predict λres and FWHM with dual models; compute Sensitivity, Q-factor, and FOM")

# ----- Model + workflow configuration -----
# IMPORTANT: Update 'path' and 'feature_order' to reflect your exact trained models.
MODEL_CONFIG = {
    "lambda": {
        "path": "models/xgb_lambda.json",    # .json (XGBoost Booster) or .pkl/.joblib (sklearn/xgb)
        "target_transform": "log10",         # one of: None | 'log10' | 'log1p' | 'ln'
        "unit": "nm",
        "feature_order": [
            # REPLACE with exact training feature names and order, e.g.:
            # "n_analyte", "thickness_nm", "wavelength_start_nm", "incident_angle_deg"
        ],
        "training_ranges": {
            # Optional domain ranges (for warnings), e.g. "n_analyte": (1.33, 1.40)
            # "n_analyte": (1.33, 1.40),
            # "thickness_nm": (0.0, 80.0),
        },
        "version": "lambda_v1.0"
    },
    "fwhm": {
        "path": "models/xgb_fwhm.json",
        "target_transform": None,
        "unit": "nm",
        "feature_order": [
            # REPLACE with exact training feature names and order
            # If FWHM used the same features as λres, copy that list.
        ],
        "training_ranges": {
            # Optional domain ranges
        },
        "version": "fwhm_v1.0"
    },
    "fwhm_floor_nm": 1e-6  # numerical safety for Q and FOM
}

# ----- Utilities -----
def invert_transform(y_pred: np.ndarray | float, kind: str | None):
    if kind is None:
        return y_pred
    if kind == "log10":
        return np.power(10.0, y_pred)
    if kind == "log1p":
        return np.expm1(y_pred)
    if kind == "ln":
        return np.exp(y_pred)
    raise ValueError(f"Unknown target_transform: {kind}")

def align_features(df_in: pd.DataFrame, expected_order: list[str]):
    """Align columns to expected order. If expected_order empty, keep current order."""
    df = df_in.copy()
    if not expected_order:
        return df
    # add missing with zeros
    for col in expected_order:
        if col not in df.columns:
            df[col] = 0.0
    # drop extras
    extras = [c for c in df.columns if c not in expected_order]
    if extras:
        st.warning(f"Dropping unexpected features: {extras}")
        df = df.drop(columns=extras)
    # reorder
    df = df[expected_order]
    return df

def compute_metrics(lambda_nm_1: float, lambda_nm_2: float, n1: float, n2: float, fwhm_nm: float, fwhm_floor=1e-6):
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
    for k, v in (ranges or {}).items():
        if k in x:
            lo, hi = v
            if not (lo <= x[k] <= hi):
                msgs.append(f"{k}={x[k]} outside training range [{lo}, {hi}]")
    if msgs:
        st.warning("Extrapolation warning: " + "; ".join(msgs))

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Loads either:
    - XGBoost Booster from .json
    - Sklearn/XGBRegressor from .pkl/.joblib
    Returns (predict_fn, model_type_str)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if p.suffix in {".pkl", ".joblib"}:
        if joblib is None:
            raise RuntimeError("joblib not available to load .pkl/.joblib models.")
        model = joblib.load(p)

        def predict_fn(X_df: pd.DataFrame):
            return np.asarray(model.predict(X_df)).reshape(-1)

        mtype = getattr(model, "__class__", type("M", (), {})).__name__
        return predict_fn, f"sklearn:{mtype}"

    if p.suffix == ".json":
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(p))

        def predict_fn(X_df: pd.DataFrame):
            dmat = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
            return booster.predict(dmat).reshape(-1)

        return predict_fn, "xgboost:Booster"

    raise ValueError(f"Unsupported model file type: {p.suffix}")

def safe_number_input(label, value, step, fmt=None, min_val=None, max_val=None, key=None):
    kwargs = {}
    if fmt:
        kwargs["format"] = fmt
    if min_val is not None:
        kwargs["min_value"] = min_val
    if max_val is not None:
        kwargs["max_value"] = max_val
    return st.number_input(label, value=value, step=step, key=key, **kwargs)

# ----- Sidebar: model paths and transforms -----
st.sidebar.header("Model settings")
lambda_path = st.sidebar.text_input("λres model path", value=MODEL_CONFIG["lambda"]["path"])
fwhm_path = st.sidebar.text_input("FWHM model path", value=MODEL_CONFIG["fwhm"]["path"])

lambda_transform = st.sidebar.selectbox(
    "λres target transform",
    options=[None, "log10", "ln", "log1p"],
    index=[None, "log10", "ln", "log1p"].index(MODEL_CONFIG["lambda"]["target_transform"]),
    help="Select the transform that was applied to λres during training."
)
fwhm_transform = st.sidebar.selectbox(
    "FWHM target transform",
    options=[None, "log10", "ln", "log1p"],
    index=[None, "log10", "ln", "log1p"].index(MODEL_CONFIG["fwhm"]["target_transform"]),
    help="Usually None; change only if you trained FWHM with a log transform."
)

fwhm_floor = st.sidebar.number_input("FWHM safety floor (nm)", value=float(MODEL_CONFIG["fwhm_floor_nm"]), step=1e-6, format="%.6f")

# Attempt to load models and report status
with st.sidebar.expander("Load status", expanded=False):
    lambda_ok = True
    fwhm_ok = True
    lambda_predict_fn = None
    fwhm_predict_fn = None
    lambda_model_type = ""
    fwhm_model_type = ""

    try:
        lambda_predict_fn, lambda_model_type = load_model(lambda_path)
        st.success(f"λres model loaded ({lambda_model_type})")
    except Exception as e:
        lambda_ok = False
        st.error(f"λres model: {e}")

    try:
        fwhm_predict_fn, fwhm_model_type = load_model(fwhm_path)
        st.success(f"FWHM model loaded ({fwhm_model_type})")
    except Exception as e:
        fwhm_ok = False
        st.error(f"FWHM model: {e}")

# ----- Sidebar: refractive indices -----
st.sidebar.header("Refractive indices")
n1 = st.sidebar.number_input("n₁ (RIU)", value=1.33, step=0.0001, format="%.4f")
n2 = st.sidebar.number_input("n₂ (RIU)", value=1.34, step=0.0001, format="%.4f")
if n1 == n2:
    st.sidebar.warning("n₁ equals n₂; sensitivity S cannot be computed.")

# ----- Sidebar: feature inputs -----
# Define your feature input fields (excluding n_analyte; it is set from n1/n2)
# Replace these with the exact features you used in training.
FEATURE_UI = [
    # name, label, default, step, (min, max), fmt
    ("thickness_nm", "Metal layer thickness (nm)", 50.0, 0.1, (0.0, 500.0), None),
    ("wavelength_start_nm", "Wavelength start (nm)", 400.0, 1.0, (200.0, 2000.0), None),
    ("wavelength_end_nm", "Wavelength end (nm)", 800.0, 1.0, (200.0, 3000.0), None),
    ("incident_angle_deg", "Incident angle (deg)", 70.0, 0.1, (0.0, 90.0), None),
    # Add/remove to match your training schema precisely
]

st.sidebar.header("Other features")
feature_values = {}
for name, label, default, step, limits, fmt in FEATURE_UI:
    min_v, max_v = limits if limits else (None, None)
    feature_values[name] = safe_number_input(label, value=default, step=step, fmt=fmt, min_val=min_v, max_val=max_v, key=f"feat_{name}")

# Build model input dicts
inputs_n1 = {"n_analyte": float(n1), **{k: float(v) for k, v in feature_values.items()}}
inputs_n2 = {"n_analyte": float(n2), **{k: float(v) for k, v in feature_values.items()}}

# ----- Main area: instructions and feature alignment info -----
with st.expander("Setup notes", expanded=False):
    st.markdown(
        "- Ensure 'Model settings' paths point to your trained models.\n"
        "- Edit MODEL_CONFIG['lambda']['feature_order'] and MODEL_CONFIG['fwhm']['feature_order'] to match training order.\n"
        "- The sidebar features must exactly match the training features (names and units).\n"
        "- Metrics use: S = Δλres/Δn (nm/RIU), Q = λres/FWHM, FOM = S/FWHM (RIU⁻¹)."
    )

# Determine expected feature order
expected_lambda_order = MODEL_CONFIG["lambda"]["feature_order"]
expected_fwhm_order = MODEL_CONFIG["fwhm"]["feature_order"]

# Show alignment status
with st.expander("Feature alignment", expanded=False):
    st.write("λres expected order:", expected_lambda_order if expected_lambda_order else "(not specified; using current input order)")
    st.write("FWHM expected order:", expected_fwhm_order if expected_fwhm_order else "(not specified; using current input order)")
    if expected_lambda_order and "n_analyte" not in expected_lambda_order:
        st.warning("λres feature_order does not include 'n_analyte' — add it if your model used it.")
    if expected_fwhm_order and "n_analyte" not in expected_fwhm_order:
        st.info("FWHM often does not use n_analyte; that's OK if your training matched.")

# ----- Predict button -----
run = st.button("Predict", type="primary", use_container_width=True)

# ----- Inference -----
def run_inference(_inputs_n1: dict, _inputs_n2: dict):
    # 1) DataFrames
    X1_l = pd.DataFrame([_inputs_n1])
    X2_l = pd.DataFrame([_inputs_n2])
    X1_f = pd.DataFrame([_inputs_n1])  # if FWHM uses the same features
    # If FWHM uses a different schema, build X1_f accordingly.

    # 2) Align features
    X1_l = align_features(X1_l, expected_lambda_order)
    X2_l = align_features(X2_l, expected_lambda_order)
    X1_f = align_features(X1_f, expected_fwhm_order)

    # 3) Predict raw targets
    y1_l = float(lambda_predict_fn(X1_l)[0])
    y2_l = float(lambda_predict_fn(X2_l)[0])
    y_f = float(fwhm_predict_fn(X1_f)[0])

    # 4) Inverse transforms
    lambda1 = float(invert_transform(y1_l, lambda_transform))
    lambda2 = float(invert_transform(y2_l, lambda_transform))
    fwhm = float(invert_transform(y_f, fwhm_transform))

    # 5) Numerical safety checks
    errors = []
    if not np.isfinite(lambda1) or lambda1 <= 0:
        errors.append("Invalid λres prediction for n₁ after inverse transform.")
    if not np.isfinite(lambda2) or lambda2 <= 0:
        errors.append("Invalid λres prediction for n₂ after inverse transform.")
    if not np.isfinite(fwhm) or fwhm <= 0:
        st.warning("FWHM prediction ≤ 0; applying safety floor.")
        fwhm = float(fwhm_floor)

    # 6) Metrics
    S, Q, FOM, msg = compute_metrics(lambda1, lambda2, _inputs_n1["n_analyte"], _inputs_n2["n_analyte"], fwhm, fwhm_floor)

    return {
        "lambda_nm_n1": lambda1,
        "lambda_nm_n2": lambda2,
        "fwhm_nm": fwhm,
        "S_nm_per_RIU": S,
        "Q": Q,
        "FOM_per_RIU": FOM,
        "errors": errors,
        "warning": msg
    }

results = None
if run:
    if not (lambda_ok and fwhm_ok):
        st.error("Load both models successfully before predicting.")
    else:
        # Domain warnings (optional)
        warn_out_of_domain(inputs_n1, MODEL_CONFIG["lambda"]["training_ranges"])
        warn_out_of_domain(inputs_n2, MODEL_CONFIG["lambda"]["training_ranges"])
        warn_out_of_domain(inputs_n1, MODEL_CONFIG["fwhm"]["training_ranges"])

        results = run_inference(inputs_n1, inputs_n2)

        # Report any critical errors
        if results["errors"]:
            for e in results["errors"]:
                st.error(e)

        if results["warning"]:
            st.warning(results["warning"])

        # ----- Display results -----
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("λres @ n₁ (nm)", None if np.isnan(results["lambda_nm_n1"]) else f"{results['lambda_nm_n1']:.3f}")
            st.metric("λres @ n₂ (nm)", None if np.isnan(results["lambda_nm_n2"]) else f"{results['lambda_nm_n2']:.3f}")
        with col2:
            st.metric("FWHM (nm)", None if np.isnan(results["fwhm_nm"]) else f"{results['fwhm_nm']:.3f}")
            st.metric("Q-factor", None if np.isnan(results["Q"]) else f"{results['Q']:.3f}")
        with col3:
            st.metric("Sensitivity S (nm/RIU)", None if np.isnan(results["S_nm_per_RIU"]) else f"{results['S_nm_per_RIU']:.3f}")
            st.metric("FOM (RIU⁻¹)", None if np.isnan(results["FOM_per_RIU"]) else f"{results['FOM_per_RIU']:.3f}")

        # Simple λ vs n plot
        if alt is not None and np.isfinite(results["lambda_nm_n1"]) and np.isfinite(results["lambda_nm_n2"]):
            df_plot = pd.DataFrame({
                "n (RIU)": [float(n1), float(n2)],
                "λres (nm)": [results["lambda_nm_n1"], results["lambda_nm_n2"]]
            })
            chart = (
                alt.Chart(df_plot)
                .mark_line(point=True)
                .encode(x="n (RIU):Q", y="λres (nm):Q")
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

        # ----- Export CSV -----
        meta = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "lambda_model_path": lambda_path,
            "lambda_model_type": lambda_model_type,
            "lambda_transform": str(lambda_transform),
            "lambda_unit": MODEL_CONFIG["lambda"]["unit"],
            "lambda_version": MODEL_CONFIG["lambda"]["version"],
            "fwhm_model_path": fwhm_path,
            "fwhm_model_type": fwhm_model_type,
            "fwhm_transform": str(fwhm_transform),
            "fwhm_unit": MODEL_CONFIG["fwhm"]["unit"],
            "fwhm_version": MODEL_CONFIG["fwhm"]["version"],
            "fwhm_floor_nm": fwhm_floor,
        }

        export_row = {
            **{f"input_{k}_n1": v for k, v in inputs_n1.items()},
            **{f"input_{k}_n2": v for k, v in inputs_n2.items()},
            **{
                "pred_lambda_nm_n1": results["lambda_nm_n1"],
                "pred_lambda_nm_n2": results["lambda_nm_n2"],
                "pred_fwhm_nm": results["fwhm_nm"],
                "metric_S_nm_per_RIU": results["S_nm_per_RIU"],
                "metric_Q": results["Q"],
                "metric_FOM_per_RIU": results["FOM_per_RIU"],
            },
            **{f"meta_{k}": v for k, v in meta.items()}
        }
        df_export = pd.DataFrame([export_row])

        csv_buf = io.StringIO()
        df_export.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv_buf.getvalue(),
            file_name=f"spr_results_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----- Footer tips -----
st.markdown("---")
st.markdown(
    "**Tips**: "
    "1) Match feature names and units to your training. "
    "2) Set the correct λres transform (log10/ln/log1p/None). "
    "3) Use training ranges to flag extrapolation. "
    "4) Keep FWHM > 0 with a small safety floor to avoid divide-by-zero."
)
