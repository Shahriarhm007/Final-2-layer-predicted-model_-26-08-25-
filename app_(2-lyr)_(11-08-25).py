import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="SPR Performance Evaluation (2-layer)", layout="wide")
st.title("SPR Performance Evaluation (2-layer)")
st.caption("Select materials. Click **Calculate (R-lam)** then **Evaluate Performance**.")

# -----------------------------
# Constants
# -----------------------------
EPS = 1e-9
MATERIAL_CODE = {"Au": 1, "Ag": 2, "Cu": 3, "C": 4}
FEATURE_COLUMNS = [
    "Analyte RI",
    "Material of 1st layer (RIU)",
    "Material of 2nd layer (RIU)",
    "thickness of 1st layer (µm)",
    "thickness of 2nd layer (µm)",
    "Distance bwtn core surface and 2nd layer (µm)",
]

# -----------------------------
# Helper functions
# -----------------------------
def thickness_um(material: str) -> float:
    """Return thickness in µm for selected material."""
    return 0.035 if material in ("Au", "Ag", "Cu") else 0.00034

@st.cache_resource
def load_models():
    """Safely load the ML models."""
    model_files = {
        "R-lam": "best_xgboost_model_wl.pkl",
        "FWHM": "best_poly_pipeline_fwhm.pkl"
    }
    models = {}
    for key, filename in model_files.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file '{filename}' not found in project folder.")
        try:
            models[key] = joblib.load(filename)
        except Exception as e:
            raise RuntimeError(f"Failed to load {key} model: {e}")
    return models["R-lam"], models["FWHM"]

def get_fixed_ri_values(mat1: str, mat2: str) -> np.ndarray:
    """Return pre-defined RI values depending on material combination."""
    mset = {mat1.lower(), mat2.lower()}
    if "c" in mset and ("au" in mset or "ag" in mset):
        return np.array([1.33, 1.35, 1.36, 1.37, 1.375, 1.38])
    elif "c" in mset and "cu" in mset:
        return np.array([1.33, 1.35, 1.36, 1.37, 1.38, 1.385, 1.39])
    elif ("au" in mset and "ag" in mset) or (mat1 == mat2 == "Au") or (mat1 == mat2 == "Ag"):
        return np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41])
    elif ("cu" in mset and ("au" in mset or "ag" in mset)) or (mat1 == mat2 == "Cu"):
        return np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])
    else:
        return np.array([1.33, 1.35, 1.37])  # fallback safe set

def build_features(ri_values, mat1, mat2) -> pd.DataFrame:
    """Build features dataframe for model input."""
    t1 = thickness_um(mat1)
    t2 = thickness_um(mat2)
    dist = 1.05 + t1
    df = pd.DataFrame({
        "Analyte RI": ri_values,
        "Material of 1st layer (RIU)": MATERIAL_CODE[mat1],
        "Material of 2nd layer (RIU)": MATERIAL_CODE[mat2],
        "thickness of 1st layer (µm)": t1,
        "thickness of 2nd layer (µm)": t2,
        "Distance bwtn core surface and 2nd layer (µm)": dist,
    })
    return df[FEATURE_COLUMNS]

def predict_rlam_um(rlam_model, X_raw: pd.DataFrame) -> np.ndarray:
    """Predict resonance wavelength in µm."""
    try:
        X_log = np.log(X_raw + EPS)
        y_log = rlam_model.predict(X_log)
        return np.exp(y_log)
    except Exception as e:
        st.error(f"R-lam prediction failed: {e}")
        return np.full(len(X_raw), np.nan)

def predict_fwhm_um(fwhm_model, X_raw: pd.DataFrame) -> np.ndarray:
    """Predict FWHM in µm."""
    try:
        return fwhm_model.predict(X_raw)
    except Exception as e:
        st.error(f"FWHM prediction failed: {e}")
        return np.full(len(X_raw), np.nan)

def sensitivity_nm_per_RIU(ri: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    """Compute wavelength sensitivity (nm/RIU)."""
    lam_nm = lam_um * 1000.0
    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    return np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn!=0)

def evaluate_metrics(ri: np.ndarray, lam_um: np.ndarray, fwhm_um: np.ndarray):
    """Evaluate S, Q, and FOM metrics."""
    S = sensitivity_nm_per_RIU(ri, lam_um)
    if len(S) == 0 or np.all(~np.isfinite(S)):
        return None
    idx_left = int(np.nanargmax(S))
    S_max = float(S[idx_left])
    ri_star = float(ri[idx_left])
    lam_nm_left = float(lam_um[idx_left] * 1000.0)
    fwhm_nm_left = float(fwhm_um[idx_left] * 1000.0)
    Q = lam_nm_left / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    FOM = S_max / fwhm_nm_left if fwhm_nm_left > 0 else np.nan
    return {
        "S_all": S,
        "S_max": S_max,
        "ri_at_Smax": ri_star,
        "lambda_nm_at_Smax_left": lam_nm_left,
        "fwhm_nm_at_Smax_left": fwhm_nm_left,
        "Q": Q,
        "FOM": FOM,
        "idx_left": idx_left
    }

# -----------------------------
# UI: Material selection
# -----------------------------
m1, m2 = st.columns(2)
with m1:
    mat1 = st.selectbox("Plasmonic Metal 1st Layer", list(MATERIAL_CODE.keys()), index=1)
with m2:
    mat2 = st.selectbox("Plasmonic Metal 2nd Layer", list(MATERIAL_CODE.keys()), index=0)

btn1, btn2 = st.columns(2)
calc_btn = btn1.button("Calculate (R-lam)", type="primary")
eval_btn = btn2.button("Evaluate Performance")

# -----------------------------
# Session State
# -----------------------------
for key in ["table", "ri_values", "lam_um", "fwhm_um"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    try:
        rlam_model, fwhm_model = load_models()
    except Exception as e:
        st.error(str(e))
        st.stop()

    ri_values = get_fixed_ri_values(mat1, mat2)
    X = build_features(ri_values, mat1, mat2)

    lam_um = predict_rlam_um(rlam_model, X)
    fwhm_um = predict_fwhm_um(fwhm_model, X)

    st.session_state.ri_values = ri_values
    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um

    table = pd.DataFrame({
        "Analyte RI": ri_values,
        "Resonance Wavelength (µm)": lam_um,
        "FWHM (µm)": fwhm_um
    })
    st.session_state.table = table

    st.subheader("Calculated R-lam & FWHM results")
    st.dataframe(table, use_container_width=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (RI vs R-lam & FWHM)",
        data=csv,
        file_name=f"R_lambda_FWHM_{mat1}-{mat2}.csv",
        mime="text/csv"
    )

if eval_btn:
    if st.session_state.table is None:
        st.warning("Please click **Calculate (R-lam)** first.")
    else:
        ri_values = st.session_state.ri_values
        lam_um = st.session_state.lam_um
        fwhm_um = st.session_state.fwhm_um

        metrics = evaluate_metrics(ri_values, lam_um, fwhm_um)
        if metrics is None:
            st.error("Unable to compute sensitivity. Check RI step and predictions.")
        else:
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Model", f"{mat1}-{mat2}")
            colB.metric("Max. Wavelength Sensitivity", f"{metrics['S_max']:.3f} nm/RIU")
            colC.metric("Q-factor", f"{metrics['Q']:.3f}")
            colD.metric("FOM", f"{metrics['FOM']:.6f}")

            st.caption(
                f"S_max at RI={metrics['ri_at_Smax']:.5f} "
                f"(λ_left={metrics['lambda_nm_at_Smax_left']:.3f} nm, "
                f"FWHM_left={metrics['fwhm_nm_at_Smax_left']:.3f} nm)"
            )
