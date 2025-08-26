import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="SPR Performance Evaluation (2-layer)", layout="wide")
st.title("SPR Performance Evaluation (2-layer)")
st.caption("Select materials. Click Calculate (R-lam) then Evaluate Performance.")

# -----------------------------
# Constants and helpers
# -----------------------------
EPS = 1e-9
MATERIAL_CODE = {"Au": 1, "Ag": 2, "Cu": 3, "C": 4}

def thickness_um(material: str) -> float:
    return 0.035 if material in ("Au", "Ag", "Cu") else 0.00034

FEATURE_COLUMNS = [
    "Analyte RI",
    "Material of 1st layer (RIU)",
    "Material of 2nd layer (RIU)",
    "thickness of 1st layer (µm)",
    "thickness of 2nd layer (µm)",
    "Distance bwtn core surface and 2nd layer (µm)",
]

@st.cache_resource
def load_models():
    # XGBoost model (for R-lam, trained on log λ)
    rlam_model = joblib.load("best_xgboost_model_wl.pkl")
    # Polynomial Regression pipeline (for FWHM, trained on log FWHM)
    fwhm_model = joblib.load("best_poly_pipeline_fwhm.pkl")
    return rlam_model, fwhm_model

def get_fixed_ri_values(mat1: str, mat2: str) -> np.ndarray:
    mset = {mat1.lower(), mat2.lower()}
    if "c" in mset and ("au" in mset or "ag" in mset):
        return np.array([1.33, 1.35, 1.36, 1.37, 1.375, 1.38])
    elif "c" in mset and "cu" in mset:
        return np.array([1.33, 1.35, 1.36, 1.37, 1.38, 1.385, 1.39])
    elif ("au" in mset and "ag" in mset) or (mat1 == "Au" and mat2 == "Au") or (mat1 == "Ag" and mat2 == "Ag"):
        return np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41])
    elif ("cu" in mset and ("au" in mset or "ag" in mset)) or (mat1 == "Cu" and mat2 == "Cu"):
        return np.array([1.33, 1.35, 1.37, 1.39, 1.40, 1.405, 1.41, 1.415, 1.42])
    else:
        return np.array([1.33, 1.35, 1.37])  # fallback

def build_features(ri_values, mat1, mat2) -> pd.DataFrame:
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
    X_log = np.log(X_raw + EPS)
    y_log = rlam_model.predict(X_log)
    y = np.exp(y_log)
    return np.maximum(y, EPS)

def predict_fwhm_um(fwhm_model, X_raw: pd.DataFrame) -> np.ndarray:
    y_log = fwhm_model.predict(X_raw)   # log(FWHM)
    y = np.exp(y_log)                   # inverse log-transform
    return np.maximum(y, EPS)

def sensitivity_nm_per_RIU(ri: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    lam_nm = lam_um * 1000.0
    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    S = np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn!=0)
    return S

def evaluate_metrics(ri: np.ndarray, lam_um: np.ndarray, fwhm_um: np.ndarray):
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
# UI: Material selection only
# -----------------------------
m1, m2 = st.columns(2)
with m1:
    mat1 = st.selectbox("Plasmonic Metal 1st Layer", ["Au", "Ag", "Cu", "C"], index=1)
with m2:
    mat2 = st.selectbox("Plasmonic Metal 2nd Layer", ["Au", "Ag", "Cu", "C"], index=0)

btn1, btn2 = st.columns(2)
calc_btn = btn1.button("Calculate (R-lam)", type="primary")
eval_btn = btn2.button("Evaluate Performance")

if "table" not in st.session_state:
    st.session_state.table = None
if "ri_values" not in st.session_state:
    st.session_state.ri_values = None
if "lam_um" not in st.session_state:
    st.session_state.lam_um = None
if "fwhm_um" not in st.session_state:
    st.session_state.fwhm_um = None
if "current_materials" not in st.session_state:
    st.session_state.current_materials = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    try:
        rlam_model, fwhm_model = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    ri_values = get_fixed_ri_values(mat1, mat2)
    X = build_features(ri_values, mat1, mat2)

    lam_um = predict_rlam_um(rlam_model, X)
    fwhm_um = predict_fwhm_um(fwhm_model, X)

    # Ensure 1D arrays
    ri_values = np.asarray(ri_values).ravel()
    lam_um = np.asarray(lam_um).ravel()
    fwhm_um = np.asarray(fwhm_um).ravel()

    st.session_state.ri_values = ri_values
    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um
    st.session_state.current_materials = f"{mat1}-{mat2}"

    # Check lengths
    if not (len(ri_values) == len(lam_um) == len(fwhm_um)):
        st.error(f"Shape mismatch: RI={len(ri_values)}, λ={len(lam_um)}, FWHM={len(fwhm_um)}")
    else:
        table = pd.DataFrame({
            "Analyte RI": ri_values,
            "Resonance Wavelength (µm)": lam_um,
            "FWHM (µm)": fwhm_um
        })
        st.session_state.table = table

if eval_btn:
    if st.session_state.table is None:
        st.warning("Please click 'Calculate (R-lam)' first.")
    else:
        ri_values = st.session_state.ri_values
        lam_um = st.session_state.lam_um
        fwhm_um = st.session_state.fwhm_um

        metrics = evaluate_metrics(ri_values, lam_um, fwhm_um)
        if metrics is None:
            st.error("Unable to compute sensitivity. Check RI step and predictions.")
        else:
            # Store metrics in session state
            st.session_state.metrics = metrics

# -----------------------------
# Display metrics if they exist
# -----------------------------
if st.session_state.metrics is not None:
    metrics = st.session_state.metrics
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Model", st.session_state.current_materials)
    colB.metric("Max. Wavelength Sensitivity", f"{metrics['S_max']:.3f} nm/RIU")
    colC.metric("Q-factor", f"{metrics['Q']:.3f}")
    colD.metric("FOM", f"{metrics['FOM']:.6f}")

    st.caption(
        f"S_max at RI={metrics['ri_at_Smax']:.5f} "
        f"(λ_left={metrics['lambda_nm_at_Smax_left']:.3f} nm, "
        f"FWHM_left={metrics['fwhm_nm_at_Smax_left']:.3f} nm)"
    )

# -----------------------------
# Display table if it exists (moved outside button conditions)
# -----------------------------
if st.session_state.table is not None:
    st.subheader("R-lam results")
    st.dataframe(st.session_state.table, use_container_width=True)

    csv = st.session_state.table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (RI vs R-lam + FWHM)",
        data=csv,
        file_name=f"R_lambda_{st.session_state.current_materials}.csv",
        mime="text/csv"
    )
