import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="SPR Performance Evaluation (2-layer)", layout="wide")
st.title("SPR Performance Evaluation (2-layer)")
st.caption("Enter analyte RI range and materials. Click Calculate (R-lam) then Evaluate Performance.")

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
    # Load models once; no UI for paths to keep it simple
    rlam_model = joblib.load("best_xgboost_model_wl.pkl")
    fwhm_model = joblib.load("best_xgb_model_fwhm.pkl")
    return rlam_model, fwhm_model

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
    # Training used log on features and target; invert afterward
    X_log = np.log(X_raw + EPS)
    y_log = rlam_model.predict(X_log)
    return np.exp(y_log)  # µm

def predict_fwhm_um(fwhm_model, X_raw: pd.DataFrame) -> np.ndarray:
    return fwhm_model.predict(X_raw)  # µm (raw-trained)

def sensitivity_nm_per_RIU(ri: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    lam_nm = lam_um * 1000.0
    dlam = lam_nm[1:] - lam_nm[:-1]
    dn = ri[1:] - ri[:-1]
    S = np.divide(dlam, dn, out=np.full_like(dlam, np.nan), where=dn!=0)
    return S  # aligned to left RI

def evaluate_metrics(ri: np.ndarray, lam_um: np.ndarray, fwhm_um: np.ndarray):
    S = sensitivity_nm_per_RIU(ri, lam_um)
    if len(S) == 0 or np.all(~np.isfinite(S)):
        return None
    idx_left = int(np.nanargmax(S))        # left index for S_max
    S_max = float(S[idx_left])             # nm/RIU
    ri_star = float(ri[idx_left])          # RI at left
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
# UI: Minimal inputs
# -----------------------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    ri_start = st.number_input("Start RI", value=1.33000, step=0.001, format="%.5f")
with col2:
    ri_step = st.number_input("Step size", value=0.00500, min_value=0.00001, step=0.001, format="%.5f")
with col3:
    ri_stop = st.number_input("Stop RI", value=1.41000, step=0.001, format="%.5f")

m1, m2 = st.columns(2)
with m1:
    mat1 = st.selectbox("Plasmonic Metal 1st Layer", ["Au", "Ag", "Cu", "C"], index=1)
with m2:
    mat2 = st.selectbox("Plasmonic Metal 2nd Layer", ["Au", "Ag", "Cu", "C"], index=0)

btn1, btn2 = st.columns(2)
calc_btn = btn1.button("Calculate (R-lam)", type="primary")
eval_btn = btn2.button("Evaluate Performance")

# Persistent storage for predictions
if "table" not in st.session_state:
    st.session_state.table = None
if "ri_values" not in st.session_state:
    st.session_state.ri_values = None
if "lam_um" not in st.session_state:
    st.session_state.lam_um = None
if "fwhm_um" not in st.session_state:
    st.session_state.fwhm_um = None

# -----------------------------
# Actions
# -----------------------------
if calc_btn:
    if ri_step <= 0 or ri_start >= ri_stop:
        st.error("Check RI range and step.")
        st.stop()
    try:
        rlam_model, fwhm_model = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    # Build RI grid inclusive of stop (with float tolerance)
    n_steps = int(np.floor((ri_stop - ri_start) / ri_step)) + 1
    ri_values = np.round(ri_start + np.arange(n_steps) * ri_step, 6)
    if ri_values[-1] < ri_stop - 1e-9:
        ri_values = np.append(ri_values, np.round(ri_stop, 6))

    X = build_features(ri_values, mat1, mat2)

    # Predict
    lam_um = predict_rlam_um(rlam_model, X)
    fwhm_um = predict_fwhm_um(fwhm_model, X)

    # Store
    st.session_state.ri_values = ri_values
    st.session_state.lam_um = lam_um
    st.session_state.fwhm_um = fwhm_um

    table = pd.DataFrame({
        "Analyte RI": ri_values,
        "Resonance Wavelength (µm)": lam_um
    })
    st.session_state.table = table

    st.subheader("R-lam results")
    st.dataframe(table, use_container_width=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (RI vs R-lam)", data=csv, file_name=f"R_lambda_{mat1}-{mat2}.csv", mime="text/csv")

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

            # Also show full table with FWHM and S aligned to left RI (prepend NaN for first row)
            S = metrics["S_all"]
            S_aligned = np.concatenate([[np.nan], S])
            full = pd.DataFrame({
                "Analyte RI": ri_values,
                "Resonance Wavelength (µm)": lam_um,
                "FWHM (µm)": fwhm_um,
                "Wavelength Sensitivity (nm/RIU)": S_aligned
            })
            st.subheader("Full results")
            st.dataframe(full, use_container_width=True)

            csv2 = full.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (full table)", data=csv2, file_name=f"full_metrics_{mat1}-{mat2}.csv", mime="text/csv")

st.markdown("---")
st.caption("Models: best_xgboost_model_wl.pkl (log-X, log-y), best_xgb_model_fwhm.pkl (raw). Geometry auto-set by material. Metrics per your specification.")

