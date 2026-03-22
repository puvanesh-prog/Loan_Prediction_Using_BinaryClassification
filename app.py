"""
╔══════════════════════════════════════════════════════════════╗
║   Credit Risk Prediction App — RBL Bank                     ║
║   Model  : Tuned XGBoost                                    ║
║   Target : Bad_label  (0 = Good Customer, 1 = Defaulter)    ║
╚══════════════════════════════════════════════════════════════╝

Folder structure expected on the server:
    app.py
    models/
        credit_risk_xgboost_model.pkl
        imputer.pkl
        scaler.pkl
        label_encoders.pkl
    data/                          ← optional, only for reference
        Cust_Account.csv
        Cust_demographics.csv
        Cust_enquiry.csv
"""

import io
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor | RBL Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DARK THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── base ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: #0f1117 !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stHeader"]       { background-color: #0f1117 !important; }
    [data-testid="stSidebar"]      { background-color: #1a1d2e !important; border-right: 1px solid #2d3748; }
    [data-testid="stSidebar"] *    { color: #e2e8f0 !important; }

    /* ── cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e2235, #252840);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label { font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; }

    /* ── result banners ── */
    .risk-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #fecaca;
        margin: 8px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #bbf7d0;
        margin: 8px 0;
    }

    /* ── section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7dd3fc;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 8px;
        margin: 18px 0 14px;
    }

    /* ── buttons ── */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    [data-testid="stButton"] > button:hover { opacity: 0.88; }

    /* ── file uploader ── */
    [data-testid="stFileUploader"] {
        background: #1a1d2e !important;
        border: 1px dashed #4b5563 !important;
        border-radius: 10px !important;
    }

    /* ── dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ── info / warning boxes ── */
    [data-testid="stAlert"] { border-radius: 10px; }

    /* ── progress bar ── */
    [data-testid="stProgress"] > div > div { background: #2563eb; border-radius: 4px; }

    /* hide Streamlit default footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL LOADING  (web-compatible: relative to app.py, not local drive)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # directory where app.py lives
MODEL_DIR  = BASE_DIR / "models"

@st.cache_resource(show_spinner="Loading model artefacts…")
def load_artefacts():
    model          = joblib.load(MODEL_DIR / "credit_risk_xgboost_model.pkl")
    imputer        = joblib.load(MODEL_DIR / "imputer.pkl")
    scaler         = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
    return model, imputer, scaler, label_encoders

try:
    model, imputer, scaler, label_encoders = load_artefacts()
    artefacts_ok = True
except FileNotFoundError as e:
    artefacts_ok = False
    artefact_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING  (mirrors notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
METRO_CITIES = [
    "Mumbai / Navi Mumbai / Thane", "Bengaluru",
    "New Delhi", "Pune", "Gurgaon",
]

COLUMNS_TO_DROP = [
    "dt_opened", "entry_time", "customer_no", "feature_2",
    "feature_23", "feature_24", "feature_25", "feature_27",
    "feature_49", "feature_51", "feature_57", "feature_58",
    "feature_74", "feature_79", "last_enquiry_dt",
]

def calculate_delinquency_score(history_str):
    if pd.isna(history_str):
        return 0
    clean = str(history_str).replace('"', "").replace("'", "").replace(" ", "")
    return clean.count("XXX") + clean.count("90") + clean.count("120")


def engineer_features(dem_df, acc_df, enq_df):
    """Replicate the exact pipeline from the notebook."""

    # ── account features ──────────────────────────────────────────────────
    acc = acc_df.copy()
    acc["high_credit_amt"]      = pd.to_numeric(acc["high_credit_amt"],      errors="coerce")
    acc["cur_balance_amt"]      = pd.to_numeric(acc["cur_balance_amt"],       errors="coerce")
    acc["amt_past_due"]         = pd.to_numeric(acc["amt_past_due"],          errors="coerce")
    acc["actualpaymentamount"]  = pd.to_numeric(acc["actualpaymentamount"],   errors="coerce")

    account_features = acc.groupby("customer_no").agg(
        num_accounts        =("acct_type",            "count"),
        total_high_credit   =("high_credit_amt",       "sum"),
        total_balance       =("cur_balance_amt",        "sum"),
        max_past_due        =("amt_past_due",           "max"),
        total_past_due      =("amt_past_due",           "sum"),
        total_payment_amt   =("actualpaymentamount",    "sum"),
        num_closed_accounts =("closed_dt",             lambda x: x.notna().sum()),
    ).reset_index()

    # ── delinquency ───────────────────────────────────────────────────────
    acc["delinquency_score"] = acc["paymenthistory1"].apply(calculate_delinquency_score)
    delinquency_features = acc.groupby("customer_no").agg(
        total_delinquency=("delinquency_score", "sum"),
        max_delinquency  =("delinquency_score", "max"),
    ).reset_index()

    # ── enquiry features ──────────────────────────────────────────────────
    enq = enq_df.copy()
    enq["enq_amt"] = pd.to_numeric(enq["enq_amt"], errors="coerce")
    enquiry_features = enq.groupby("customer_no").agg(
        total_enquiries   =("enquiry_dt",  "count"),
        total_enq_amount  =("enq_amt",     "sum"),
        max_enq_amount    =("enq_amt",     "max"),
        avg_enq_amount    =("enq_amt",     "mean"),
        unique_enq_purpose=("enq_purpose", "nunique"),
    ).reset_index()

    enq["enquiry_dt"] = pd.to_datetime(enq["enquiry_dt"], dayfirst=True, errors="coerce")
    latest_enq = enq.groupby("customer_no")["enquiry_dt"].max().reset_index()
    latest_enq.columns = ["customer_no", "last_enquiry_dt"]
    latest_enq["days_since_last_enquiry"] = (
        pd.Timestamp("2016-01-01") - latest_enq["last_enquiry_dt"]
    ).dt.days
    enquiry_features = enquiry_features.merge(
        latest_enq[["customer_no", "days_since_last_enquiry"]], on="customer_no", how="left"
    )

    # ── demographics derived ──────────────────────────────────────────────
    dem = dem_df.copy()
    dem["feature_24"] = pd.to_datetime(dem["feature_24"], dayfirst=True, errors="coerce")
    dem["customer_age"] = (
        pd.Timestamp("2016-01-01") - dem["feature_24"]
    ).dt.days // 365

    dem["feature_38"] = pd.to_numeric(
        dem["feature_38"].astype(str).str.replace(",", ""), errors="coerce"
    )
    dem["feature_9"] = pd.to_numeric(
        dem["feature_9"].astype(str).str.replace(",", ""), errors="coerce"
    )
    dem["income_to_credit_ratio"] = dem["feature_38"] / (dem["feature_9"] + 1)
    dem["is_metro_city"] = dem["feature_31"].isin(METRO_CITIES).astype(int)
    dem["credit_score"]  = pd.to_numeric(dem["feature_3"], errors="coerce")

    # ── merge → master ────────────────────────────────────────────────────
    master = dem.merge(account_features,    on="customer_no", how="left")
    master = master.merge(delinquency_features, on="customer_no", how="left")
    master = master.merge(enquiry_features,     on="customer_no", how="left")

    return master


def preprocess_and_predict(master_df):
    """Encode → align → impute → scale → predict. Returns result DataFrame."""
    # Drop same columns as notebook
    drop_cols = [c for c in COLUMNS_TO_DROP if c in master_df.columns]
    if "Bad_label" in master_df.columns:
        drop_cols.append("Bad_label")
    X = master_df.drop(columns=drop_cols)

    # Encode categoricals using saved encoders
    X_enc = X.copy()
    for col, enc in label_encoders.items():
        if col in X_enc.columns:
            # Handle unseen categories gracefully
            known = set(enc.classes_)
            X_enc[col] = X_enc[col].astype(str).apply(
                lambda v: v if v in known else enc.classes_[0]
            )
            X_enc[col] = enc.transform(X_enc[col])

    # Align to imputer expected columns
    expected_cols = list(imputer.feature_names_in_)
    aligned = pd.DataFrame(
        np.full((len(X_enc), len(expected_cols)), np.nan),
        columns=expected_cols,
    )
    for col in X_enc.columns:
        if col in aligned.columns:
            aligned[col] = X_enc[col].values

    # Impute → scale → predict
    imputed = pd.DataFrame(imputer.transform(aligned),   columns=expected_cols)
    scaled  = pd.DataFrame(scaler.transform(imputed),    columns=expected_cols)
    probs   = model.predict_proba(scaled)[:, 1]

    results = pd.DataFrame({
        "Default_Probability": probs.round(4),
        "Risk_Category":       ["🔴 HIGH RISK" if p >= 0.5 else "🟢 LOW RISK" for p in probs],
    })
    return results, probs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 RBL Bank")
    st.markdown("### Credit Risk Predictor")
    st.markdown("---")

    st.markdown("#### 📋 How to Use")
    st.markdown(
        """
        1. Upload **Cust_demographics.csv**
        2. Upload **Cust_Account.csv**
        3. Upload **Cust_enquiry.csv**
        4. Click **Run Prediction**
        5. View & download results
        """
    )
    st.markdown("---")

    st.markdown("#### 🎯 Model Info")
    st.markdown(
        """
        - **Algorithm** : XGBoost (Tuned)
        - **AUC-ROC**   : ~0.9998
        - **Accuracy**  : ~98.7 %
        - **Threshold** : 0.50
        - **Target**    : `Bad_label`
          - `0` → Good Customer
          - `1` → Defaulter
        """
    )
    st.markdown("---")
    st.caption("Prepared by: Puvaneshvaran K | RBL Bank | March 2026")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#7dd3fc; margin-bottom:4px;'>🏦 Credit Risk Prediction</h1>"
    "<p style='color:#94a3b8; margin-top:0;'>RBL Bank · Powered by Tuned XGBoost · Bad_label Classifier</p>",
    unsafe_allow_html=True,
)

# ── artefact check ────────────────────────────────────────────────────────────
if not artefacts_ok:
    st.error(
        f"❌ Model artefacts not found.\n\n"
        f"Expected folder: `{MODEL_DIR}`\n\n"
        f"Error: `{artefact_error}`\n\n"
        "Place the 4 `.pkl` files inside a `models/` folder next to `app.py`."
    )
    st.stop()

st.success("✅ Model artefacts loaded successfully!", icon="✅")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FILE UPLOAD SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Upload Input Files</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Cust_demographics.csv**")
    dem_file = st.file_uploader(
        "Demographics", type=["csv"], key="dem",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**Cust_Account.csv**")
    acc_file = st.file_uploader(
        "Accounts", type=["csv"], key="acc",
        label_visibility="collapsed",
    )

with col3:
    st.markdown("**Cust_enquiry.csv**")
    enq_file = st.file_uploader(
        "Enquiries", type=["csv"], key="enq",
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PREVIEW & PREDICT
# ─────────────────────────────────────────────────────────────────────────────
all_uploaded = dem_file and acc_file and enq_file

if all_uploaded:
    # ── read uploaded files from in-memory bytes (web-compatible) ─────────
    dem_df = pd.read_csv(io.BytesIO(dem_file.read()))
    acc_df = pd.read_csv(io.BytesIO(acc_file.read()))
    enq_df = pd.read_csv(io.BytesIO(enq_file.read()))

    # ── quick data preview ────────────────────────────────────────────────
    st.markdown('<div class="section-header">👁️ Data Preview</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        f"📋 Demographics ({len(dem_df):,} rows)",
        f"🏦 Accounts ({len(acc_df):,} rows)",
        f"🔍 Enquiries ({len(enq_df):,} rows)",
    ])
    with tab1: st.dataframe(dem_df.head(5), use_container_width=True)
    with tab2: st.dataframe(acc_df.head(5), use_container_width=True)
    with tab3: st.dataframe(enq_df.head(5), use_container_width=True)

    # ── summary metrics ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Dataset Summary</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Customers</div>'
            f'<div class="value" style="color:#7dd3fc;">{len(dem_df):,}</div></div>',
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Account Records</div>'
            f'<div class="value" style="color:#a78bfa;">{len(acc_df):,}</div></div>',
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f'<div class="metric-card"><div class="label">Enquiry Records</div>'
            f'<div class="value" style="color:#34d399;">{len(enq_df):,}</div></div>',
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            f'<div class="metric-card"><div class="label">Demo Features</div>'
            f'<div class="value" style="color:#fbbf24;">{dem_df.shape[1]}</div></div>',
            unsafe_allow_html=True,
        )

    # ── predict button ────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🚀 Run Prediction", use_container_width=True)

    if predict_clicked:
        with st.spinner("⚙️ Engineering features and running predictions…"):
            try:
                master = engineer_features(dem_df, acc_df, enq_df)
                results_df, probs = preprocess_and_predict(master)

                # Attach customer_no back for display
                if "customer_no" in master.columns:
                    results_df.insert(0, "customer_no", master["customer_no"].values)

                # ── results section ───────────────────────────────────────
                st.markdown(
                    '<div class="section-header">🎯 Prediction Results</div>',
                    unsafe_allow_html=True,
                )

                n_high = (probs >= 0.5).sum()
                n_low  = (probs  < 0.5).sum()
                pct_high = n_high / len(probs) * 100

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.markdown(
                        f'<div class="metric-card"><div class="label">Total Customers</div>'
                        f'<div class="value" style="color:#7dd3fc;">{len(probs):,}</div></div>',
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f'<div class="metric-card"><div class="label">🔴 High Risk</div>'
                        f'<div class="value" style="color:#f87171;">{n_high:,}</div></div>',
                        unsafe_allow_html=True,
                    )
                with r3:
                    st.markdown(
                        f'<div class="metric-card"><div class="label">🟢 Low Risk</div>'
                        f'<div class="value" style="color:#4ade80;">{n_low:,}</div></div>',
                        unsafe_allow_html=True,
                    )
                with r4:
                    st.markdown(
                        f'<div class="metric-card"><div class="label">Default Rate</div>'
                        f'<div class="value" style="color:#fbbf24;">{pct_high:.1f}%</div></div>',
                        unsafe_allow_html=True,
                    )

                # Risk summary banner
                if pct_high > 20:
                    st.markdown(
                        f'<div class="risk-high">⚠️ {pct_high:.1f}% of customers are HIGH RISK — immediate review recommended</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="risk-low">✅ Portfolio looks healthy — only {pct_high:.1f}% high-risk customers</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # ── full results table ────────────────────────────────────
                st.markdown("#### 📋 Full Prediction Table")

                # Colour-coded Risk_Category
                def highlight_risk(val):
                    if "HIGH" in str(val):
                        return "background-color:#7f1d1d; color:#fecaca; font-weight:600;"
                    elif "LOW" in str(val):
                        return "background-color:#14532d; color:#bbf7d0; font-weight:600;"
                    return ""

                styled = results_df.style.applymap(
                    highlight_risk, subset=["Risk_Category"]
                ).format({"Default_Probability": "{:.4f}"})

                st.dataframe(styled, use_container_width=True, height=400)

                # ── filters ───────────────────────────────────────────────
                st.markdown("#### 🔍 Filter Results")
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    risk_filter = st.selectbox(
                        "Show", ["All Customers", "High Risk Only", "Low Risk Only"]
                    )
                with filter_col2:
                    prob_threshold = st.slider(
                        "Minimum Default Probability", 0.0, 1.0, 0.0, 0.01
                    )

                filtered = results_df.copy()
                if risk_filter == "High Risk Only":
                    filtered = filtered[filtered["Risk_Category"].str.contains("HIGH")]
                elif risk_filter == "Low Risk Only":
                    filtered = filtered[filtered["Risk_Category"].str.contains("LOW")]
                filtered = filtered[filtered["Default_Probability"] >= prob_threshold]

                st.info(f"Showing **{len(filtered):,}** customers after filter.")
                st.dataframe(filtered, use_container_width=True, height=350)

                # ── download ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 💾 Download Results")
                dl1, dl2 = st.columns(2)

                csv_all = results_df.to_csv(index=False).encode("utf-8")
                with dl1:
                    st.download_button(
                        label="⬇️ Download All Predictions (CSV)",
                        data=csv_all,
                        file_name="credit_risk_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                csv_high = filtered.to_csv(index=False).encode("utf-8")
                with dl2:
                    st.download_button(
                        label="⬇️ Download Filtered Results (CSV)",
                        data=csv_high,
                        file_name="credit_risk_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")
                st.exception(exc)

else:
    st.info(
        "👆 Please upload all **three CSV files** above to get started.\n\n"
        "- `Cust_demographics.csv` — customer profile & features\n"
        "- `Cust_Account.csv` — account & payment history\n"
        "- `Cust_enquiry.csv` — credit enquiry history"
    )
