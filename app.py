import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PAGE CONFIG & THEME (Dark Theme)
# ==========================================
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS with custom accents
st.markdown("""
<style>
    /* Main body background and text color */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar background and text color */
    [data-testid="stSidebar"] {
        background-color: #1A1D24;
    }
    [data-testid="stSidebar"] * {
        color: #FAFAFA !important;
    }

    /* Header style */
    .header-style {
        background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
        color: #A78BFA; /* Lavender accent color */
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #313A46;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .header-style h1 { color: #A78BFA; margin: 0; }
    .header-style p { color: #94A3B8; margin: 0.5rem 0 0; }

    /* Input labels color */
    .stNumberInput label, .stSelectbox label, .stTextInput label, .stSlider label {
        color: #94A3B8 !important;
    }

    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #1A1D24;
        border: 1px solid #313A46;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
    }
    div[data-testid="metric-container"] div {
        color: #FAFAFA !important;
    }

    /* Subheader color */
    .stMarkdown h3 {
        color: #A78BFA !important;
        margin-top: 1.5rem;
    }

    /* Prediction cards styling */
    .prediction-card {
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
    .risk-high {
        background-color: #7F1D1D; /* Deep red for high risk */
        border: 2px solid #F87171;
    }
    .risk-low {
        background-color: #064E3B; /* Deep green for low risk */
        border: 2px solid #34D399;
    }
    
    /* Center the Predict button */
    div.stButton > button:first-child {
        background-color: #7C3AED;
        color: white;
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #6D28D9;
        color: white;
        border: 2px solid #A78BFA;
    }

</style>
""", unsafe_allow_html=True)

# dark matplotlib style for plots to match theme
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1A1D24",
    "axes.edgecolor": "#313A46",
    "axes.labelcolor": "#94A3B8",
    "axes.titlecolor": "#FAFAFA",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "text.color": "#FAFAFA",
    "grid.color": "#313A46",
    "legend.facecolor": "#1A1D24",
    "legend.edgecolor": "#313A46",
    "legend.labelcolor": "#FAFAFA",
})

# ==========================================
# FILE PATHS & ARTEFACT LOADING
# ==========================================
# Relative paths for GitHub deployment
# Make sure these files exist in these exact subfolders in your repo
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "credit_risk_xgboost_model.pkl") # Use your specific filename
IMP_PATH = os.path.join(MODEL_DIR, "imputer.pkl")
SCL_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

@st.cache_resource(show_spinner="Loading pre-trained models...")
def load_artefacts():
    # Helper function to safely load pkl file with error handling for deployment
    def safe_load(path, name):
        if not os.path.exists(path):
            st.error(f"❌ Error: File not found at **`{path}`**. Please check your GitHub repo structure.")
            st.stop()
        return joblib.load(path)

    # Load all models and preprocessing objects
    model = safe_load(MODEL_PATH, "Model")
    imputer = safe_load(IMP_PATH, "Imputer")
    scaler = safe_load(SCL_PATH, "Scaler")
    label_encoders = safe_load(ENC_PATH, "Label Encoders")
    return model, imputer, scaler, label_encoders

# Main Page - Header and Metrics (Single page, no navigation needed)
st.markdown("""
<div class="header-style">
    <h1>💰 Loan Credit Risk Prediction App</h1>
    <p>Predicting loan default probability for new customers using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

try:
    # Load model and encoders
    model, imputer, scaler, label_encoders = load_artefacts()

    # Define categorical features expected by your model based on label_encoders keys
    # Make sure these keys match exactly what's in your label_encoders.pkl
    # Adjust names if they are different, e.g., 'feature_city', 'feature_employment'
    categorical_features = ['City', 'Employment', 'Card Type', 'Primary Bank'] 
    
    # ------------------------------------------
    # Predictor Section
    # ------------------------------------------
    st.markdown("### 📝 Enter Customer Details for Prediction")
    
    # User Inputs for Prediction
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        # User input widgets - adjust these based on your specific features
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700, step=10, help="Customer's current CIBIL or internal score.")
            age = st.number_input("Age (years)", min_value=18, max_value=80, value=40)
            income = st.number_input("Annual Income (₹)", min_value=0, max_value=10000000, value=500000, step=10000, help="Customer's declared annual income.")
            employment = st.selectbox("Employment Status", ["Self","Salaried","Rented","Other"]) # Example options, adjust as per label_encoders

        with col2:
            credit_limit = st.number_input("Requested Credit Limit (₹)", min_value=0, max_value=5000000, value=200000, step=10000, help="The loan amount customer is applying for.")
            # Metro cities - example option, adjust as per your logic
            METRO_CITIES = [
                "Mumbai / Navi Mumbai / Thane", "Bengaluru",
                "New Delhi", "Pune", "Gurgaon"
            ]
            city = st.selectbox("Current City", METRO_CITIES + ["Other"])
            card_type = st.selectbox("Card Type Applied For", ["Insignia","Titanium Delight","Other"]) # Adjust options
            total_accounts = st.number_input("Total Existing Accounts", min_value=0, max_value=20, value=2)

        with col3:
            total_past_due = st.number_input("Total Past Due (₹)", min_value=0, max_value=1000000, value=0, step=1000, help="Total overdue amount on all loans.")
            total_delinq = st.number_input("Total Delinquency Score", min_value=0, max_value=50, value=0, help="Summary score from payment history string.")
            unique_purp = st.number_input("Unique Enquiry Purposes", min_value=0, max_value=10, value=2, help="Number of different reasons for credit enquiries.")
            days_since = st.number_input("Days Since Last Enquiry", min_value=0, max_value=3000, value=180)
            
        # Optional: Additional feature based on input
        income_ratio = income / (credit_limit + 1)
        # st.metric("Income/Credit Ratio (Derived)", f"{income_ratio:.3f}")

        st.markdown("---")
        # Centered Predict Button inside the form
        submitted = st.form_submit_button("🔮 Predict Credit Risk")

    # Output section safely outside the form
    if submitted:
        # 1. Gather input into a dictionary with correct feature names (must match imputer.feature_names_in_)
        # Adjust key names here to match your model's exact features (e.g., 'feature_9' -> 'Requests Credit Limit')
        input_data = {
            'feature_4': credit_score,          # Replace 'feature_4' with your model's feature name for Credit Score
            'feature_24_DOB_parsed_AGE': age,  # Use your model's exact derived AGE feature name
            'feature_38': income,               # Replace with model's Income feature name
            'feature_9': credit_limit,          # Replace with model's Credit Limit feature name
            'income_credit_ratio': income_ratio, 
            'is_metro': 1 if city in METRO_CITIES else 0, # Logic based on input
            'total_accounts': total_accounts,
            'total_past_due': total_past_due,
            'total_delinquency': total_delinq,
            'unique_enq_purpose': unique_purp,
            'days_since_last_enquiry': days_since,
            'City': city,
            'Employment': employment,
            'Card Type': card_type,
            'Primary Bank': "Other" # Adjust if needed or remove if not categorical
        }
        
        # Oru safety check features adjust panna (based on imputer or scaler names)
        # expected_features = scaler.feature_names_in_ # Or imputer
        # st.write(f"expected_features: {expected_features}") # Use this to debug if needed

        # 2. Encode categorical features using loaded label_encoders
        for col, enc in label_encoders.items():
            if col in input_data:
                try:
                    # str(input_data[col]) because encoders usually work with strings
                    # transform returns a list, take first element [0]
                    input_data[col] = enc.transform([str(input_data[col])])[0]
                except ValueError:
                    # Handling unseen labels during transformation
                    st.warning(f"Unseen value for '{col}'. Using default encoding (0).")
                    input_data[col] = 0 # Default fallback or handle differently if needed

        # 3. Handle missing values (Imputer)
        expected_cols_in = list(imputer.feature_names_in_)
        df_raw = pd.DataFrame([input_data])
        # reindex to ensure correct column order for imputer
        df_in = df_raw.reindex(columns=expected_cols_in)
        
        arr_imp = imputer.transform(df_in)
        
        # 4. Handle numerical scaling (Scaler)
        # Imputer output column order must match Scaler input column order
        out_cols = imputer.get_feature_names_out()
        X_imp = pd.DataFrame(arr_imp, columns=out_cols)
        arr_sc = scaler.transform(X_imp)
        
        # 5. Prediction (Probabilities)
        prob = model.predict_proba(arr_sc)[0, 1] # Probability of belonging to class 1 (High Risk)

        # ------------------------------------------
        # Prediction Output Section
        # ------------------------------------------
        st.markdown("### 🔮 Prediction Result")
        
        # Visualization: Risk Gauge Chart
        fig, ax = plt.subplots(figsize=(6, 1.5))
        # Accent colours work on dark background
        C_BLUE = "#818CF8" # Good class color
        C_RED = "#F87171"  # Bad class color
        # Stacked bar to show risk
        ax.barh([0], [prob*100], color=C_RED, label='High Risk (Default)', edgecolor='#0E1117', linewidth=1)
        ax.barh([0], [(1-prob)*100], left=[prob*100], color=C_BLUE, label='Low Risk (Repay)', edgecolor='#0E1117', linewidth=1)
        ax.set_yticks([]) # Hide y-axis ticks
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontweight='bold', color='#FAFAFA')
        ax.set_title(f"Credit Risk Score: {prob*100:.0f} / 100", fontweight='bold', fontsize=12)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=2)
        st.pyplot(fig)
        plt.close() # Important to close matplotlib figures after plot

        # Decision Badge based on probability threshold (e.g., 50%)
        if prob > 0.5:
             # High risk card
             st.markdown(f"""
             <div class="prediction-card risk-high">
                 <h2>⚠️ High Credit Risk Customer</h2>
                 <p>Default Probability: <strong>{prob:.2%}</strong></p>
                 <p>This application shows a high likelihood of loan default. Manual review or rejection is recommended.</p>
             </div>
             """, unsafe_allow_html=True)
        else:
             # Low risk card
             st.markdown(f"""
             <div class="prediction-card risk-low">
                 <h2>✅ Low Credit Risk Customer</h2>
                 <p>Default Probability: <strong>{prob:.2%}</strong></p>
                 <p>This application shows a high likelihood of loan repayment. Recommended for approval.</p>
             </div>
             """, unsafe_allow_html=True)
        
        # Informational note
        st.info("💡 **Note:** This prediction is based on the input data provided above. It is a decision support tool, not a final guarantee of loan performance.")

    # ------------------------------------------
    # Sidebar Section - Project Info
    # ------------------------------------------
    with st.sidebar:
        st.markdown("## 🏦 Credit Risk Model")
        st.markdown("### Model Details")
        st.markdown("- **Algorithm:** Tuned XGBoost Classifier")
        # Metric example, replace with actual values from training notebook
        st.metric("Test AUC Score (example)", "89.2%", delta="+0.5%")
        st.metric("Test Recall (example)", "85.1%", help="Targeting recall to capture maximum defaults.")
        st.markdown("---")
        st.markdown("### Business Context")
        st.markdown("Developing a model to predict the probability of loan default (Bad label = 1). The goal is to **minimize default rate (Recall)** while maintaining good customer approval rate.")
        st.markdown("---")
        st.markdown("<small style='color:#4B5563'>Developed with ❤️ using Python & Streamlit.</small>", unsafe_allow_html=True)
        # Link to the GitHub Repo - adjust name if needed
        st.markdown(f"<small style='color:#4B5563'><a href='https://github.com/puvanesh-prog/Loan_Prediction_Using_BinaryClassificetion' target='_blank'>[Source Code on GitHub]</a></small>", unsafe_allow_html=True)

# Main exception handler
except Exception as e:
    st.error(f"❌ An unexpected error occurred: {e}. Please contact the developer.")
    st.write("Detailed error information for debugging:")
    st.exception(e)
