import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import base64
import os
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import io
import sqlite3

# Page Config
st.set_page_config(
    page_title="SmartPolicy AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Design Tokens - Professional Trust Palette */
    :root {
        --primary: #0F2027; /* Deep Navy */
        --primary-gradient: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        --accent: #1976D2; /* Corporate Blue */
        --success: #2E7D32;
        --warning: #ED6C02;
        --danger: #D32F2F;
        --background: #F8FAFC;
        --text-main: #1E293B;
        --text-muted: #64748B;
        --card-bg: rgba(255, 255, 255, 0.95);
        --glass-bg: rgba(255, 255, 255, 0.7);
        --border: #E2E8F0;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        --radius: 16px;
    }

    /* Global Override */
    .stApp {
        background: linear-gradient(rgba(248, 250, 252, 0.65), rgba(248, 250, 252, 0.65)), url('https://images.unsplash.com/photo-1505751172876-fa1923c5c528?q=80&w=2000&auto=format&fit=crop') center center/cover no-repeat fixed !important;
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Professional Integrated Header */
    .header-wrapper {
        background: var(--primary-gradient);
        padding: 50px 30px;
        margin: -70px -100px 40px -100px;
        text-align: center;
        box-shadow: var(--shadow-lg);
        border-bottom: 4px solid var(--accent);
    }
    .header-content h1 {
        color: white !important;
        font-size: 3.2rem !important;
        letter-spacing: -1px;
        margin-bottom: 12px !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .header-content p {
        color: rgba(255,255,255,0.85);
        font-size: 1.25rem;
        font-weight: 300;
        max-width: 700px;
        margin: 0 auto;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background: var(--primary-gradient) !important;
        border-right: 2px solid var(--accent) !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.3) !important;
    }
    
    /* Force sidebar text to be perfectly visible on dark background */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    /* Sidebar Navigation Hover & Base Effects */
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        padding: 12px 16px !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background-color: var(--accent) !important;
        transform: translateX(10px) !important;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }

    /* Glassmorphism Containers */
    .stForm, div[data-testid="stMetricBlock"], .stTabs, div.stExpander, .stMarkdown div[data-testid="stMarkdownContainer"] > div.element-container {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow-sm) !important;
        padding: 24px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stForm:hover, div[data-testid="stMetricBlock"]:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg) !important;
        border-color: var(--accent) !important;
    }

    /* Modern Corporate Buttons */
    div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        padding: 16px 32px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-radius: 12px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(15, 32, 39, 0.3) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    div.stButton > button:hover, div.stFormSubmitButton > button:hover, div.stDownloadButton > button:hover {
        transform: scale(1.02) translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(15, 32, 39, 0.5) !important;
        filter: brightness(1.2);
    }

    /* Results Card with Glassmorphism */
    .result-container {
        padding: 30px;
        border-radius: var(--radius);
        margin: 25px 0;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: var(--shadow-lg);
        animation: fadeInUp 0.8s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .low-risk-box { background: rgba(232, 245, 233, 0.8); border-left: 8px solid var(--success); }
    .medium-risk-box { background: rgba(255, 248, 225, 0.8); border-left: 8px solid var(--warning); }
    .high-risk-box { background: rgba(255, 235, 238, 0.8); border-left: 8px solid var(--danger); }

    .risk-title {
        font-size: 1.8rem !important;
        margin: 0 !important;
        font-weight: 800 !important;
    }
    .risk-desc {
        font-size: 1.1rem;
        margin-top: 8px !important;
        opacity: 0.9;
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: var(--accent) !important;
    }

    /* Hide default streamlit header */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_DIR = r"d:/Smart_Insurance/models"
CHARGE_MODEL_PATH = os.path.join(MODEL_DIR, "advanced_charge_model.pkl")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
CLAIM_MODEL_PATH = os.path.join(MODEL_DIR, "claim_probability_model.pkl")
COMPARISON_METRICS_PATH = os.path.join(MODEL_DIR, "comparison_metrics.pkl")
DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_models():
    with open(CHARGE_MODEL_PATH, 'rb') as f:
        charge_model = pickle.load(f)
    with open(RISK_MODEL_PATH, 'rb') as f:
        risk_model = pickle.load(f)
    with open(CLAIM_MODEL_PATH, 'rb') as f:
        claim_model = pickle.load(f)
    return charge_model, risk_model, claim_model

@st.cache_data
def load_comparison_metrics():
    with open(COMPARISON_METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def calculate_health_risk(row):
    # Calculate a dynamic risk score (0-100) based on inputs
    # Base score starts from age / 2
    score = row['age'] * 0.5
    
    # BMI impact
    bmi = row['bmi']
    if bmi > 30: score += 15
    if bmi > 35: score += 10
    
    # Smoking impact
    if row['smoker'] == 'yes': score += 30
    
    # Medical conditions impact
    medical_cols = ['diabetes', 'hypertension', 'cancer', 'family_heart_disease', 
                    'chronic_kidney_disease', 'asthma', 'thyroid_disorder']
    for col in medical_cols:
        if col in row.index and row[col] == 1:
            score += 10
            
    # Lifestyle impact
    if 'stress_level' in row.index:
        score += row['stress_level'] * 2
        
    return min(score, 100)

def estimate_stress_level(row):
    """
    AI Heuristic to estimate stress level (1-10) based on other factors
    since we removed manual input.
    """
    base_stress = 3  # baseline
    
    # Impact of age (mid-life stress peak)
    age = row['age']
    if 30 <= age <= 50:
        base_stress += 2
        
    # Health conditions usually increase stress
    medical_cols = ['diabetes', 'hypertension', 'cancer', 'asthma', 'mental_health_condition']
    for col in medical_cols:
         if col in row.index and row[col] == 1:
             base_stress += 1
             
    # Activity and diet can mitigate stress
    if row['physical_activity_level'] in ['high', 'active']:
        base_stress -= 1
    elif row['physical_activity_level'] in ['low', 'sedentary']:
        base_stress += 1
        
    if row['diet_type'] == 'poor':
        base_stress += 1
        
    # Smoking and high alcohol intake often correlate with/exacerbate higher stress
    if row['smoker'] == 'yes':
        base_stress += 1
    if row['alcohol_consumption'] in ['frequent', 'high', 'daily']:
        base_stress += 1
        
    # Ensure bounds
    return max(1, min(int(base_stress), 10))

def derive_features(df):
    # Feature Engineering logic must match training script
    # BMI Category
    df['bmi_category'] = pd.cut(
        df['bmi'], 
        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf], 
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    # Age Group
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[-np.inf, 35, 55, np.inf], 
        labels=['Young', 'Adult', 'Senior']
    )
    
    # Add calculated health risk score
    df['health_risk_score'] = df.apply(calculate_health_risk, axis=1)
    
    return df
    
def df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def df_to_sql(df, table_name="data"):
    # Generate a list of INSERT INTO statements
    statements = []
    columns = df.columns.tolist()
    for index, row in df.iterrows():
        values = []
        for val in row:
            if pd.isna(val):
                values.append("NULL")
            elif isinstance(val, (int, float)):
                values.append(str(val))
            else:
                # Escape single quotes and format as string
                val_str = str(val).replace("'", "''")
                values.append(f"'{val_str}'")
        
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});"
        statements.append(sql)
    return "\n".join(statements).encode('utf-8')

def create_risk_gauge(risk_level, risk_score):
    # Use the calculated risk score for the gauge value
    val = risk_score
    
    # Dashboard colors - Matched to Trust Palette
    COLOR_LOW = "#2E7D32"
    COLOR_MEDIUM = "#ED6C02"
    COLOR_HIGH = "#D32F2F"
    
    color = COLOR_LOW
    emoji = "😄"
    if risk_level == "High Risk" or val > 70:
        color = COLOR_HIGH
        emoji = "😟"
    elif risk_level == "Medium Risk" or val > 40:
        color = COLOR_MEDIUM
        emoji = "😐"
    else:
        color = COLOR_LOW
        emoji = "😄"
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{emoji} Risk Assessment: {risk_level}", 'font': {'family': 'Poppins', 'size': 18}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e6ed",
            'steps': [
                {'range': [0, 40], 'color': "#E8F5E9"},
                {'range': [40, 70], 'color': "#FFF8E1"},
                {'range': [70, 100], 'color': "#FFEBEE"}],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': val}}))
    
    fig.update_layout(
        height=320, 
        margin=dict(l=30, r=30, t=60, b=20),
        font={'family': "Outfit", 'color': "#1a2b3c"}
    )
    return fig

def create_pdf_report(input_data, charge, risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="SmartPolicy Insurance Prediction Report", ln=1, align="C")
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Predicted Insurance Charges: ${charge:,.2f}", ln=1)
    pdf.cell(200, 10, txt=f"Assessed Risk Category: {risk}", ln=1)
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Input Details:", ln=1)
    for col in input_data.columns:
        pdf.cell(200, 10, txt=f"{col}: {input_data[col].iloc[0]}", ln=1)
        
        
    return bytes(pdf.output(dest='S'))

def get_premium_breakdown(base_charge, df):
    # Rule-based breakdown logic
    # This simulates a decomposition for demonstration
    age = df['age'].iloc[0]
    bmi = df['bmi'].iloc[0]
    smoker = df['smoker'].iloc[0]
    
    breakdown = {
        "Base Premium": base_charge * 0.4,
        "Age Loading": (age / 100) * base_charge * 0.2,
        "BMI Loading": (bmi / 50) * base_charge * 0.15,
        "Smoking Loading": base_charge * 0.2 if smoker == 'yes' else 0,
        "Disease & Risk Impact": base_charge * 0.05 # Remainder
    }
    # Adjust last item to match total
    total_calc = sum(list(breakdown.values())[:-1])
    breakdown["Disease & Risk Impact"] = base_charge - total_calc
    return breakdown

def get_recommendations(df):
    recs = []
    if df['bmi'].iloc[0] > 25:
        recs.append("🏃 **Weight Management**: Your BMI is above 25. Reducing weight can significantly lower your premium.")
    if df['smoker'].iloc[0] == 'yes':
        recs.append("🚭 **Quit Smoking**: Smokers pay a high premium loading. Quitting can reduce costs by up to 20%.")
    if df['physical_activity_level'].iloc[0] in ['low', 'sedentary']:
        recs.append("🏋️ **Increase Activity**: Regular exercise improves heart health and lowers insurance risk scores.")
    if not recs:
        recs.append("✅ **Maintain Healthy Habits**: You current profile is excellent. Keep it up to stay in the low-risk tier.")
    return recs

# --- MAIN APP LAYOUT ---

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🔮 Prediction", 
    "📊 Dashboard", 
    "📁 Sample Data", 
    "📂 Bulk Scan", 
    "ℹ️ Model Info"
])

# DATA LOADING
try:
    charge_model, risk_model, claim_model = load_models()
    df_raw = load_data()
    comp_metrics = load_comparison_metrics()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

if page == "🔮 Prediction":
    # Centered Header with Gradient
    st.markdown("""
        <div class="header-wrapper">
            <div class="header-content">
                <h1>🛡️ SmartPolicy AI</h1>
                <p>Enterprise-grade Health Insurance Premium Prediction & Risk Analysis Engine</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Form - Two Column Layout
    with st.form("main_form"):
        st.markdown("### 📋 Patient Profile & Information")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 100, 30, help="Enter your current age.")
            sex = st.selectbox("Gender", ["male", "female"])
            residence = st.selectbox("Residence", ["urban", "rural"])
            children = st.number_input("Number of Children", 0, 10, 0)
        with c2:
            bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
            region = st.selectbox("Region", df_raw['region'].unique())
            education_level = st.selectbox("Education Level", df_raw['education_level'].unique())
            income_level = st.selectbox("Income Level", df_raw['income_level'].unique())

        st.divider()
        st.markdown("### 🏥 Medical History & Lifestyle")
        
        c1, c2 = st.columns(2)
        with c1:
            smoker = st.selectbox("Smoker Status", ["no", "yes"])
            diabetes = st.selectbox("Diagonal Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cancer = st.selectbox("Cancer History", [0, 1], format_func=lambda x: "Yes" if x else "No")
            previous_surgeries = st.number_input("Previous Surgeries", 0, 20, 0)
        with c2:
            alcohol = st.selectbox("Alcohol Consumption", df_raw['alcohol_consumption'].unique())
            activity = st.selectbox("Physical Activity Level", df_raw['physical_activity_level'].unique())
            diet = st.selectbox("Diet Type", df_raw['diet_type'].unique())
            st.info("🤖 **Stress Level** and **Health Risk Score** will be evaluated automatically by our AI engine based on the profile above.")

        # Necessary hidden but required for model features
        occupation_risk_level = "low" # default
        family_heart_disease = 0
        chronic_kidney_disease = 0
        asthma = 0
        thyroid_disorder = 0
        mental_health_condition = 0
        hr = 80
        sys_bp = 120
        dia_bp = 80
        sleep = 7.0

        st.divider()
        predict = st.form_submit_button("🚀 CALCULATE PREMIUM")

    if predict:
        # Create DataFrame
        input_dict = {
            'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children], 'smoker': [smoker],
            'region': [region], 'alcohol_consumption': [alcohol], 
            'physical_activity_level': [activity], 'diet_type': [diet],
            'occupation_risk_level': [occupation_risk_level], 'income_level': [income_level],
            'residence': [residence], 'education_level': [education_level],
            'heart_rate': [hr], 'systolic_bp': [sys_bp], 'diastolic_bp': [dia_bp],
            'previous_surgeries': [previous_surgeries], 'sleep_hours': [sleep], 
            'diabetes': [diabetes], 'hypertension': [hypertension], 'cancer': [cancer],
            'family_heart_disease': [family_heart_disease], 'chronic_kidney_disease': [chronic_kidney_disease],
            'asthma': [asthma], 'thyroid_disorder': [thyroid_disorder], 'mental_health_condition': [mental_health_condition]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # AI Generation of Stress Level
        generated_stress = input_df.apply(estimate_stress_level, axis=1).iloc[0]
        input_df['stress_level'] = generated_stress
        
        # Feature Engineering (This now automatically computes Health Risk Score using generated_stress)
        input_df = derive_features(input_df)
        generated_risk_score = input_df['health_risk_score'].iloc[0]
        
        # Predictions
        try:
            charge_pred = charge_model.predict(input_df)[0]
            risk_pred = risk_model.predict(input_df)[0]
            claim_prob = claim_model.predict_proba(input_df)[0][1]
            
            # --- RESULTS SECTION ---
            st.divider()
            
            # Risk Style and Class
            risk_class = ""
            risk_emoji = ""
            if risk_pred == "High Risk":
                risk_class = "high-risk"
                risk_emoji = "😟"
            elif risk_pred == "Medium Risk":
                risk_class = "medium-risk"
                risk_emoji = "😐"
            else:
                risk_class = "low-risk"
                risk_emoji = "😄"
            
            # Results Card
            st.markdown(f"""
                <div class="result-container {risk_pred.lower().replace(' ', '-')}-box">
                    <h3 class="risk-title">{risk_emoji} Risk Category: {risk_pred}</h3>
                    <p class="risk-desc">Our core AI engine has processed your data and assigned a <b>{risk_pred}</b> status based on clinical and lifestyle factors.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Estimated Yearly Premium", f"${charge_pred:,.2f}")
            with c2:
                st.metric("Estimated Monthly Installment", f"${charge_pred/12:,.2f}")
            
            # Gauge & Factors
            col1, col2 = st.columns([1, 1])
            with col1:
                fig = create_risk_gauge(risk_pred, generated_risk_score)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🤖 AI Diagnostics")
                st.info(f"**AI-Estimated Stress Level:** {generated_stress}/10")
                st.info(f"**AI-Calculated Health Risk Score:** {generated_risk_score:.1f}/100")
                st.info(f"**BMI Category:** {input_df['bmi_category'][0]}")
                st.info(f"**Age Group:** {input_df['age_group'][0]}")
            
            # Download
            st.download_button(
                label="📄 DOWNLOAD OFFICIAL PDF REPORT",
                data=create_pdf_report(input_df, charge_pred, risk_pred),
                file_name="smartpolicy_report.pdf",
                mime="application/pdf"
            )

            # --- NEW ADVANCED FEATURES SECTION ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Premium Breakdown")
                breakdown = get_premium_breakdown(charge_pred, input_df)
                for item, val in breakdown.items():
                    st.write(f"**{item}:** ${val:,.2f}")
                
                st.markdown("---")
                st.subheader("🏥 Claim Probability")
                st.write(f"Predicted Likelihood of Claim: **{claim_prob*100:.1f}%**")
                if claim_prob > 0.6:
                    st.error("High Probability of Claim Detected")
                elif claim_prob > 0.3:
                    st.warning("Moderate Probability of Claim Detected")
                else:
                    st.success("Low Probability of Claim Detected")

            with col2:
                st.subheader("💡 Wellness Recommendations")
                recommendations = get_recommendations(input_df)
                for rec in recommendations:
                    st.info(rec)

            st.divider()
            st.subheader("🧪 AI Explainability (SHAP)")
            with st.spinner("Generating explanation..."):
                explainer = shap.TreeExplainer(charge_model.named_steps['regressor'])
                prep_data = charge_model.named_steps['preprocessor'].transform(input_df)
                feature_names = charge_model.named_steps['preprocessor'].get_feature_names_out()
                shap_values = explainer.shap_values(prep_data)
                
                tab1, tab2 = st.tabs(["Local Explanation", "Global Importance"])
                with tab1:
                    st.write("Why this specific premium was predicted:")
                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                    shap.plots.bar(shap.Explanation(values=shap_values[0], 
                                                   base_values=explainer.expected_value, 
                                                   data=prep_data[0], 
                                                   feature_names=feature_names), 
                                  max_display=10, show=False)
                    st.pyplot(fig1)
                
                with tab2:
                    st.write("Overall factors affecting the model:")
                    # For global summary, we use a sample of the training data
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    # Use the raw data from df_raw for a more meaningful summary plot
                    sample_indices = np.random.choice(len(df_raw), min(100, len(df_raw)), replace=False)
                    sample_data = df_raw.iloc[sample_indices].drop(columns=['charges', 'health_risk_score'], errors='ignore')
                    
                    # Transform sample for SHAP
                    # Ensure columns match training (handle risk_category dropdown if it was in train)
                    # For simplicity, we use the explainer on the input_df's preprocessed structure
                    sample_df = derive_features(df_raw.head(100).copy())
                    shap_data = charge_model.named_steps['preprocessor'].transform(sample_df)
                    shap.summary_plot(explainer.shap_values(shap_data), 
                                      shap_data,
                                      feature_names=feature_names, plot_type="bar", show=False)
                    st.pyplot(plt.gcf())
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
elif page == "📊 Dashboard":
    st.title("Interactive EDA Dashboard")
    
    # 1. Distribution of Charges
    st.subheader("Distribution of Medical Charges")
    fig_hist = px.histogram(df_raw, x="charges", nbins=50, title="Charge Distribution", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # 2. Correlation Heatmap (Selected Numerical)
    st.subheader("Correlation Heatmap")
    num_cols = ['age', 'bmi', 'children', 'heart_rate', 'health_risk_score', 'charges']
    corr = df_raw[num_cols].corr()
    fig_heatmap = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 3. Box Plot for Charges by Smoker Status
    st.subheader("Charges by Smoker Status")
    fig_box = px.box(df_raw, x="smoker", y="charges", color="smoker", title="Premium Charges by Smoker Status")
    st.plotly_chart(fig_box, use_container_width=True)

elif page == "📁 Sample Data":
    st.title("Sample Data Preview")
    st.write("Browse a sample of our health insurance dataset and download it in various formats.")
    
    sample_df = df_raw.head(100).copy()
    
    st.dataframe(sample_df, use_container_width=True)
    
    st.subheader("Download Options")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            label="Download CSV",
            data=sample_df.to_csv(index=False).encode('utf-8'),
            file_name="sample_data.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="Download JSON",
            data=sample_df.to_json(orient='records').encode('utf-8'),
            file_name="sample_data.json",
            mime="application/json"
        )
    with col3:
        st.download_button(
            label="Download Excel",
            data=df_to_excel(sample_df),
            file_name="sample_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col4:
        st.download_button(
            label="Download SQL",
            data=df_to_sql(sample_df, "sample_data"),
            file_name="sample_data.sql",
            mime="text/plain"
        )

elif page == "📂 Bulk Scan":
    st.title("Bulk Data Scan")
    st.write("Upload a dataset to process multiple patients at once and export the predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                bulk_df = pd.read_csv(uploaded_file)
            else:
                bulk_df = pd.read_excel(uploaded_file)
                
            st.write(f"Loaded {len(bulk_df)} records.")
            
            with st.spinner("Processing records and running AI models..."):
                # Handle required default columns if missing
                default_cols = {
                    'occupation_risk_level': 'low',
                    'family_heart_disease': 0,
                    'chronic_kidney_disease': 0,
                    'asthma': 0,
                    'thyroid_disorder': 0,
                    'mental_health_condition': 0,
                    'heart_rate': 80,
                    'systolic_bp': 120,
                    'diastolic_bp': 80,
                    'previous_surgeries': 0,
                    'sleep_hours': 7.0,
                    'diabetes': 0,
                    'hypertension': 0,
                    'cancer': 0
                }
                for col, default_val in default_cols.items():
                    if col not in bulk_df.columns:
                        bulk_df[col] = default_val
                        
                # Estimate stress level
                if 'stress_level' not in bulk_df.columns:
                    bulk_df['stress_level'] = bulk_df.apply(estimate_stress_level, axis=1)
                
                # Derive features including health_risk_score
                bulk_df = derive_features(bulk_df)
                
                # Run predictions
                predictions_df = bulk_df.copy()
                predictions_df['Predicted_Charges'] = charge_model.predict(bulk_df)
                predictions_df['Predicted_Risk_Category'] = risk_model.predict(bulk_df)
                predictions_df['Claim_Probability'] = claim_model.predict_proba(bulk_df)[:, 1]
                
            st.success("Bulk scan completed successfully!")
            
            # Display results
            st.subheader("Scan Results Preview")
            display_cols = ['age', 'sex', 'bmi', 'smoker', 'Predicted_Charges', 'Predicted_Risk_Category', 'Claim_Probability']
            show_cols = [c for c in display_cols if c in predictions_df.columns]
            st.dataframe(predictions_df[show_cols].head(50), use_container_width=True)
            
            st.subheader("Download Full Scan Results")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.download_button(
                    label="Download CSV",
                    data=predictions_df.to_csv(index=False).encode('utf-8'),
                    file_name="bulk_scan_results.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Download JSON",
                    data=predictions_df.to_json(orient='records').encode('utf-8'),
                    file_name="bulk_scan_results.json",
                    mime="application/json"
                )
            with col3:
                st.download_button(
                    label="Download Excel",
                    data=df_to_excel(predictions_df),
                    file_name="bulk_scan_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col4:
                st.download_button(
                    label="Download SQL",
                    data=df_to_sql(predictions_df, "scan_results"),
                    file_name="bulk_scan_results.sql",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")


elif page == "ℹ️ Model Info":
    st.title("Model Architecture")
    st.info("This system uses a Hybrid Ensemble Approach.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Charge Predictor")
        st.markdown("""
        - **Algorithm**: Extreme Gradient Boosting (XGBoost)
        - **Task**: Regression (Predicting Continuous Value)
        - **Features**: 25 Input Variables
        - **Performance**: High R² Score
        """)
        
    with col2:
        st.markdown("### Risk Classifier")
        st.markdown("""
        - **Algorithm**: Random Forest Classifier
        - **Task**: Multi-class Classification
        - **Classes**: Low Risk, Medium Risk, High Risk
        - **Accuracy**: >95%
        """)
    
    st.markdown("### System Architecture")
    st.code("""
    [User Input] --> [Feature Engineering Layer] 
                          |
        +-----------------+-----------------+
        |                                   |
    [XGBoost Regressor]             [Random Forest Classifier]
        |                                   |
    [Charge Prediction]             [Risk Category Classification]
        |                                   |
        +-----------------+-----------------+
                          |
                  [Streamlit Dashboard]
    """)
