# ============================================================
# Startup_prediction UI.py
# This is the Streamlit web app for the Startup Success Predictor.
# It loads the trained ML model and lets users enter startup
# details to get a success / failure prediction.
# ============================================================

# Import all required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="AI Startup Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD THE SAVED MODEL FILES
# @st.cache_resource makes the model load only ONCE,
# not every time the user clicks something.
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler        = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_cols  = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
    cols_to_scale = joblib.load(os.path.join(BASE_DIR, "cols_to_scale.pkl"))
    return model, scaler, feature_cols, cols_to_scale

# Try to load the model. If files are missing, show an error.
try:
    model, scaler, feature_cols, cols_to_scale = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ============================================================
# PREDICTION FUNCTION
# This takes all user inputs, prepares them exactly the same
# way we prepared training data, then calls the model.
#
# KEY STEPS (must match train_and_save_model.py exactly):
#   1. Compute derived ratio features
#   2. Determine Company Status (Profit / Loss)
#   3. One-hot encode categoricals (same drop_first=True logic)
#   4. Align columns to training order
#   5. Scale with the saved scaler
#   6. Predict
# ============================================================
def make_prediction(funding, revenue, valuation, customers,
                    employees, age, acquired, ipo,
                    country, industry, stage, tech,
                    social_followers):

    # ---- Derived numeric features (same as training) ----
    funding_per_emp   = funding  / (employees + 1)
    revenue_per_emp   = revenue  / (employees + 1)
    val_to_fund_ratio = (valuation * 1000) / (funding + 1)
    rev_to_fund_ratio = revenue / (funding + 1)       # NEW feature added in training
    customer_per_emp  = customers / (employees + 1)   # NEW feature added in training

    # ---- Company Status ----
    company_status = "Profit" if revenue >= funding else "Loss"

    # ---- Build input row dictionary ----
    input_data = {
        "Total Funding ($M)":        funding,
        "Number of Employees":       employees,
        "Annual Revenue ($M)":       revenue,
        "Valuation ($B)":            valuation,
        "Acquired?":                 1 if acquired else 0,
        "IPO?":                      1 if ipo else 0,
        "Customer Base (Millions)":  customers,
        "Social Media Followers":    social_followers,
        "Age":                       age,
        "Funding_per_Employee":      funding_per_emp,
        "Revenue_per_Employee":      revenue_per_emp,
        "Valuation_to_Funding_Ratio": val_to_fund_ratio,
        "Revenue_to_Funding_Ratio":  rev_to_fund_ratio,
        "Customer_per_Employee":     customer_per_emp,
    }

    # ---- One-hot encode categoricals ----
    # sorted() is critical — must match the order used by pd.get_dummies(drop_first=True)
    # The first (alphabetically lowest) value is dropped; we create 0/1 for the rest.

    country_vals = sorted(["Australia", "Brazil", "Canada", "China", "France",
                            "Germany", "India", "Japan", "UK", "USA"])
    industry_vals = sorted(["AI", "E-commerce", "EdTech", "Energy", "FinTech",
                             "FoodTech", "Gaming", "Healthcare", "Logistics", "Tech"])
    stage_vals    = sorted(["IPO", "Seed", "Series A", "Series B", "Series C"])
    tech_vals     = sorted(["C++, ML", "Java, Spring", "Node.js, React", "PHP, Laravel", "Python, AI"])
    status_vals   = sorted(["Loss", "Profit"])

    # Country (drops "Australia")
    for val in country_vals[1:]:
        input_data["Country_" + val] = 1 if country == val else 0

    # Industry (drops "AI")
    for val in industry_vals[1:]:
        input_data["Industry_" + val] = 1 if industry == val else 0

    # Funding Stage (drops "IPO")
    for val in stage_vals[1:]:
        input_data["Funding Stage_" + val] = 1 if stage == val else 0

    # Tech Stack (drops "C++, ML")
    for val in tech_vals[1:]:
        input_data["Tech Stack_" + val] = 1 if tech == val else 0

    # Company Status (drops "Loss")
    for val in status_vals[1:]:
        input_data["Company Status_" + val] = 1 if company_status == val else 0

    # ---- Convert to DataFrame ----
    X_input = pd.DataFrame([input_data])

    # ---- Make sure all training columns are present (fill missing with 0) ----
    for col in feature_cols:
        if col not in X_input.columns:
            X_input[col] = 0

    # ---- Reorder columns to exactly match training order ----
    X_input = X_input[feature_cols]

    # ---- Apply scaling ----
    X_scaled = X_input.copy()
    scale_cols = [c for c in cols_to_scale if c in X_input.columns]
    X_scaled[scale_cols] = scaler.transform(X_input[scale_cols])

    # ---- Get prediction and probability ----
    prediction  = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])  # probability of success

    return prediction, probability


# ============================================================
# PAGE CSS STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    min-height: 100vh;
}

.block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 1400px;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1f3a 0%, #0f2744 50%, #0a1a2e 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,179,237,0.06) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50%       { transform: scale(1.05); opacity: 1; }
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(120deg, #63b3ed, #90cdf4, #4fc3f7, #81e6d9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
    position: relative;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(203,213,225,0.75);
    font-weight: 400;
    position: relative;
    margin: 0;
}

/* Section headers */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #63b3ed;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1.5rem;
}

/* Success result card (green) */
.result-success {
    background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,78,59,0.1) 100%);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
    box-shadow: 0 20px 60px rgba(16,185,129,0.1);
}

/* Failure result card (red) */
.result-failure {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(127,29,29,0.1) 100%);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
    box-shadow: 0 20px 60px rgba(239,68,68,0.1);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-icon    { font-size: 4rem; margin-bottom: 0.5rem; }
.result-title   { font-size: 2rem; font-weight: 800; margin: 0.5rem 0; }
.result-subtitle{ font-size: 1rem; color: rgba(203,213,225,0.8); margin: 0; }
.success-text   { color: #34d399; }
.failure-text   { color: #f87171; }

/* Probability bar */
.prob-bar-wrap {
    background: rgba(15,30,55,0.6);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.prob-label {
    font-size: 0.85rem;
    color: rgba(148,163,184,0.8);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.prob-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.prob-track {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.prob-fill-success {
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #34d399);
}
.prob-fill-failure {
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444, #f87171);
}

/* Insight card */
.advice-card {
    background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(99,179,237,0.02) 100%);
    border-left: 3px solid #63b3ed;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: rgba(226,232,240,0.9);
    line-height: 1.6;
}

/* Input widget styling */
.stSelectbox > div > div {
    background: rgba(15,30,55,0.8) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stNumberInput > div > div > input {
    background: rgba(15,30,55,0.8) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSlider > div > div > div > div {
    background: #63b3ed !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 30px rgba(37,99,235,0.4);
    text-transform: uppercase;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(37,99,235,0.5);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Label colors */
.stSelectbox label, .stNumberInput label, .stSlider label,
.stCheckbox label, .stRadio label {
    color: rgba(203,213,225,0.85) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

hr { border-color: rgba(99,179,237,0.1) !important; }

.stCheckbox > label {
    color: rgba(203,213,225,0.85) !important;
    font-weight: 500 !important;
}

.stAlert {
    border-radius: 12px !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    background: rgba(15,30,55,0.6) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO BANNER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div class="result-icon">🚀</div>
    <h1 class="hero-title">AI Startup Predictor</h1>
    <p class="hero-subtitle">
        Enter your startup details &middot; Powered by Advanced Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Show an error if model files are missing
if not model_loaded:
    st.error("Model files not found! Please run: python train_and_save_model.py")
    st.stop()


# ============================================================
# MAIN LAYOUT
# ============================================================
col_left, col_right = st.columns([3, 2], gap="large")

# ============================================================
# LEFT COLUMN — Input Fields
# ============================================================
with col_left:

    # ---- Section 1: Financial Metrics ----
    st.markdown('<div class="section-title">💰 Financial Metrics</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)

    with f1:
        funding = st.number_input(
            "Total Funding ($M)",
            min_value=0.0, max_value=10000.0,
            value=150.0, step=10.0,
            help="Total money raised in millions USD"
        )
        revenue = st.number_input(
            "Annual Revenue ($M)",
            min_value=0.0, max_value=50000.0,
            value=45.0, step=5.0,
            help="Yearly revenue in millions USD"
        )

    with f2:
        valuation = st.number_input(
            "Valuation ($B)",
            min_value=0.0, max_value=1000.0,
            value=1.2, step=0.1,
            help="Company valuation in billions USD"
        )
        customers = st.number_input(
            "Customer Base (Millions)",
            min_value=0.0, max_value=5000.0,
            value=5.5, step=0.5,
            help="Total customers in millions"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Section 2: Operations & Growth ----
    st.markdown('<div class="section-title">🏢 Operations &amp; Growth</div>', unsafe_allow_html=True)
    o1, o2 = st.columns(2)

    with o1:
        employees = st.number_input(
            "Number of Employees",
            min_value=1, max_value=100000,
            value=120, step=10,
            help="Total full-time employees"
        )
        social_followers = st.number_input(
            "Social Media Followers",
            min_value=0, max_value=1000000000,
            value=150000, step=10000,
            help="Total followers across all platforms"
        )

    with o2:
        age = st.slider(
            "Startup Age (Years)",
            min_value=0, max_value=30,
            value=4, step=1,
            help="Years since founded"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        acquired = st.checkbox("🤝 Has Been Acquired", value=False)
        ipo      = st.checkbox("📈 Has Done an IPO",   value=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Section 3: Company Profile ----
    st.markdown('<div class="section-title">🌍 Company Profile</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)

    with p1:
        country = st.selectbox(
            "Country",
            options=["USA", "UK", "Canada", "China", "Germany",
                     "France", "Japan", "India", "Brazil", "Australia"],
            index=0
        )
        industry = st.selectbox(
            "Industry Sector",
            options=["Tech", "FinTech", "E-commerce", "EdTech", "Healthcare",
                     "Logistics", "Gaming", "Energy", "FoodTech", "AI"],
            index=0
        )

    with p2:
        stage = st.selectbox(
            "Funding Stage",
            options=["Seed", "Series A", "Series B", "Series C", "IPO"],
            index=2
        )
        tech = st.selectbox(
            "Primary Tech Stack",
            options=["Python, AI", "Node.js, React", "Java, Spring", "PHP, Laravel", "C++, ML"],
            index=0
        )


# ============================================================
# RIGHT COLUMN — Prediction Panel
# ============================================================
with col_right:

    st.markdown('<div class="section-title">🎯 Prediction Panel</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Show live ratio metrics above the button
    emp              = max(employees, 1)
    funding_per_emp  = funding / emp
    revenue_per_emp  = revenue / emp
    val_to_funding   = (valuation * 1000.0) / max(funding, 0.001)
    rev_to_funding   = revenue / max(funding, 0.001)

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m1.metric("Funding / Employee",  "$" + str(round(funding_per_emp, 2)) + "M")
    m2.metric("Revenue / Employee",  "$" + str(round(revenue_per_emp, 2)) + "M")
    m3.metric("Val / Funding Ratio", str(round(val_to_funding, 1)) + "x")
    m4.metric("Rev / Funding Ratio", str(round(rev_to_funding, 2)) + "x")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Predict Button ----
    predict_btn = st.button("🔮  Predict Startup Success", key="predict_btn")

    if predict_btn:

        # Run the prediction
        with st.spinner("Running AI prediction..."):
            prediction, probability = make_prediction(
                funding=funding,
                revenue=revenue,
                valuation=valuation,
                customers=customers,
                employees=employees,
                age=age,
                acquired=acquired,
                ipo=ipo,
                country=country,
                industry=industry,
                stage=stage,
                tech=tech,
                social_followers=social_followers
            )

        prob_percent = probability * 100

        # ---- Show result card ----
        if prediction == 1:
            st.markdown("""
            <div class="result-success">
                <div class="result-icon">🏆</div>
                <div class="result-title success-text">High Success Potential</div>
                <p class="result-subtitle">Your startup shows strong business fundamentals</p>
            </div>
            """, unsafe_allow_html=True)
            bar_class = "prob-fill-success"
            bar_color = "#34d399"
        else:
            st.markdown("""
            <div class="result-failure">
                <div class="result-icon">⚠️</div>
                <div class="result-title failure-text">Needs Improvement</div>
                <p class="result-subtitle">Key business metrics need to be strengthened</p>
            </div>
            """, unsafe_allow_html=True)
            bar_class = "prob-fill-failure"
            bar_color = "#f87171"

        # ---- Probability bar ----
        st.markdown(f"""
        <div class="prob-bar-wrap">
            <div class="prob-label">Success Probability</div>
            <div class="prob-value" style="color: {bar_color}">
                {prob_percent:.1f}%
            </div>
            <div class="prob-track">
                <div class="{bar_class}" style="width:{prob_percent:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Startup Snapshot Table ----
        st.markdown("**📊 Startup Snapshot**")

        field_names = [
            "Total Funding", "Annual Revenue", "Valuation", "Customers",
            "Employees", "Startup Age", "Country", "Industry",
            "Funding Stage", "Tech Stack", "Acquired", "IPO"
        ]
        field_values = [
            "$" + str(round(funding, 1))   + "M",
            "$" + str(round(revenue, 1))   + "M",
            "$" + str(round(valuation, 2)) + "B",
            str(round(customers, 1))       + "M",
            str(employees),
            str(age) + " yrs",
            country, industry, stage, tech,
            "Yes ✅" if acquired else "No ❌",
            "Yes ✅" if ipo      else "No ❌"
        ]

        snapshot_df = pd.DataFrame({"Field": field_names, "Value": field_values})
        st.dataframe(snapshot_df, hide_index=True, use_container_width=True)

        # ---- Key Metrics Table ----
        st.markdown("---")
        st.markdown("**📐 Key Business Ratios**")

        ratio_df = pd.DataFrame({
            "Metric": [
                "Funding per Employee",
                "Revenue per Employee",
                "Valuation-to-Funding",
                "Revenue-to-Funding",
                "Company Status"
            ],
            "Value": [
                "$" + str(round(funding_per_emp, 3)) + "M",
                "$" + str(round(revenue_per_emp, 3)) + "M",
                str(round(val_to_funding, 2)) + "x",
                str(round(rev_to_funding, 2)) + "x",
                "Profit 💚" if revenue >= funding else "Loss 🔴"
            ]
        })
        st.dataframe(ratio_df, hide_index=True, use_container_width=True)

        # ---- Key Insights ----
        st.markdown("---")
        st.markdown("**💡 Key Insights**")

        insights = []

        # Revenue vs Funding insight
        if revenue >= funding:
            insights.append("✅ Revenue exceeds total funding — excellent unit economics.")
        elif revenue >= funding * 0.5:
            insights.append("📈 Revenue is 50%+ of funding — on the right track.")
        else:
            insights.append("⚠️ Revenue is well below total funding — high burn risk.")

        # Valuation insight
        if val_to_funding >= 10:
            insights.append("🚀 Valuation is " + str(round(val_to_funding, 1)) + "x funding — outstanding investor confidence.")
        elif val_to_funding >= 5:
            insights.append("📈 Valuation-to-funding of " + str(round(val_to_funding, 1)) + "x is excellent.")
        elif val_to_funding >= 2:
            insights.append("✅ Valuation-to-funding of " + str(round(val_to_funding, 1)) + "x is healthy.")
        else:
            insights.append("⚠️ Low valuation ratio (" + str(round(val_to_funding, 1)) + "x) — needs improvement.")

        # Revenue per employee insight
        if revenue_per_emp >= 1:
            insights.append("✅ Revenue per employee of $" + str(round(revenue_per_emp, 2)) + "M is very strong.")
        elif revenue_per_emp >= 0.3:
            insights.append("📊 Revenue per employee of $" + str(round(revenue_per_emp, 2)) + "M is decent.")
        else:
            insights.append("⚠️ Low revenue per employee — workforce may be oversized.")

        # IPO / Acquisition insight
        if ipo and acquired:
            insights.append("🏆 Both IPO and Acquisition — strongest possible exit signals.")
        elif ipo:
            insights.append("📈 IPO completed — major milestone showing market confidence.")
        elif acquired:
            insights.append("🤝 Acquisition completed — validated by a strategic buyer.")

        # Funding stage insight
        if stage == "IPO":
            insights.append("🌟 IPO stage — company has demonstrated public market readiness.")
        elif stage == "Series C":
            insights.append("📊 Series C — late-stage with strong investor backing.")
        elif stage == "Seed":
            insights.append("🌱 Seed stage — early days, high risk but high potential.")

        # Display each insight
        for insight in insights:
            st.markdown('<div class="advice-card">' + insight + '</div>', unsafe_allow_html=True)

    else:
        # Placeholder shown before clicking Predict
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: rgba(148,163,184,0.5);">
            <div style="font-size:3rem; margin-bottom:1rem;">🔮</div>
            <p style="font-size:1rem; margin:0;">
                Fill in your startup details on the left<br>
                and click <strong style="color:#63b3ed">Predict</strong>
                to get your AI-powered result
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding: 1.5rem;
            border-top: 1px solid rgba(99,179,237,0.1);
            color: rgba(148,163,184,0.5); font-size:0.8rem;">
    🚀 <strong style="color:#63b3ed">AI Startup Predictor</strong>
    &nbsp;&middot;&nbsp; Built with Streamlit
    &nbsp;&middot;&nbsp; Powered by Advanced Machine Learning
    &nbsp;&middot;&nbsp; Trained on 5,000 global startups
</div>
""", unsafe_allow_html=True)
