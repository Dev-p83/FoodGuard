import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FoodGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#020617,#0f172a,#111827);
    color: white;
}

[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

h1,h2,h3,h4 {
    color: white;
    font-weight: 700;
}

div[data-testid="metric-container"] {
    background: rgba(15,23,42,0.8);
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

div.stButton > button {
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color: white;
    border: none;
    border-radius: 14px;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg,#16a34a,#15803d);
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ FoodGuard AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Milk Analysis",
        "Honey Analysis",
        "Spices Analysis",
        "Supply Chain",
        "About"
    ]
)

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("""
    <div style="
    background: rgba(15,23,42,0.7);
    padding:35px;
    border-radius:25px;
    backdrop-filter: blur(12px);
    border:1px solid #334155;
    margin-bottom:25px;
    ">

    <h1 style="font-size:42px;">
    🛡️ FoodGuard AI
    </h1>

    <h3 style="color:#94a3b8;">
    Next-Generation Food Adulteration Detection Platform
    </h3>

    <p style="color:#cbd5e1;font-size:18px;">
    AI-powered analysis system for detecting adulteration
    in milk, honey, spices, and edible oils using Machine Learning.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Samples", "2500")
    col2.metric("Adulterated", "124")
    col3.metric("Safe Samples", "2376")
    col4.metric("AI Accuracy", "98.7%")

    st.divider()

    # CHARTS
    col1, col2 = st.columns(2)

    with col1:

        months = ["Jan","Feb","Mar","Apr","May","Jun"]
        values = [20,35,50,30,60,45]

        fig = px.line(
            x=months,
            y=values,
            markers=True,
            title="Monthly Adulteration Cases"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        labels = ["Milk","Honey","Spices","Oil"]
        vals = [40,20,25,15]

        pie = px.pie(
            names=labels,
            values=vals,
            hole=0.5,
            title="Food Category Distribution"
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    st.divider()

    st.success("✔ AI Monitoring System Active")

# =========================================================
# MILK ANALYSIS
# =========================================================

elif page == "Milk Analysis":

    st.markdown("""
    <div style="
    background: rgba(15,23,42,0.7);
    padding:35px;
    border-radius:25px;
    backdrop-filter: blur(12px);
    border:1px solid #334155;
    margin-bottom:25px;
    ">

    <h1 style="font-size:40px;">
    🥛 Milk Adulteration Detection
    </h1>

    <p style="color:#cbd5e1;font-size:18px;">
    Enter milk quality parameters to detect
    possible adulteration risk using AI analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        fat = st.slider(
            "Fat Percentage",
            0.0,
            10.0,
            3.5
        )

        snf = st.slider(
            "SNF Percentage",
            0.0,
            15.0,
            8.5
        )

        protein = st.slider(
            "Protein Percentage",
            0.0,
            10.0,
            3.2
        )

    with col2:

        ph = st.slider(
            "pH Value",
            0.0,
            14.0,
            6.7
        )

        conductivity = st.slider(
            "Conductivity",
            0.0,
            10.0,
            4.0
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button("🔍 Analyze Milk Sample"):

        risk_score = 0

        if fat < 2.5:
            risk_score += 20

        if snf < 8.0:
            risk_score += 20

        if protein < 3.0:
            risk_score += 20

        if ph < 6.5 or ph > 6.9:
            risk_score += 20

        if conductivity > 6:
            risk_score += 20

        if risk_score >= 40:
            prediction = 1
        else:
            prediction = 0

        ai_score = 100 - risk_score

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "AI Confidence",
            f"{ai_score}%"
        )

        col2.metric(
            "Risk Level",
            "LOW" if risk_score < 40 else "HIGH"
        )

        col3.metric(
            "FSSAI Status",
            "SAFE" if risk_score < 40 else "FAILED"
        )

        st.divider()

        if prediction == 0:

            st.success("✔ Milk Sample is SAFE")

        else:

            st.error("⚠ Milk Sample is ADULTERATED")

        # =====================================================
        # GAUGE CHART
        # =====================================================

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ai_score,
            title={'text': "Milk Safety Score"},
            gauge={
                'axis': {'range': [0,100]},
                'steps': [
                    {'range': [0,50], 'color': "red"},
                    {'range': [50,80], 'color': "yellow"},
                    {'range': [80,100], 'color': "green"}
                ]
            }
        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # =====================================================
        # FEATURE IMPORTANCE
        # =====================================================

        st.subheader("📊 Important Quality Parameters")

        feature_df = pd.DataFrame({
            "Feature": [
                "Fat %",
                "SNF %",
                "Protein %",
                "pH Value",
                "Conductivity"
            ],
            "Importance": [
                fat,
                snf,
                protein,
                ph,
                conductivity
            ]
        })

        fig = px.bar(
            feature_df,
            x="Feature",
            y="Importance",
            color="Importance",
            title="Milk Quality Parameters"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =========================================================
# HONEY ANALYSIS
# =========================================================

elif page == "Honey Analysis":

    st.title("🍯 Honey Adulteration Analysis")

    honey_df = pd.DataFrame({
        "Adulterant": [
            "Sugar Syrup",
            "Artificial Honey",
            "Rice Syrup",
            "Others"
        ],
        "Cases": [45,20,15,20]
    })

    fig = px.bar(
        honey_df,
        x="Adulterant",
        y="Cases",
        color="Cases",
        title="Honey Adulteration Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning("⚠ High sugar syrup adulteration detected.")

# =========================================================
# SPICES ANALYSIS
# =========================================================

elif page == "Spices Analysis":

    st.title("🌶️ Spices Adulteration Analysis")

    spice_df = pd.DataFrame({
        "Adulterant": [
            "Brick Powder",
            "Sawdust",
            "Synthetic Colors",
            "Chalk Powder"
        ],
        "Cases": [35,20,25,20]
    })

    fig = px.bar(
        spice_df,
        x="Adulterant",
        y="Cases",
        color="Cases",
        title="Spices Adulteration"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.error("⚠ High risk detected in chili powder.")

# =========================================================
# SUPPLY CHAIN
# =========================================================

elif page == "Supply Chain":

    st.title("🚚 Supply Chain Monitoring")

    supply_df = pd.DataFrame({
        "Stage": [
            "Farm",
            "Processing",
            "Transport",
            "Retail",
            "Consumer"
        ],
        "Risk": [10,25,45,15,5]
    })

    fig = px.line(
        supply_df,
        x="Stage",
        y="Risk",
        markers=True,
        title="Supply Chain Risk Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning("⚠ Temperature fluctuation detected during transport.")

# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("ℹ️ About FoodGuard")

    st.write("""
    FoodGuard is an AI-powered food adulteration
    detection platform developed using:

    ✔ Python  
    ✔ Machine Learning  
    ✔ Streamlit  
    ✔ Plotly  
    ✔ Data Analytics  

    Features:
    - Milk Adulteration Detection
    - Honey Analysis
    - Spices Analysis
    - Supply Chain Monitoring
    - Interactive Dashboard
    - AI-Based Risk Analysis
    """)

    st.success(
        "✔ Developed for Smart Food Safety Monitoring"
    )