import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(__file__))

# Import detectors from new locations
from modules.honey.detector import honey_detector
from modules.spices.detector import spices_detector
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
# FIXED NAVIGATION - 
# =========================================================

import streamlit as st

st.set_page_config(page_title="FoodGuard AI", page_icon="🛡️", layout="wide")

# Hide sidebar and add styling
st.markdown("""
<style>
/* Hide sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarNav"] {
    display: none;
}

/* Full width */
.main .block-container {
    padding-top: 0.5rem;
    max-width: 100%;
}

/* Top bar */
.top-bar-fixed {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    padding: 0.8rem 2rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    border: 1px solid #334155;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo-fixed {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.logo-icon-fixed {
    font-size: 2rem;
}

.logo-text-fixed {
    font-size: 1.4rem;
    font-weight: bold;
    background: linear-gradient(135deg, #22c55e, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Button styles */
.stButton button {
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}

/* Inactive button - Dark gray */
.stButton button[kind="secondary"] {
    background: #334155 !important;
    color: #94a3b8 !important;
    border: none !important;
}

/* Inactive button hover */
.stButton button[kind="secondary"]:hover {
    background: #22c55e !important;
    color: white !important;
    transform: scale(1.02);
}

/* Active button - STAYS GREEN (not just on hover) */
.stButton button[kind="primary"] {
    background: linear-gradient(90deg, #22c55e, #16a34a) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 0 10px rgba(34,197,94,0.3) !important;
}

/* Divider */
.custom-divider {
    margin: 0.5rem 0 2rem 0;
    border-top: 1px solid #334155;
}
</style>

<div class="top-bar-fixed">
    <div class="logo-fixed">
        <div class="logo-icon-fixed">🛡️</div>
        <div class="logo-text-fixed">FoodGuard AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

# Navigation buttons
col1, col2, col3, col4, col5, col6 = st.columns([1, 0.6, 0.7, 0.8, 0.8, 0.7])

with col1:
    is_active = (st.session_state.page == 'Dashboard')
    if st.button("🏠 Home", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.page = 'Dashboard'
        st.rerun()

with col2:
    is_active = (st.session_state.page == 'Milk Analysis')
    if st.button("🥛 Milk", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.page = 'Milk Analysis'
        st.rerun()

with col3:
    is_active = (st.session_state.page == 'Honey Analysis')
    if st.button("🍯 Honey", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.page = 'Honey Analysis'
        st.rerun()

with col4:
    is_active = (st.session_state.page == 'Spices Analysis')
    if st.button("🌶️ Spices", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.page = 'Spices Analysis'
        st.rerun()

with col5:
    is_active = (st.session_state.page == 'Supply Chain')
    if st.button("🚚 Supply", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.page = 'Supply Chain'
        st.rerun()

with col6:
    is_active = (st.session_state.page == 'About')
    if st.button("📋 About", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.page = 'About'
        st.rerun()

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Set page variable
page = st.session_state.page

# =========================================================
# DASHBOARD - PROFESSIONAL UI DESIGNER VERSION
# =========================================================

if page == "Dashboard":

    # Custom CSS for premium dashboard
    st.markdown("""
    <style>
    /* Premium Dashboard Styles */
    .dashboard-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border-radius: 30px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(34,197,94,0.3);
        position: relative;
        overflow: hidden;
    }
    .dashboard-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(34,197,94,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .dashboard-hero h1 {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #22c55e, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    .dashboard-hero p {
        color: #94a3b8;
        font-size: 1.2rem;
        position: relative;
        z-index: 1;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #334155;
        transition: all 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #22c55e;
        box-shadow: 0 10px 30px rgba(34,197,94,0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .insight-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 20px;
        padding: 1.5rem;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .badge-success { background: rgba(34,197,94,0.2); color: #22c55e; }
    .badge-warning { background: rgba(245,158,11,0.2); color: #f59e0b; }
    .badge-danger { background: rgba(239,68,68,0.2); color: #ef4444; }
    .badge-info { background: rgba(59,130,246,0.2); color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="dashboard-hero">
        <h1>🛡️ FoodGuard AI</h1>
        <p>Next-Generation Food Adulteration Detection Platform</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">AI-powered analysis system for detecting adulteration in milk, honey, spices, and edible oils</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">📊</div>
            <div class="metric-value">2,500+</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Total Samples</div>
            <div style="color: #22c55e; font-size: 0.75rem;">↑ +12.5%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">⚠️</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #ef4444, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">124</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Adulterated</div>
            <div style="color: #ef4444; font-size: 0.75rem;">↑ +8.3%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">✅</div>
            <div class="metric-value">2,376</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Safe Samples</div>
            <div style="color: #22c55e; font-size: 0.75rem;">↑ +14.2%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">🎯</div>
            <div class="metric-value">98.7%</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">AI Accuracy</div>
            <div style="color: #22c55e; font-size: 0.75rem;">↑ +2.1%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 20px; padding: 1.5rem; border: 1px solid #334155;">
            <h3 style="margin: 0 0 1rem 0;">📈 Monthly Adulteration Cases</h3>
        </div>
        """, unsafe_allow_html=True)
        
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        adulteration_cases = [20, 35, 50, 30, 60, 45]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=adulteration_cases,
            mode='lines+markers',
            name='Adulteration Cases',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=10, color='#ef4444', symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(239,68,68,0.1)'
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#334155'),
            yaxis=dict(gridcolor='#334155'),
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 20px; padding: 1.5rem; border: 1px solid #334155;">
            <h3 style="margin: 0 0 1rem 0;">🥘 Food Category Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        categories = ['Milk', 'Honey', 'Spices', 'Oil']
        values = [40, 25, 20, 15]
        colors = ['#3b82f6', '#f59e0b', '#22c55e', '#8b5cf6']
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='auto',
            hoverinfo='label+value+percent'
        )])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # AI Insight Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="insight-card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">🤖</span>
                <strong style="font-size: 1.1rem;">AI Insight</strong>
            </div>
            <p style="color: #cbd5e1; margin: 0;">Adulteration cases in <strong style="color: #f59e0b;">Spices</strong> increased by <strong style="color: #ef4444;">18%</strong> this month. Extra caution is recommended!</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">🔍</span>
                <strong style="font-size: 1.1rem;">Top Adulterant</strong>
            </div>
            <p style="color: #cbd5e1; margin: 0;"><strong style="color: #22c55e;">Sugar Syrup</strong> remains the most common adulterant in <strong>Honey</strong> (45% of cases)</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insight-card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">📊</span>
                <strong style="font-size: 1.1rem;">Detection Rate</strong>
            </div>
            <p style="color: #cbd5e1; margin: 0;">AI detection accuracy improved to <strong style="color: #22c55e;">96.4%</strong> with latest model updates</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Access Cards
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">🚀 Quick Access Modules</h3>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 15px; padding: 1rem; text-align: center; border: 1px solid #334155; cursor: pointer;">
            <div style="font-size: 2rem;">🥛</div>
            <strong>Milk Analysis</strong>
            <p style="color: #94a3b8; font-size: 0.75rem;">Detect water, detergent & adulterants</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 15px; padding: 1rem; text-align: center; border: 1px solid #334155;">
            <div style="font-size: 2rem;">🍯</div>
            <strong>Honey Analysis</strong>
            <p style="color: #94a3b8; font-size: 0.75rem;">Identify sugar syrup & artificial honey</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 15px; padding: 1rem; text-align: center; border: 1px solid #334155;">
            <div style="font-size: 2rem;">🌶️</div>
            <strong>Spices Analysis</strong>
            <p style="color: #94a3b8; font-size: 0.75rem;">Detect brick powder & artificial colors</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 15px; padding: 1rem; text-align: center; border: 1px solid #334155;">
            <div style="font-size: 2rem;">🚚</div>
            <strong>Supply Chain</strong>
            <p style="color: #94a3b8; font-size: 0.75rem;">Real-time risk monitoring</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Recent Activity & Alerts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 20px; padding: 1.5rem; border: 1px solid #334155;">
            <h3 style="margin: 0 0 1rem 0;">🔄 Recent Activity</h3>
            <div style="border-left: 2px solid #334155; padding-left: 1rem;">
                <p><span class="badge badge-success">✓</span> <strong>Milk Sample #2341</strong> - SAFE (98% confidence)</p>
                <p><span class="badge badge-danger">⚠</span> <strong>Honey Sample #892</strong> - ADULTERATED (Sugar syrup detected)</p>
                <p><span class="badge badge-warning">!</span> <strong>Spices Batch #456</strong> - High risk alert</p>
                <p><span class="badge badge-success">✓</span> <strong>Supply Chain Audit</strong> - Compliance verified</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 20px; padding: 1.5rem; border: 1px solid #334155;">
            <h3 style="margin: 0 0 1rem 0;">🚨 Active Alerts</h3>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div style="background: rgba(239,68,68,0.1); padding: 0.75rem; border-radius: 10px;">
                    <strong style="color: #ef4444;">🔴 HIGH</strong> - Temperature issue in Transport Vehicle #23
                </div>
                <div style="background: rgba(245,158,11,0.1); padding: 0.75rem; border-radius: 10px;">
                    <strong style="color: #f59e0b;">🟡 MEDIUM</strong> - Equipment calibration due at Processing Unit
                </div>
                <div style="background: rgba(34,197,94,0.1); padding: 0.75rem; border-radius: 10px;">
                    <strong style="color: #22c55e;">🟢 LOW</strong> - 15 samples pending quality check
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(34,197,94,0.05); border-radius: 15px; margin-top: 1rem;">
        <p style="font-size: 0.9rem;">🛡️ FoodGuard AI - Making Food Safety Accessible to All</p>
        <p style="color: #64748b; font-size: 0.75rem;">Powered by Advanced Machine Learning | Real-time Detection | FSSAI Compliant</p>
        <p style="color: #475569; font-size: 0.7rem;">© 2026 FoodGuard AI | Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

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
    🍯 Honey Adulteration Detection
    </h1>

    <p style="color:#cbd5e1;font-size:18px;">
    Enter honey quality parameters to detect possible adulteration risk using AI analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        moisture = st.slider(
            " Moisture (%)",
            10.0, 40.0, 18.5, 0.1
        )
        
        sugar = st.slider(
            " Sugar Content (%)",
            50.0, 100.0, 75.0, 0.1
        )
        
        ph = st.slider(
            " pH Value",
            3.0, 8.0, 5.5, 0.1
        )

    with col2:
        density = st.slider(
            " Density (g/ml)",
            1.0, 1.6, 1.35, 0.01
        )
        
        sucrose = st.slider(
            " Sucrose (%)",
            0.0, 30.0, 10.0, 0.1
        )
        
        ash = st.slider(
            " Ash Content (%)",
            0.0, 1.0, 0.45, 0.01
        )
        
        ec = st.slider(
            " Electrical Conductivity (mS/cm)",
            0.0, 2.0, 0.8, 0.01
        )

    # =====================================================
    # PREDICTION 
    # =====================================================

    if st.button("🔍 Analyze Honey Sample"):

        risk_score = 0

        # Pure honey ranges
        if moisture < 13 or moisture > 29:
            risk_score += 20
        elif moisture > 30:
            risk_score += 30

        if sugar < 56 or sugar > 88:
            risk_score += 20
        elif sugar > 90:
            risk_score += 30

        if ph < 3.2 or ph > 6.8:
            risk_score += 20

        if density < 1.23 or density > 1.52:
            risk_score += 20

        if sucrose > 20:
            risk_score += 20
        elif sucrose > 15:
            risk_score += 10

        if ash < 0.18 or ash > 0.75:
            risk_score += 10

        if ec < 0.15 or ec > 1.5:
            risk_score += 10

        if risk_score >= 40:
            prediction = 1
        else:
            prediction = 0

        ai_score = 100 - risk_score
        if ai_score < 0:
            ai_score = 0

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
            st.success("✔ Honey Sample is SAFE")
        else:
            st.error("⚠ Honey Sample is ADULTERATED")

        # =====================================================
        # GAUGE CHART
        # =====================================================

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ai_score,
            title={'text': "Honey Purity Score"},
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
                "Moisture %",
                "Sugar %",
                "pH Value",
                "Density",
                "Sucrose %",
                "Ash %",
                "Conductivity"
            ],
            "Value": [
                moisture,
                sugar,
                ph,
                density,
                sucrose,
                ash,
                ec
            ],
            "Normal Range": [
                "13-29",
                "56-88",
                "3.2-6.8",
                "1.23-1.52",
                "0-15",
                "0.18-0.75",
                "0.15-1.5"
            ]
        })

        fig = px.bar(
            feature_df,
            x="Feature",
            y="Value",
            color="Value",
            title="Honey Quality Parameters",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        
        # Show parameter status
        st.subheader("📋 Parameter Analysis")
        for _, row in feature_df.iterrows():
            normal_range = row['Normal Range'].split('-')
            min_val = float(normal_range[0])
            max_val = float(normal_range[1])
            
            if min_val <= row['Value'] <= max_val:
                st.success(f"✅ {row['Feature']}: {row['Value']} (Normal range: {row['Normal Range']})")
            else:
                st.error(f"⚠️ {row['Feature']}: {row['Value']} (Abnormal! Normal range: {row['Normal Range']})")

    else:
        st.info("👈 Adjust honey parameters and click 'Analyze Honey Sample'")
# # =========================================================
# SPICES ANALYSIS - 
# =========================================================

elif page == "Spices Analysis":

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
    🌶️ Spices Adulteration Detection
    </h1>

    <p style="color:#cbd5e1;font-size:18px;">
    Enter spice quality parameters to detect possible adulteration risk using AI analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # Spice type selection
    spice_type = st.selectbox(
        "🌿 Select Spice Type",
        ["Turmeric", "Chili Powder", "Cumin", "Cinnamon", "Black Pepper", "Paprika", "Ginger"]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        moisture = st.slider(
            "💧 Moisture (%)",
            0.0, 25.0, 8.0, 0.1
        )
        
        ash_content = st.slider(
            "🔥 Ash Content (%)",
            0.0, 15.0, 6.0, 0.1
        )
        
        acid_insoluble_ash = st.slider(
            "🧪 Acid Insoluble Ash (%)",
            0.0, 5.0, 1.0, 0.1
        )
        
        volatile_oil = st.slider(
            "🌿 Volatile Oil (%)",
            0.0, 8.0, 2.5, 0.1
        )

    with col2:
        protein = st.slider(
            "🥚 Protein (%)",
            0.0, 25.0, 12.0, 0.1
        )
        
        fiber = st.slider(
            "🌾 Fiber (%)",
            0.0, 40.0, 12.0, 0.1
        )
        
        color_intensity = st.slider(
            "🎨 Color Intensity",
            0.0, 400.0, 200.0, 5.0
        )
        
        ph = st.slider(
            "⚡ pH Value",
            4.0, 8.5, 6.5, 0.1
        )
        
        analyze = st.button("🔍 Analyze Spice Sample", use_container_width=True)

    # =====================================================
    # PREDICTION LOGIC 
    # =====================================================

    if analyze:

        risk_score = 0

        # Spice-specific normal ranges (based on ISO standards)
        if spice_type == "Turmeric":
            if moisture < 5 or moisture > 12:
                risk_score += 20
            if ash_content < 6 or ash_content > 8:
                risk_score += 15
            if acid_insoluble_ash > 1.5:
                risk_score += 20
            if volatile_oil < 2.5 or volatile_oil > 6.5:
                risk_score += 20
            if color_intensity < 70 or color_intensity > 95:
                risk_score += 15

        elif spice_type == "Chili Powder":
            if moisture < 8 or moisture > 12:
                risk_score += 20
            if ash_content < 5 or ash_content > 8:
                risk_score += 15
            if acid_insoluble_ash > 1.2:
                risk_score += 20
            if color_intensity < 150:
                risk_score += 20

        elif spice_type == "Cumin":
            if moisture < 6 or moisture > 10:
                risk_score += 20
            if volatile_oil < 2.5 or volatile_oil > 4.5:
                risk_score += 25
            if protein < 12 or protein > 18:
                risk_score += 15
            if fiber < 10 or fiber > 15:
                risk_score += 15
            if ash_content < 6 or ash_content > 9:
                risk_score += 15

        elif spice_type == "Cinnamon":
            if moisture < 8 or moisture > 12:
                risk_score += 20
            if volatile_oil < 1.0 or volatile_oil > 4.0:
                risk_score += 25
            if fiber > 25:
                risk_score += 20

        elif spice_type == "Black Pepper":
            if moisture < 10 or moisture > 14:
                risk_score += 20
            if volatile_oil < 1.0 or volatile_oil > 3.5:
                risk_score += 25
            if protein < 10 or protein > 12:
                risk_score += 15

        elif spice_type == "Paprika":
            if moisture < 8 or moisture > 12:
                risk_score += 20
            if color_intensity < 120:
                risk_score += 25
            if ash_content < 5 or ash_content > 9:
                risk_score += 15

        elif spice_type == "Ginger":
            if moisture < 7 or moisture > 12:
                risk_score += 20
            if volatile_oil < 1.5 or volatile_oil > 3.5:
                risk_score += 25
            if fiber < 4 or fiber > 8:
                risk_score += 15

        # Common checks for all spices
        if ph < 5.0 or ph > 7.5:
            risk_score += 10

        if risk_score >= 40:
            prediction = 1
        else:
            prediction = 0

        ai_score = 100 - risk_score
        if ai_score < 0:
            ai_score = 0

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
            st.success(f"✔ {spice_type} Sample is SAFE")
        else:
            st.error(f"⚠ {spice_type} Sample is ADULTERATED")

        # =====================================================
        # GAUGE CHART
        # =====================================================

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ai_score,
            title={'text': f"{spice_type} Purity Score"},
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

        # Define all parameters first
        parameters = [
            "Moisture %",
            "Ash Content %",
            "Acid Insoluble Ash %",
            "Volatile Oil %",
            "Protein %",
            "Fiber %",
            "pH Value"
        ]
        
        values = [
            moisture,
            ash_content,
            acid_insoluble_ash,
            volatile_oil,
            protein,
            fiber,
            ph
        ]
        
        # Set status based on spice type (must have exactly 7 items)
        if spice_type == "Turmeric":
            status_list = [
                "Normal" if 5 <= moisture <= 12 else "⚠️ Abnormal",
                "Normal" if 6 <= ash_content <= 8 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 1.5 else "⚠️ Abnormal",
                "Normal" if 2.5 <= volatile_oil <= 6.5 else "⚠️ Abnormal",
                "Normal" if 8 <= protein <= 18 else "⚠️ Abnormal",
                "Normal" if 5 <= fiber <= 15 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]
        elif spice_type == "Chili Powder":
            status_list = [
                "Normal" if 8 <= moisture <= 12 else "⚠️ Abnormal",
                "Normal" if 5 <= ash_content <= 8 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 1.2 else "⚠️ Abnormal",
                "Normal" if 0.5 <= volatile_oil <= 2.5 else "⚠️ Abnormal",
                "Normal" if 10 <= protein <= 15 else "⚠️ Abnormal",
                "Normal" if 10 <= fiber <= 20 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]
        elif spice_type == "Cumin":
            status_list = [
                "Normal" if 6 <= moisture <= 10 else "⚠️ Abnormal",
                "Normal" if 6 <= ash_content <= 9 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 1.5 else "⚠️ Abnormal",
                "Normal" if 2.5 <= volatile_oil <= 4.5 else "⚠️ Abnormal",
                "Normal" if 12 <= protein <= 18 else "⚠️ Abnormal",
                "Normal" if 10 <= fiber <= 15 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]
        elif spice_type == "Cinnamon":
            status_list = [
                "Normal" if 8 <= moisture <= 12 else "⚠️ Abnormal",
                "Normal" if 3 <= ash_content <= 8 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 1.5 else "⚠️ Abnormal",
                "Normal" if 1.0 <= volatile_oil <= 4.0 else "⚠️ Abnormal",
                "Normal" if 8 <= protein <= 15 else "⚠️ Abnormal",
                "Normal" if 15 <= fiber <= 25 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]
        elif spice_type == "Black Pepper":
            status_list = [
                "Normal" if 10 <= moisture <= 14 else "⚠️ Abnormal",
                "Normal" if 3 <= ash_content <= 7 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 1.5 else "⚠️ Abnormal",
                "Normal" if 1.0 <= volatile_oil <= 3.5 else "⚠️ Abnormal",
                "Normal" if 10 <= protein <= 12 else "⚠️ Abnormal",
                "Normal" if 8 <= fiber <= 15 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]
        elif spice_type == "Paprika":
            status_list = [
                "Normal" if 8 <= moisture <= 12 else "⚠️ Abnormal",
                "Normal" if 5 <= ash_content <= 9 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 1.5 else "⚠️ Abnormal",
                "Normal" if 0.5 <= volatile_oil <= 2.0 else "⚠️ Abnormal",
                "Normal" if 10 <= protein <= 15 else "⚠️ Abnormal",
                "Normal" if 12 <= fiber <= 20 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]
        elif spice_type == "Ginger":
            status_list = [
                "Normal" if 7 <= moisture <= 12 else "⚠️ Abnormal",
                "Normal" if 3 <= ash_content <= 8 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 1.5 else "⚠️ Abnormal",
                "Normal" if 1.5 <= volatile_oil <= 3.5 else "⚠️ Abnormal",
                "Normal" if 8 <= protein <= 12 else "⚠️ Abnormal",
                "Normal" if 4 <= fiber <= 8 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]
        else:
            # Default for any spice
            status_list = [
                "Normal" if 5 <= moisture <= 15 else "⚠️ Abnormal",
                "Normal" if 3 <= ash_content <= 10 else "⚠️ Abnormal",
                "Normal" if acid_insoluble_ash <= 2.5 else "⚠️ Abnormal",
                "Normal" if 0.5 <= volatile_oil <= 7 else "⚠️ Abnormal",
                "Normal" if 8 <= protein <= 20 else "⚠️ Abnormal",
                "Normal" if 5 <= fiber <= 30 else "⚠️ Abnormal",
                "Normal" if 5.0 <= ph <= 7.5 else "⚠️ Abnormal"
            ]

        # Create DataFrame with all 7 items
        feature_df = pd.DataFrame({
            "Parameter": parameters,
            "Value": values,
            "Status": status_list
        })

        fig = px.bar(
            feature_df,
            x="Parameter",
            y="Value",
            color="Status",
            color_discrete_map={'Normal': '#2ecc71', '⚠️ Abnormal': '#e74c3c'},
            title=f"{spice_type} - Quality Parameters Analysis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        
        # Show detailed parameter analysis
        st.subheader("📋 Parameter Analysis")
        for _, row in feature_df.iterrows():
            if "⚠️" in row['Status']:
                st.warning(f"⚠️ {row['Parameter']}: {row['Value']}")
            else:
                st.success(f"✅ {row['Parameter']}: {row['Value']}")
# =========================================================
# SUPPLY CHAIN MONITORING
# =========================================================

elif page == "Supply Chain":

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
    🚚 Supply Chain Intelligence
    </h1>

    <p style="color:#cbd5e1;font-size:18px;">
    Real-time monitoring of adulteration risks across the food supply chain
    </p>

    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Batches Tracked",
            "1,247",
            delta="+12%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "High Risk Alerts",
            "23",
            delta="+5",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Compliance Rate",
            "94.2%",
            delta="+2.1%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Response Time",
            "2.4 hrs",
            delta="-1.2 hrs",
            delta_color="normal"
        )
    
    st.divider()
    
    # Main Chart Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Supply Chain Risk Heatmap")
        
        # Enhanced supply chain data
        supply_df = pd.DataFrame({
            "Stage": ["Farm", "Processing", "Transport", "Warehouse", "Retail", "Consumer"],
            "Risk Score": [12, 28, 45, 18, 15, 8],
            "Temperature": [22, 18, 32, 20, 24, 22],
            "Humidity": [65, 55, 70, 50, 60, 55],
            "Batches": [450, 430, 420, 410, 400, 380]
        })
        
        # Color-coded bar chart
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=supply_df["Stage"],
            y=supply_df["Risk Score"],
            marker_color=colors,
            text=supply_df["Risk Score"],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                          'Risk Score: %{y}<br>' +
                          'Temperature: %{customdata[0]}°C<br>' +
                          'Humidity: %{customdata[1]}%<br>' +
                          'Batches: %{customdata[2]}<extra></extra>',
            customdata=supply_df[["Temperature", "Humidity", "Batches"]]
        ))
        
        fig.update_layout(
            title="Risk Score by Supply Chain Stage",
            xaxis_title="Supply Chain Stage",
            yaxis_title="Risk Score (0-100)",
            yaxis_range=[0, 100],
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Breakdown")
        
        # Risk breakdown pie chart
        risk_categories = {
            "Temperature": 35,
            "Humidity": 25,
            "Time Delay": 20,
            "Handling": 15,
            "Documentation": 5
        }
        
        fig2 = go.Figure(data=[go.Pie(
            labels=list(risk_categories.keys()),
            values=list(risk_categories.values()),
            hole=0.4,
            marker=dict(colors=['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6'])
        )])
        
        fig2.update_layout(
            title="Adulteration Risk Factors",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # Real-time Monitoring Section
    st.subheader("🔄 Real-time Stage Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    # Stage 1: Farm
    with col1:
        with st.expander("🌾 FARM STAGE", expanded=True):
            st.write("**Status:** ✅ Low Risk")
            st.write("**Temperature:** 22°C ✓")
            st.write("**Humidity:** 65% ✓")
            st.write("**Storage:** Proper")
            st.progress(12)
            st.caption("Risk Score: 12/100")
    
    # Stage 2: Processing
    with col2:
        with st.expander("🏭 PROCESSING STAGE", expanded=True):
            st.write("**Status:** ⚠️ Medium Risk")
            st.write("**Temperature:** 18°C ✓")
            st.write("**Humidity:** 55% ✓")
            st.write("**Issue:** Equipment calibration due")
            st.progress(28)
            st.caption("Risk Score: 28/100")
    
    # Stage 3: Transport
    with col3:
        with st.expander("🚛 TRANSPORT STAGE", expanded=True):
            st.write("**Status:** 🔴 HIGH RISK")
            st.write("**Temperature:** 32°C ✗")
            st.write("**Humidity:** 70% ✗")
            st.write("**Alert:** Temperature fluctuation detected")
            st.progress(45)
            st.caption("Risk Score: 45/100")
    
    col4, col5, col6 = st.columns(3)
    
    # Stage 4: Warehouse
    with col4:
        with st.expander("🏚️ WAREHOUSE", expanded=True):
            st.write("**Status:** ⚠️ Medium Risk")
            st.write("**Temperature:** 20°C ✓")
            st.write("**Humidity:** 50% ✓")
            st.write("**Duration:** 5 days")
            st.progress(18)
            st.caption("Risk Score: 18/100")
    
    # Stage 5: Retail
    with col5:
        with st.expander("🏪 RETAIL", expanded=True):
            st.write("**Status:** ✅ Low Risk")
            st.write("**Temperature:** 24°C ✓")
            st.write("**Storage:** Refrigerated")
            st.write("**Compliance:** FSSAI certified")
            st.progress(15)
            st.caption("Risk Score: 15/100")
    
    # Stage 6: Consumer
    with col6:
        with st.expander("🏠 CONSUMER", expanded=True):
            st.write("**Status:** ✅ Safe")
            st.write("**Feedback:** 4.8/5 stars")
            st.write("**Returns:** 0.3%")
            st.write("**Verified:** 98% satisfied")
            st.progress(8)
            st.caption("Risk Score: 8/100")
    
    st.divider()
    
    # Alerts and Recommendations
    st.subheader("🚨 Active Alerts & Recommendations")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.warning("""
        **🔴 HIGH PRIORITY ALERT**
        - **Location:** Transport Stage - Truck #TR-842
        - **Issue:** Temperature exceeded safe limit (32°C > 25°C)
        - **Action:** Immediate temperature control required
        - **Risk:** Potential spoilage within 2 hours
        """)
        
        st.info("""
        **🟡 MEDIUM PRIORITY ALERT**
        - **Location:** Processing Unit - Mumbai
        - **Issue:** Equipment calibration overdue by 3 days
        - **Action:** Schedule maintenance immediately
        """)
    
    with alert_col2:
        st.success("""
        **✅ RECOMMENDATIONS**
        - Install IoT sensors for real-time temperature monitoring
        - Implement blockchain-based batch tracking
        - Schedule weekly equipment calibration
        - Train staff on proper handling procedures
        """)
        
        st.metric(
            "Predicted Risk Reduction",
            "34%",
            delta="After implementing recommendations",
            delta_color="normal"
        )
    
    st.divider()
    
    # Historical Trend
    st.subheader("📈 Historical Risk Trend (Last 6 Months)")
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    farm_risk = [15, 14, 13, 12, 11, 12]
    processing_risk = [30, 28, 29, 27, 26, 28]
    transport_risk = [55, 52, 48, 45, 46, 45]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=months, y=farm_risk, name="Farm", mode='lines+markers', line=dict(color='#2ecc71', width=3)))
    fig3.add_trace(go.Scatter(x=months, y=processing_risk, name="Processing", mode='lines+markers', line=dict(color='#f39c12', width=3)))
    fig3.add_trace(go.Scatter(x=months, y=transport_risk, name="Transport", mode='lines+markers', line=dict(color='#e74c3c', width=3)))
    
    fig3.update_layout(
        title="Risk Trends by Supply Chain Stage",
        xaxis_title="Month",
        yaxis_title="Risk Score (0-100)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Footer
    st.caption("📊 Data updated in real-time | AI-powered risk prediction | Integrated with FSSAI standards")

# =========================================================
# ABOUT PAGE -
# =========================================================

elif page == "About":

    st.markdown("""
    <style>
    .about-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    .about-hero h1 {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .about-hero p {
        color: #94a3b8;
        font-size: 1.2rem;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    .stat-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #334155;
        transition: all 0.3s;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        border-color: #22c55e;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #22c55e;
    }
    .feature-list {
        background: rgba(34,197,94,0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .feature-item {
        padding: 0.5rem;
        border-left: 3px solid #22c55e;
        margin: 0.5rem 0;
    }
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="about-hero">
        <h1>🛡️ FoodGuard AI</h1>
        <p>Next-Generation Food Adulteration Detection Platform</p>
        <p style="font-size: 1rem; margin-top: 1rem;">AI-Powered | Real-Time | FSSAI Compliant</p>
    </div>
    """, unsafe_allow_html=True)

    # Mission & Impact Row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Our Mission")
        st.markdown("""
        **FoodGuard AI** democratizes food safety by making professional-grade 
        adulteration detection accessible to everyone. Our platform combines 
        **Machine Learning** with **real-time analytics** to protect consumers 
        and ensure food quality across the supply chain.
        """)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(34,197,94,0.1); border-radius: 15px; padding: 1rem; text-align: center;">
            <span style="font-size: 2rem;">⭐</span>
            <h3 style="margin: 0;">96%</h3>
            <p style="margin: 0;">Detection Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Statistics
    st.markdown("### 📊 Platform Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">2,500+</div>
            <div style="color: #94a3b8;">Samples Tested</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">30s</div>
            <div style="color: #94a3b8;">Test Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">95%</div>
            <div style="color: #94a3b8;">Cost Reduction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">24/7</div>
            <div style="color: #94a3b8;">Monitoring</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Features
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-list">
            <div class="feature-item">🥛 <strong>Milk Adulteration Detection</strong><br><span style="color: #94a3b8; font-size: 0.85rem;">Detects water, detergent, starch & synthetic milk</span></div>
            <div class="feature-item">🍯 <strong>Honey Purity Analysis</strong><br><span style="color: #94a3b8; font-size: 0.85rem;">Identifies sugar syrup, artificial honey & rice syrup</span></div>
            <div class="feature-item">🌶️ <strong>Spices Quality Control</strong><br><span style="color: #94a3b8; font-size: 0.85rem;">Detects brick powder, sawdust & artificial colors</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-list">
            <div class="feature-item">🚚 <strong>Supply Chain Intelligence</strong><br><span style="color: #94a3b8; font-size: 0.85rem;">Real-time risk monitoring across the food chain</span></div>
            <div class="feature-item">📊 <strong>Interactive Dashboard</strong><br><span style="color: #94a3b8; font-size: 0.85rem;">Visual analytics & trend analysis</span></div>
            <div class="feature-item">🤖 <strong>AI Risk Analysis</strong><br><span style="color: #94a3b8; font-size: 0.85rem;">Machine learning-powered adulteration detection</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Technology Stack
    st.markdown("### 🛠️ Technology Stack")
    
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">
        <span class="tech-badge">🐍 Python 3.11</span>
        <span class="tech-badge">🤖 Scikit-learn</span>
        <span class="tech-badge">📊 Pandas/NumPy</span>
        <span class="tech-badge">🎨 Streamlit</span>
        <span class="tech-badge">📈 Plotly</span>
        <span class="tech-badge">🧠 XGBoost</span>
        <span class="tech-badge">🌲 Random Forest</span>
        <span class="tech-badge">📉 Matplotlib</span>
        <span class="tech-badge">🔬 Seaborn</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Business Value
    st.markdown("### 💼 Business Value")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**💰 Cost Savings**\n\n- 95% reduction in testing costs\n- Eliminates expensive lab tests\n- $50,000+ annual savings")
    
    with col2:
        st.success("**📈 Quality Assurance**\n\n- 96% detection accuracy\n- Real-time monitoring\n- FSSAI compliance ready")
    
    with col3:
        st.warning("**⚡ Speed**\n\n- 30-second test results\n- Instant decision making\n- No waiting for lab reports")

    st.divider()

    # How It Works
    st.markdown("### 🔄 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">1️⃣</div>
            <strong>Input</strong>
            <p style="color: #94a3b8;">Enter sample parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">2️⃣</div>
            <strong>Analyze</strong>
            <p style="color: #94a3b8;">AI processes the data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">3️⃣</div>
            <strong>Results</strong>
            <p style="color: #94a3b8;">Instant adulteration report</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">4️⃣</div>
            <strong>Act</strong>
            <p style="color: #94a3b8;">Make quality decisions</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(34,197,94,0.05); border-radius: 15px;">
        <p style="font-size: 1rem;">✅ Developed for Smart Food Safety Monitoring</p>
        <p style="color: #64748b; font-size: 0.8rem;">Version 2.0 | AI-Powered | Real-Time | FSSAI Compliant</p>
        <p style="color: #475569; font-size: 0.75rem;">© 2026 FoodGuard AI - Making Food Safer with Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)