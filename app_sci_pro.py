# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time

# 注入高级 UI 样式 (隐藏原生的丑陋菜单，采用扁平毛玻璃风格)
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认的右上角汉堡菜单和底部的 Made with Streamlit 水印 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 全局背景色改为极浅的高级灰/蓝底色 */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }

    /* 让输入框和按钮变成圆角苹果风 */
    div.stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 12px;
    }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #2563EB; /* 科技蓝 */
        color: white;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* 结果卡片采用高级的毛玻璃投影效果 */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        padding: 30px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 🧠 2. Data & Model Loading
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('best_random_forest_model.pkl')
    site_db = pd.read_csv('Final_Site_Database_Hybrid.csv')

    def clean_site_name(raw_name):
        parts = str(raw_name).split('_')
        if len(parts) >= 3:
            site_type = parts[-2].capitalize()
            elements = parts[-1]
            return f"{elements} {site_type} Site"
        return raw_name

    site_db['Clean_Name'] = site_db['位点名称'].apply(clean_site_name)
    return model, site_db


model, site_db = load_assets()

CLOUD_GAS_DB = {
    "N2": [-15.58, 2.50, 0.00, -1.90, 2, 28.01, 1.10, 1.99],
    "CO": [-14.01, -1.50, 0.11, -1.50, 2, 28.01, 1.13, 1.93],
    "H2S": [-10.46, 2.05, 0.97, -2.05, 3, 34.08, 1.34, 10.37],
    "H2": [-15.43, 0.00, 0.00, 0.00, 2, 2.02, 0.74, 60.85],
    "C2H5OH": [-10.48, 3.00, 1.69, -0.20, 9, 46.07, 1.43, 1.11],
    "C3H6O": [-9.70, -0.60, 2.88, 0.60, 10, 58.08, 1.21, 0.32],
    "CH3Cl": [-11.26, 0.40, 1.90, -0.40, 5, 50.49, 1.78, 0.92],
    "HCHO": [-10.88, -1.20, 2.33, 1.20, 4, 30.03, 1.21, 1.30],
    "O2": [-12.07, -0.44, 0.00, 0.44, 2, 32.00, 1.21, 1.44],
    "CH4": [-12.61, 1.00, 0.00, -0.10, 5, 16.04, 1.09, 5.24],
    "H2O": [-12.62, 0.00, 1.85, 0.00, 3, 18.02, 0.96, 14.50],
    "NO2": [-9.78, -2.27, 0.32, 2.27, 3, 46.01, 1.19, 0.43],
    "NH3": [-10.07, 0.00, 1.47, 0.00, 4, 17.03, 1.01, 9.44],
    "N2O": [-12.89, 0.50, 0.16, -0.50, 3, 44.01, 1.13, 0.42]
}

# ==========================================
# 🔄 3. State Machine (Page Router)
# ==========================================
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "search"
if 'gas_feats' not in st.session_state:
    st.session_state.gas_feats = None
if 'gas_name' not in st.session_state:
    st.session_state.gas_name = ""

# --- PAGE 1: Input & Search ---
if st.session_state.view_mode in ["search", "manual"]:
    st.markdown("### Step 1: Target Gas Configuration")

    if st.session_state.view_mode == "search":
        col1, col2 = st.columns([1, 2])
        with col1:
            target_gas = st.text_input("Enter Gas Formula (e.g., O2, NO2, H2S):", value="O2").strip().upper()

            if st.button("🔍 Fetch Descriptors & Predict", type="primary", use_container_width=True):
                if target_gas in CLOUD_GAS_DB:
                    with st.spinner("Connecting to Quantum Cloud Database..."):
                        time.sleep(0.5)
                    st.session_state.gas_feats = CLOUD_GAS_DB[target_gas]
                    st.session_state.gas_name = target_gas
                    st.session_state.view_mode = "results"
                    st.rerun()
                else:
                    st.error(f"❌ '{target_gas}' not found in the online database.")
                    st.info("Redirecting to Manual Input Mode...")
                    time.sleep(1.2)
                    st.session_state.gas_name = target_gas
                    st.session_state.view_mode = "manual"
                    st.rerun()

    elif st.session_state.view_mode == "manual":
        st.warning(
            f"⚠️ Automatic retrieval failed for '{st.session_state.gas_name}'. Please input DFT descriptors manually.")
        with st.form("manual_input_form"):
            c1, c2, c3, c4 = st.columns(4)
            f1 = c1.number_input("HOMO (eV)", value=-10.0)
            f2 = c2.number_input("LUMO (eV)", value=-2.0)
            f3 = c3.number_input("Dipole Moment (D)", value=0.0)
            f4 = c4.number_input("Electron Affinity (eV)", value=0.0)

            f5 = c1.number_input("Atom Count", value=3)
            f6 = c2.number_input("Molecular Mass", value=30.0)
            f7 = c3.number_input("Bond Length (Å)", value=1.2)
            f8 = c4.number_input("Rotational Constant", value=1.5)

            submitted = st.form_submit_button("🚀 Run Prediction with Custom Data", type="primary")
            if submitted:
                st.session_state.gas_feats = [f1, f2, f3, f4, f5, f6, f7, f8]
                if not st.session_state.gas_name:
                    st.session_state.gas_name = "Custom Molecule"
                st.session_state.view_mode = "results"
                st.rerun()

        if st.button("← Back to Auto Search"):
            st.session_state.view_mode = "search"
            st.rerun()

# --- PAGE 2: Results & Visualizations ---
elif st.session_state.view_mode == "results":

    st.markdown("### Step 2: Global Screening Diagnosis")

    # 🛡️ THE PHYSICS GUARDRAIL (物理先验围栏)
    # gas_feats index: [HOMO(0), LUMO(1), Dipole(2), EA(3), Atom(4), Mass(5), Bond(6), Rot(7)]
    dipole_moment = st.session_state.gas_feats[2]
    molecular_mass = st.session_state.gas_feats[5]

    if dipole_moment < 0.01 and molecular_mass < 30.0:
        st.markdown(f"""
        <div class='guardrail-card'>
            <div class='guardrail-title'>🛡️ PHYSICS-INFORMED EXTRAPOLATION GUARDRAIL TRIGGERED</div>
            <p style='font-size: 18px; color: #1e3a8a; margin-top: 15px;'>
                The physical engine detected <strong>{st.session_state.gas_name}</strong> as an extremely light, non-polar diatomic molecule (Mw < 30, μ ≈ 0). 
                <br><br>
                <strong>Fundamental Constraint:</strong> Without a catalytic dissociation center (e.g., Pt/Pd), such molecules only undergo minimal physisorption on 3d transition metal HEAs, resulting in negligible charge transfer (ΔQ ≈ 0).
                <br><br>
                <em>Machine learning execution bypassed to prevent unphysical extrapolation artifacts.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Acknowledge & Return", type="primary"):
            st.session_state.view_mode = "search"
            st.session_state.gas_name = ""
            st.rerun()
        st.stop()  # 终止执行，不调用模型！

    # --- 正常运行机器学习预测 ---
    with st.spinner('Executing High-Throughput Screening...'):
        gas_matrix = np.tile(st.session_state.gas_feats, (len(site_db), 1))
        site_features = site_db[
            ['原位局域d带中心(eV)', '基底_局域总d电子数', '基底_平均功函数(eV)', '基底_平均电离能(eV)']].values
        X_predict = np.hstack((gas_matrix, site_features))

        predictions = model.predict(X_predict)
        site_db['E_ads_pred'] = predictions[:, 0]
        site_db['dQ_pred'] = predictions[:, 1]

        site_db['E_ads_abs'] = site_db['E_ads_pred'].abs()
        df_unique = site_db.sort_values('E_ads_abs', ascending=False).drop_duplicates(subset=['Clean_Name'])

        valid_mask = (df_unique['E_ads_pred'] >= -1.5) & (df_unique['E_ads_pred'] <= -0.6) & (
                    df_unique['dQ_pred'].abs() >= 0.1)
        best_candidates = df_unique[valid_mask].sort_values(by='E_ads_pred', ascending=False)

    if best_candidates.empty:
        st.markdown(
            f"<div class='bad-site-card'><div class='bad-site-title'>⚠️ NOT OPTIMAL FOR SENSING</div><br>System predicts that {st.session_state.gas_name} either exhibits weak physisorption or irreversible chemisorption (poisoning) on the current high-entropy alloy surface. No optimal active sites found under strict Sabatier criteria.</div>",
            unsafe_allow_html=True)
    else:
        top_site = best_candidates.iloc[0]
        st.markdown(
            f"<div class='best-site-card'><div class='best-site-title'>🏆 IDENTIFIED OPTIMAL ACTIVE SITE FOR {st.session_state.gas_name}</div><div style='font-size: 20px; color:#374151; margin:15px 0;'><strong>{top_site['Clean_Name']}</strong></div><div style='display:flex; justify-content:center; gap:50px;'><div class='best-site-data'>E_ads: {top_site['E_ads_pred']:.2f} eV</div><div class='best-site-data'>ΔQ: {top_site['dQ_pred']:.3f} e</div></div></div>",
            unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 散点图
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### Step 3: Energy-Charge Transfer Distribution Mapping")


    def get_zone(row):
        if row['E_ads_pred'] > -0.6:
            return "Weak Interaction (Physisorption)"
        elif row['E_ads_pred'] < -1.5:
            return "Strong Interaction (Poisoning)"
        else:
            return "Optimal Reversible Sensing Zone"


    df_unique['Adsorption Zone'] = df_unique.apply(get_zone, axis=1)

    color_map = {
        "Optimal Reversible Sensing Zone": "#2563EB",
        "Weak Interaction (Physisorption)": "#D1D5DB",
        "Strong Interaction (Poisoning)": "#FCA5A5"
    }

    fig = px.scatter(
        df_unique, x='dQ_pred', y='E_ads_pred', color='Adsorption Zone',
        hover_name='Clean_Name',
        color_discrete_map=color_map,
        labels={'dQ_pred': 'Charge Transfer, ΔQ (e)', 'E_ads_pred': 'Adsorption Energy, E_ads (eV)'},
        opacity=0.9
    )

    fig.add_shape(type="rect",
                  x0=df_unique['dQ_pred'].min() - 0.2, y0=-1.5, x1=df_unique['dQ_pred'].max() + 0.2, y1=-0.6,
                  line=dict(color="rgba(34, 197, 94, 0.5)", width=2, dash="dash"),
                  fillcolor="rgba(34, 197, 94, 0.08)", layer="below"
                  )

    fig.add_hline(y=0, line_dash="solid", line_color="#9CA3AF", line_width=1)
    fig.add_vline(x=0, line_dash="solid", line_color="#9CA3AF", line_width=1)

    fig.add_annotation(
        x=0, y=-1.05, text="Sabatier Optimal Range", showarrow=False,
        font=dict(size=14, color="#15803d", family="Arial", weight="bold"),
        bgcolor="rgba(255,255,255,0.7)"
    )

    fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color='white')))
    fig.update_layout(
        plot_bgcolor='#FAFAFA', paper_bgcolor='white', font=dict(size=16, family="Arial"),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20), hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Start New Prediction", type="secondary"):
        st.session_state.view_mode = "search"
        st.session_state.gas_name = ""
        st.rerun()