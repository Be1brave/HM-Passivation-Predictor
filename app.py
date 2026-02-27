import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. 全局页面格局设置 (铺满全屏) ---
st.set_page_config(page_title="HM Passivation Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 2. 加载模型 ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model.pkl")
    except Exception:
        return None

model = load_model()

# ==========================================
# 区域 1：侧边栏 (Sidebar) - 纯粹的控制台格局
# ==========================================
with st.sidebar:
    st.title("⚙️ Parameter Setup")
    st.markdown("Adjust the material and soil properties below.")
    
    st.subheader("Material Properties")
    material_type = st.number_input("Material Type", value=5, step=1)
    material_ph = st.number_input("Material pH", value=9.83)
    material_ssa = st.number_input("Material SSA", value=83.05)
    app_dosage = st.number_input("App. Dosage", value=2.00)
    
    st.subheader("Soil Conditions")
    soil_type = st.number_input("Soil Type", value=22, step=1)
    soil_ph = st.number_input("Soil pH", value=6.13)
    soil_cec = st.number_input("Soil CEC", value=19.89)
    om = st.number_input("OM", value=16.37)
    hm_type = st.number_input("HM type", value=2, step=1)
    orig_hm_aval = st.number_input("Original HM Aval.", value=1.96)
    duration = st.number_input("Duration (days)", value=60.0)

    input_data = pd.DataFrame({
        'Material Type': [material_type], 'Material pH': [material_ph],
        'Soil CEC': [soil_cec], 'Material SSA': [material_ssa],
        'Soil pH': [soil_ph], 'OM': [om], 'Duration': [duration],
        'Soil Type': [soil_type], 'HM type': [hm_type],
        'Original HM Aval.': [orig_hm_aval], 'Application Dosage': [app_dosage]
    })

# ==========================================
# 区域 2：主界面顶部 - 核心 KPI 卡片格局
# ==========================================
st.title("🛡️ Heavy Metal Passivation Dashboard")
st.markdown("Real-time prediction driven by XGBoost Algorithm")
st.markdown("<br>", unsafe_allow_html=True)

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.info("🧠 Core Algorithm")
    st.metric(label="Model", value="XGBoost Regressor", delta="Optimized")
    
with kpi2:
    st.info("🎯 Accuracy Score")
    st.metric(label="Test Set R²", value="0.6860", delta="High")

with kpi3:
    st.info("📊 Feature Space")
    st.metric(label="Input Variables", value="11 Features", delta="Active", delta_color="off")

st.markdown("---")

# ==========================================
# 区域 3：主界面中下部 - 左右对比格局
# ==========================================
left_col, right_col = st.columns([4, 6])

if model is not None:
    pred_val = model.predict(input_data)[0]
    pred_val = max(0.0, min(100.0, pred_val))
else:
    pred_val = 0.0

with left_col:
    st.subheader("📋 Current Input Vector")
    st.markdown("The specific values currently fed into the model:")
    display_df = input_data.T.rename(columns={0: "Parameter Value"})
    
    # 修复了这里的警告：将 use_container_width=True 替换为了 width='stretch'
    st.dataframe(display_df, width='stretch', height=380)

with right_col:
    st.subheader("🚀 Real-time Prediction")
    
    if model is not None:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_val,
            number = {'suffix': "%", 'font': {'size': 50, 'color': "#E0E0E0"}},
            title = {'text': "Passivation Rate", 'font': {'size': 24, 'color': "#2EA043"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#2EA043"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#30363D",
                'steps': [
                    {'range': [0, 30], 'color': "#30363D"},
                    {'range': [30, 70], 'color': "#4C566A"},
                    {'range': [70, 100], 'color': "#81A1C1"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#E0E0E0"}, height=350, margin=dict(l=20, r=20, t=50, b=20))
        
        # 修复了这里的警告
        st.plotly_chart(fig, use_container_width=True) # 注意：Plotly 目前依然推荐这个用法，如果依然报错，后续我们可以删掉这个参数
    else:

        st.error("Model Error: rf_model.pkl not found.")
