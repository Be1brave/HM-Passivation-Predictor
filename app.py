import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 已经为您取消了注释

# --- 1. 页面基本设置 ---
st.set_page_config(page_title="HM Passivation Prediction", layout="wide")
st.title("重金属钝化率预测系统 (HM Passivation Rate Prediction)")
st.markdown("基于机器学习模型预测不同材料和土壤条件下的重金属钝化率。")
st.markdown("---")

# --- 2. 加载机器学习模型 ---
@st.cache_resource
def load_model():
    try:
        # 这里已经写好了您刚才生成的真实模型路径
        return joblib.load("D:/Python/GUI/rf_model.pkl")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()

# --- 3. 侧边栏：获取用户输入特征 ---
st.sidebar.header("输入预测参数 (Input Parameters)")

def user_input_features():
    st.sidebar.subheader("1. 材料特性 (Material)")
    material_type = st.sidebar.number_input("Material Type (材料类型编号)", value=5, step=1)
    material_ph = st.sidebar.number_input("Material pH (材料pH)", value=9.83, format="%.2f")
    material_ssa = st.sidebar.number_input("Material SSA (材料比表面积)", value=83.05, format="%.2f")
    app_dosage = st.sidebar.number_input("Application Dosage (施用量)", value=2.00, format="%.3f")

    st.sidebar.subheader("2. 土壤与重金属属性 (Soil & HM)")
    soil_type = st.sidebar.number_input("Soil Type (土壤类型编号)", value=22, step=1)
    soil_ph = st.sidebar.number_input("Soil pH (土壤pH)", value=6.13, format="%.2f")
    soil_cec = st.sidebar.number_input("Soil CEC (土壤CEC)", value=19.89, format="%.2f")
    om = st.sidebar.number_input("OM (有机质)", value=16.37, format="%.2f")
    hm_type = st.sidebar.number_input("HM type (重金属类型编号)", value=2, step=1)
    orig_hm_aval = st.sidebar.number_input("Original HM Aval. (原始重金属有效态)", value=1.96, format="%.2f")

    st.sidebar.subheader("3. 实验条件 (Condition)")
    duration = st.sidebar.number_input("Duration (实验时长/天)", value=60.0, step=1.0)

    # 整理数据格式
    data = {
        'Material Type': material_type,
        'Material pH': material_ph,
        'Soil CEC': soil_cec,
        'Material SSA': material_ssa,
        'Soil pH': soil_ph,
        'OM': om,
        'Duration': duration,
        'Soil Type': soil_type,
        'HM type': hm_type,
        'Original HM Aval.': orig_hm_aval,
        'Application Dosage': app_dosage
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. 主界面展示与预测 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("当前输入的参数摘要")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), height=450)

with col2:
    st.subheader("预测结果 (Prediction)")
    
    if st.button("🚀 开始预测 (Predict HM Pass. Rate)"):
        if model is not None:
            # --- 真实预测核心逻辑 ---
            prediction = model.predict(input_df)
            result = prediction[0]
            
            # 限制结果在 0~100 之间，符合物理意义
            result = max(0.0, min(100.0, result)) 
            
            st.success(f"基于您的模型，预测的重金属钝化率 (HM Pass. Rate) 为: **{result:.2f} %**")
            st.progress(int(result))
        else:
            st.error("模型未加载，请确保 D:/Python/GUI/rf_model.pkl 文件存在。")