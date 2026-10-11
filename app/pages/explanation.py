"""Saved XGBoost SHAP attributions."""
import pandas as pd
import plotly.express as px
import streamlit as st


def render(shap_data):
    st.header("Giải thích XGBoost")
    if not shap_data:
        st.info("Chưa có SHAP. Hãy huấn luyện XGBoost bằng CLI.")
        return
    importance = pd.DataFrame(shap_data["global_importance"])
    st.plotly_chart(px.bar(importance, x="mean_abs_shap", y="feature", orientation="h",
                           title="Độ quan trọng SHAP trung bình"), width="stretch")
    dates = [row["date"] for row in shap_data["local"]]
    selected = st.selectbox("Ngày giải thích", dates, index=len(dates)-1)
    local = next(row for row in shap_data["local"] if row["date"] == selected)
    st.write(f"Giá trị nền: {local['base_value']:.3f}; đầu ra thô: {local['raw_model_output']:.3f}; dự báo sau chặn âm: {local['clipped_forecast']:.3f}")
    local_frame = pd.DataFrame({"feature": list(local["shap_values"]), "SHAP": list(local["shap_values"].values()),
                                "input": list(local["features"].values())})
    st.dataframe(local_frame, hide_index=True)
    st.caption("SHAP giải thích đầu ra thô của XGBoost; không chứng minh quan hệ nhân quả. GARCH và LSTM không có SHAP trong bản mẫu này.")
