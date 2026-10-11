"""Latest saved forecasts and their training-fitted risk levels."""
import pandas as pd
import streamlit as st
from components.charts import forecast_chart


def render(predictions, warning, metadata):
    st.header("Dự báo & cảnh báo")
    models = sorted(predictions.model.unique())
    model = st.selectbox("Mô hình", models)
    latest = predictions[(predictions.model == model) & (predictions.partition == "latest")]
    if latest.empty:
        st.info("Chưa có dự báo ngày mới nhất cho mô hình này.")
        return
    row = latest.iloc[-1]
    st.metric(f"Dự báo độ biến động {metadata['horizon']} phiên", f"{row.predicted_volatility:.3f} điểm %")
    st.write(f"Mức cảnh báo: **{row.risk_level}** · Thông tin đến hết phiên **{pd.Timestamp(row.date).date()}**")
    st.caption(f"Ngưỡng train: WARNING ≥ {warning['warning_threshold']:.3f}; HIGH ≥ {warning['high_threshold']:.3f} điểm %.")
    st.caption(f"Dự báo mới nhất chưa có thực tế {metadata['horizon']} phiên kế tiếp để đối chiếu.")
    history = predictions[(predictions.model == model) & (predictions.partition == "test")]
    if not history.empty:
        st.plotly_chart(forecast_chart(history), width="stretch")
    st.caption("Nhãn cảnh báo là phân loại độ biến động, không phải xác suất thua lỗ hay tín hiệu giao dịch.")
