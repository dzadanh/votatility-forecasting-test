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
    high_score_threshold = warning.get("model_high_probability_thresholds", {}).get(model)
    if high_score_threshold is not None and pd.notna(row.get("high_probability")):
        st.caption(f"Ngưỡng WARNING ≥ {warning['warning_threshold']:.3f} điểm %; "
                   f"HIGH của {model} khi điểm HIGH {row.high_probability:.3f} ≥ "
                   f"{high_score_threshold:.3f}. Điểm này chưa được hiệu chuẩn thành xác suất.")
    else:
        high_threshold = warning.get("model_high_thresholds", {}).get(model, warning["high_threshold"])
        st.caption(f"Ngưỡng WARNING ≥ {warning['warning_threshold']:.3f}; HIGH của {model} ≥ {high_threshold:.3f} điểm %.")
    if "high_alert_event" in row and row.high_alert_event:
        st.info("Đây là lần chuyển mới sang mức HIGH trong chuỗi dự báo đã lưu.")
    elif row.risk_level == "HIGH":
        st.caption("Mức HIGH đang tiếp diễn; high_alert_event giúp tránh phát lại cùng một cảnh báo liên tiếp.")
    st.caption(f"Dự báo mới nhất chưa có thực tế {metadata['horizon']} phiên kế tiếp để đối chiếu.")
    history = predictions[(predictions.model == model) & (predictions.partition == "test")]
    if not history.empty:
        st.plotly_chart(forecast_chart(history), width="stretch")
    st.caption("Nhãn cảnh báo là phân loại độ biến động, không phải xác suất thua lỗ hay tín hiệu giao dịch.")
