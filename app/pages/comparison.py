"""Matched test-date model comparison."""
import pandas as pd
import streamlit as st
from components.charts import forecast_chart


def render(predictions, metrics, metadata):
    st.header("So sánh mô hình")
    rows = []
    for name, m in metrics.items():
        alert = m["high_warning"]
        rows.append({"mô hình": name, "MAE": m["mae"], "RMSE": m["rmse"],
                     "Precision HIGH": alert["precision"], "Recall HIGH": alert["recall"],
                     "F1 HIGH": alert["f1"], "mẫu": m["samples"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.write(f"Ngày test khớp giữa mô hình: {metadata['matched_test_count']}; ngày bị loại khi căn chỉnh: {metadata['dropped_test_origins']}")
    history = predictions[predictions.partition == "test"]
    if not history.empty:
        st.plotly_chart(forecast_chart(history), width="stretch")
    st.caption(f"GARCH dùng phương sai có điều kiện làm xấp xỉ; target thực tế là độ lệch chuẩn mẫu của {metadata['horizon']} return tương lai. Kết quả chỉ là kiểm thử lịch sử trên dữ liệu cục bộ.")
