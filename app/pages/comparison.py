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
                     "F1 HIGH": alert["f1"], "Số cảnh báo HIGH": alert.get("alert_count"),
                     "Tỷ lệ cảnh báo HIGH (%)": alert.get("alert_rate_percent"),
                     "mẫu": m["samples"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    class_rows = []
    for name, m in metrics.items():
        for label, score in m.get("risk_levels", {}).get("per_class", {}).items():
            class_rows.append({"mô hình": name, "mức cảnh báo": label,
                               "Precision": score["precision"], "Recall": score["recall"],
                               "F1": score["f1"], "số mẫu thực tế": score["support"]})
    if class_rows:
        st.subheader("Đánh giá theo từng mức cảnh báo")
        st.dataframe(pd.DataFrame(class_rows), hide_index=True)
    st.write(f"Ngày test khớp giữa mô hình: {metadata['matched_test_count']}; ngày bị loại khi căn chỉnh: {metadata['dropped_test_origins']}")
    policy = metadata.get("alert_policy")
    if policy:
        st.caption(f"Ngưỡng HIGH thực tế: {policy['actual_high_threshold']:.3f}. Ngưỡng dự báo được chọn trên validation; giới hạn tần suất {policy['max_alert_rate']:.0%}, precision tối thiểu {policy['min_precision']:.0%}.")
        for name, override in policy.get("model_overrides", {}).items():
            st.caption(f"{name}: giới hạn cảnh báo validation {override.get('max_alert_rate', policy['max_alert_rate']):.0%}, precision tối thiểu {override.get('min_precision', policy['min_precision']):.0%}.")
    st.caption("Các ngày HIGH liên tiếp được gộp thành một high_alert_event ở lần đầu chuyển mức; cảnh báo thực tế cần lớp gửi thông báo đọc cờ này.")
    history = predictions[predictions.partition == "test"]
    if not history.empty:
        st.plotly_chart(forecast_chart(history), width="stretch")
    st.caption(f"GARCH dùng phương sai có điều kiện làm xấp xỉ; target thực tế là độ lệch chuẩn mẫu của {metadata['horizon']} return tương lai. Kết quả chỉ là kiểm thử lịch sử trên dữ liệu cục bộ.")
