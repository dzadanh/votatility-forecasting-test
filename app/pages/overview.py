"""Historical offline price and realized volatility view."""
import streamlit as st
import pandas as pd
from pathlib import Path
from src.config import ROOT
from src.data.preprocessing import clean_data, preprocess_data
from src.features.builder import build_features
from components.charts import line_chart


def render(ticker, csv_path=None):
    st.header("Tổng quan dữ liệu")
    if csv_path:
        path = Path(csv_path)
        if not path.is_absolute():
            path = ROOT / path
        source = clean_data(pd.read_csv(path))
    else:
        source = preprocess_data(ticker, save=False)
    frame = build_features(source)
    st.write(f"{ticker}: {frame.date.min().date()} đến {frame.date.max().date()} · {len(frame)} phiên")
    st.plotly_chart(line_chart(frame, "date", "close", "Giá đóng cửa"), width="stretch")
    st.plotly_chart(line_chart(frame.dropna(subset=["rolling_volatility_20"]), "date",
                              "rolling_volatility_20", "Độ biến động 20 phiên (điểm phần trăm)"), width="stretch")
    st.caption("Dữ liệu CSV cục bộ. Biến động quá khứ không phải dự báo.")
