"""Read-only Streamlit frontend for saved offline experiments."""
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config
from pages import overview, forecast, comparison, explanation


@st.cache_data
def read_outputs(path: str):
    folder = Path(path)
    predictions = pd.read_csv(folder / "train_predictions.csv", parse_dates=["date"])
    files = {name: json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
             for name in ("metrics", "metadata", "warning")}
    shap_path = folder / "shap.json"
    files["shap"] = (json.loads(shap_path.read_text(encoding="utf-8"))
                     if shap_path.exists() and "xgboost" in files["metadata"].get("model_settings", {}) else None)
    return predictions, files


def main():
    st.set_page_config(page_title="Volatility Forecasting", layout="wide")
    st.title("Giám sát độ biến động · bản mẫu nghiên cứu")
    st.caption("Chạy với dữ liệu CSV cục bộ; không cập nhật thời gian thực. Không phải lời khuyên đầu tư.")
    config = load_config()
    base = ROOT / config["paths"]["artifacts"]
    tickers = sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "metadata.json").exists()) if base.exists() else []
    ticker = st.sidebar.selectbox("Mã chứng khoán", tickers or [config["data"]["ticker"]])
    page = st.sidebar.radio("Trang", ["Tổng quan", "Dự báo & Cảnh báo", "So sánh", "Giải thích"])
    if page == "Tổng quan":
        try:
            metadata_path = base / ticker / "metadata.json"
            source = json.loads(metadata_path.read_text(encoding="utf-8")).get("source_csv") if metadata_path.exists() else None
            default_source = str(Path(config["paths"]["raw"]) / f"{ticker.lower()}.csv")
            overview.render(ticker, source if source != default_source else None)
        except FileNotFoundError:
            st.info(f"Chưa có data/raw/{ticker.lower()}.csv trong máy.")
        return
    folder = base / ticker
    try:
        predictions, files = read_outputs(str(folder))
    except FileNotFoundError:
        st.info(f"Chưa có kết quả cho {ticker}. Chạy `python -m scripts.train --ticker {ticker} --model all` trước.")
        return
    if page == "Dự báo & Cảnh báo":
        forecast.render(predictions, files["warning"], files["metadata"])
    elif page == "So sánh":
        comparison.render(predictions, files["metrics"], files["metadata"])
    else:
        explanation.render(files["shap"])


if __name__ == "__main__":
    main()
