# Offline Volatility Forecasting Prototype

Bản mẫu nghiên cứu dự báo độ biến động ACB trong 5 phiên giao dịch tiếp theo bằng GARCH(1,1), XGBoost và LSTM. Chạy hoàn toàn từ `data/raw/acb.csv` sau khi cài thư viện. Không tải dữ liệu trong huấn luyện, đánh giá, test hoặc dashboard; không phải hệ thống giao dịch hay lời khuyên đầu tư.

## Cài đặt (Python 3.10+)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy bản mẫu

Tại thư mục gốc repo:

```powershell
python -m scripts.train --ticker ACB --model all --fast
python -m scripts.evaluate --ticker ACB
python -m scripts.predict --ticker ACB --model xgboost
python -m pytest -q
python -m streamlit run app/app.py
```

`--fast` giới hạn XGBoost ở hai cấu hình thử và LSTM ở tối đa ba epoch. Bỏ cờ để dùng tám cấu hình và số epoch trong `config.yaml`. Có thể thay `--model all` bằng `garch`, `xgboost`, hoặc `lstm`; để so sánh đủ ba mô hình trên cùng ngày test, chạy `all`. Test `test_offline_acb_end_to_end` tự chạy cả ba mô hình ở chế độ nhanh.

`--csv đường/dẫn/file.csv` cho phép chọn CSV OHLCV cục bộ khác; hoặc đổi `paths.raw` trong config để trỏ tới thư mục CSV theo ticker. Mặc định vẫn dùng `data/raw/acb.csv` qua preprocessing hiện có. Dữ liệu mới phải được kiểm định riêng trước khi diễn giải kết quả.

## Hợp đồng dữ liệu và mô hình

`src/pipeline/dataset.py` luôn gọi preprocessing, feature builder và target builder trên CSV thô. Dữ liệu được sắp theo ngày, loại dòng feature không hợp lệ khi huấn luyện, nhưng giữ các ngày gần nhất có feature và chưa có target cho suy luận. Chia train/validation/test theo thời gian 70/15/15; bỏ ít nhất `horizon` origin trước ranh giới để target tương lai của nhóm trước không chạm nhóm sau. LSTM có thể dùng feature quá khứ làm ngữ cảnh; scaler chỉ fit trên train. XGBoost chỉ nhận danh sách 14 feature trong `config.yaml` theo đúng thứ tự đã lưu; các cột khác trong frame gốc được bộ chọn feature loại bỏ có chủ đích.

`return_1d` là log return nhân 100, đơn vị **điểm phần trăm**. `target_volatility_5d` là độ lệch chuẩn **mẫu** của năm return ở các phiên `t+1…t+5`, cũng theo điểm phần trăm; không annualize. Target không được làm đầu vào. `data/processed/acb_features.csv` là bản cũ có return xấp xỉ `-0.026` tại phiên 2018-12-17, trong khi builder hiện nay cho xấp xỉ `-2.616`; vì vậy pipeline không dùng bản processed này.

Mỗi model nhận frame lịch sử và danh sách origin date, xuất `date`, `model`, `predicted_volatility` cùng đơn vị không âm. XGBoost và LSTM còn lưu `raw_prediction`; giá trị hồi quy âm được chặn ở 0 cho cảnh báo. SHAP giải thích **đầu ra thô** XGBoost, trước khi chặn. GARCH khớp tham số trên train rồi lọc return quan sát đến từng origin với tham số cố định. Căn bậc hai trung bình phương sai dự báo nhiều bước chỉ là **xấp xỉ** target độ lệch chuẩn mẫu thực tế, không phải cùng một đại lượng theo toán học.

Ngưỡng NORMAL/WARNING/HIGH lấy quantile 0.70/0.90 của **target train thực tế**. Chúng là nhãn theo độ biến động, không phải xác suất lỗ hay tín hiệu mua bán. Model được chọn bằng validation theo thời gian; test chỉ dùng để báo cáo cuối. GARCH và LSTM không có SHAP trong bản mẫu.

## Artifact ổn định

`artifacts/<TICKER>/` (bị Git ignore) gồm:

| File | Nội dung |
|---|---|
| `metadata.json` | ticker, target, horizon, danh sách feature, đơn vị, seed, cấu hình model, khoảng ngày và số mẫu mỗi split, số ngày test khớp |
| `warning.json` | quantile và hai ngưỡng số đã fit |
| `<model>.joblib` | model đã fit; chỉ tải artifact đáng tin cậy |
| `metrics.json` | MAE, RMSE, R², số mẫu; precision, recall, F1, support và confusion matrix cho HIGH |
| `predictions.csv` | `date,ticker,partition,model,actual_volatility,predicted_volatility,raw_prediction,risk_level,actual_risk_level`; partition là `test` hoặc `latest`, actual của latest trống |
| `shap.json` | global mean absolute SHAP và giá trị local, feature input, base value, raw output, clipped forecast |

Dashboard chỉ đọc các output đã lưu và CSV cục bộ; không tự huấn luyện lại khi rerender. CLI `predict` tái tạo feature từ CSV thô và từ chối model lưu có schema/target/horizon không khớp.

## Phạm vi tiếp theo

`scripts/download_data.py` giữ nguyên hành vi provider cũ và không được gọi bởi pipeline. Việc lựa chọn, kiểm định và tích hợp nguồn dữ liệu khác được để lại cho giai đoạn sau. Repo cũ đang track nhiều file `.venv`; `.gitignore` chỉ chặn file tạo mới, không tự xóa hoặc untrack phần đang track. Dọn lịch sử Git là công việc riêng.
