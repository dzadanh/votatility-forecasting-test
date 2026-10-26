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

`--fast` giới hạn XGBoost ở hai cấu hình thử và LSTM ở tối đa ba epoch; chế độ này chỉ để kiểm tra luồng chạy, không dùng đánh giá chất lượng cảnh báo. Bỏ cờ để dùng lưới XGBoost trong `config.yaml` (15 cấu hình, 5 fold; 75 lượt fit cho bước chọn tham số) và số epoch LSTM đã cấu hình. Lưới có cả cấu hình 5 cây cũ và các cấu hình 50–300 cây. XGBoost dùng GridSearchCV trên các fold thời gian; số luồng của estimator được cố định và việc song song hóa chỉ diễn ra ở tầng tìm kiếm. Ngưỡng HIGH phục vụ chọn tham số được tính từ train của fold đầu tiên, trước mọi validation. Bộ chọn ưu tiên average precision của HIGH trong các cấu hình có RMSE validation không quá 5% so với RMSE tốt nhất; metadata lưu cả hai chỉ số và kết quả từng cấu hình. Có thể thay `--model all` bằng `garch`, `xgboost`, hoặc `lstm`; để so sánh đủ ba mô hình trên cùng ngày test, chạy `all`. Test `test_offline_acb_end_to_end` tự chạy cả ba mô hình ở chế độ nhanh.

`--csv đường/dẫn/file.csv` cho phép chọn CSV OHLCV cục bộ khác; hoặc đổi `paths.raw` trong config để trỏ tới thư mục CSV theo ticker. Mặc định vẫn dùng `data/raw/acb.csv` qua preprocessing hiện có. Dữ liệu mới phải được kiểm định riêng trước khi diễn giải kết quả.

Để kiểm tra XGBoost mới mà vẫn giữ artifact chính, có thể dùng `python -m scripts.train --ticker ACB --model xgboost --artifact-dir artifacts/xgboost_trial` rồi `python -m scripts.evaluate --ticker ACB --artifact-dir artifacts/xgboost_trial`. Thay ticker cho FPT, HPG, VIC; mỗi lần chạy chỉ lưu kết quả mô hình được chọn vào thư mục thử nghiệm.

## Hợp đồng dữ liệu và mô hình

`src/pipeline/dataset.py` luôn gọi preprocessing, feature builder và target builder trên CSV thô. Dữ liệu được sắp theo ngày, loại dòng feature không hợp lệ khi huấn luyện, nhưng giữ các ngày gần nhất có feature và chưa có target cho suy luận. Chia dữ liệu theo thời gian: test cuối chiếm 15% số origin có target hợp lệ; phần trước test dùng `TimeSeriesSplit(n_splits=5, test_size=125, gap=5, max_train_size=None)`. Gap 5 origin được giữ trước mỗi fold và trước test để target tương lai không chạm sang đoạn kế tiếp. Các fold validation mở rộng tập train theo thời gian; tuning và hiệu chỉnh cảnh báo dùng dự báo out-of-fold. LSTM có thể dùng feature quá khứ làm ngữ cảnh; scaler chỉ fit trên train. XGBoost chỉ nhận danh sách 14 feature trong `config.yaml` theo đúng thứ tự đã lưu; các cột khác trong frame gốc được bộ chọn feature loại bỏ có chủ đích.

`return_1d` là log return nhân 100, đơn vị **điểm phần trăm**. `target_volatility_5d` là độ lệch chuẩn **mẫu** của năm return ở các phiên `t+1…t+5`, cũng theo điểm phần trăm; không annualize. Target không được làm đầu vào. `data/processed/acb_features.csv` là bản cũ có return xấp xỉ `-0.026` tại phiên 2018-12-17, trong khi builder hiện nay cho xấp xỉ `-2.616`; vì vậy pipeline không dùng bản processed này.

Mỗi model nhận frame lịch sử và danh sách origin date, xuất `date`, `model`, `predicted_volatility` cùng đơn vị không âm. XGBoost và LSTM còn lưu `raw_prediction`; giá trị hồi quy âm được chặn ở 0 cho cảnh báo. SHAP giải thích **đầu ra thô** XGBoost, trước khi chặn. GARCH khớp tham số trên train rồi lọc return quan sát đến từng origin với tham số cố định. Căn bậc hai trung bình phương sai dự báo nhiều bước chỉ là **xấp xỉ** target độ lệch chuẩn mẫu thực tế, không phải cùng một đại lượng theo toán học.

Ngưỡng NORMAL/WARNING và nhãn HIGH thực tế lấy quantile **0.50/0.70** của target train thực tế. Ngưỡng dự báo HIGH được hiệu chỉnh riêng từng model trên validation, ưu tiên recall trong giới hạn tỷ lệ cảnh báo và precision tối thiểu. Giới hạn mặc định là 15% số ngày validation; `alert_policy.model_overrides.xgboost` cho phép XGBoost dùng tối đa **30%** để ưu tiên phát hiện HIGH. Đây là giới hạn trên validation, không bảo đảm cùng tỷ lệ trên test; tăng cảnh báo có thể làm tăng báo động sai. Nếu không có ngưỡng thỏa điều kiện thì dùng lại quantile HIGH train. `high_alert_event` đánh dấu lần đầu bước vào HIGH trong chuỗi dự báo đã lưu để lớp thông báo bên ngoài có thể gộp các ngày HIGH liên tiếp. Trong pipeline, test chỉ dùng để báo cáo, không dùng chọn tham số hay ngưỡng. GARCH và LSTM không có SHAP trong bản mẫu.

Các giai đoạn test hiện có đã được xem trong quá trình phát triển, kể cả khi so ngân sách cảnh báo XGBoost 15% và 30%. Vì thế kết quả trên những giai đoạn này là **thử nghiệm lịch sử**, chưa phải bằng chứng trên holdout chưa từng xem. Cần một giai đoạn dữ liệu mới để xác nhận mức cải thiện và số báo động sai.

## Artifact ổn định

`artifacts/<TICKER>/` (bị Git ignore) gồm:

| File | Nội dung |
|---|---|
| `metadata.json` | ticker, target, horizon, danh sách feature, đơn vị, seed, cấu hình model, khoảng ngày và số mẫu mỗi split, số ngày test khớp, chính sách và kết quả hiệu chỉnh cảnh báo trên validation |
| `warning.json` | ngưỡng WARNING/HIGH train và ngưỡng HIGH vận hành theo từng model |
| `<model>.joblib` | model đã fit; chỉ tải artifact đáng tin cậy |
| `metrics.json` | MAE, RMSE, R², số mẫu; precision, recall, F1 và support cho NORMAL/WARNING/HIGH, confusion matrix 3×3 (hàng là mức thực tế, cột là mức dự báo), cùng metric cảnh báo HIGH |
| `train_predictions.csv` | dự báo test/latest được tạo trong pipeline train, mức rủi ro và `high_alert_event` đánh dấu lần chuyển mới sang HIGH; actual của latest trống |
| `shap.json` | global mean absolute SHAP và giá trị local, feature input, base value, raw output, clipped forecast |

Dashboard chỉ đọc các output đã lưu và CSV cục bộ; không tự huấn luyện lại khi rerender. CLI `predict` tái tạo feature từ CSV thô và từ chối model lưu có schema/target/horizon không khớp.

## Phạm vi tiếp theo

`scripts/download_data.py` giữ nguyên hành vi provider cũ và không được gọi bởi pipeline. Việc lựa chọn, kiểm định và tích hợp nguồn dữ liệu khác được để lại cho giai đoạn sau. Repo cũ đang track nhiều file `.venv`; `.gitignore` chỉ chặn file tạo mới, không tự xóa hoặc untrack phần đang track. Dọn lịch sử Git là công việc riêng.
