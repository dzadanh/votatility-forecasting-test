# Tổng quan dự án Volatility Forecasting

Tài liệu này giải thích cấu trúc và luồng xử lý của dự án theo mã nguồn hiện có. Đây là prototype nghiên cứu dự báo độ biến động cổ phiếu; không phải hệ thống giao dịch, dự báo giá hay công cụ tư vấn đầu tư.

## 1. Dự án làm gì?

Dự án nhận dữ liệu giá và khối lượng OHLCV theo ngày (Open, High, Low, Close, Volume), tạo các đặc trưng từ lịch sử, rồi dự báo độ biến động của lợi suất trong `horizon` phiên tiếp theo. Cấu hình mặc định dùng mã ACB, horizon 5 phiên và so sánh ba mô hình:

- **GARCH(1,1):** mô hình chuỗi thời gian kinh tế lượng, mô tả phương sai có điều kiện thay đổi theo thời gian.
- **XGBoost:** mô hình cây tăng cường, học quan hệ giữa các đặc trưng hiện tại/quá khứ và nhãn biến động tương lai.
- **LSTM:** mạng hồi tiếp đọc một cửa sổ đặc trưng theo thứ tự thời gian.

Pipeline huấn luyện và dự báo dùng CSV cục bộ. Dashboard Streamlit đọc kết quả đã lưu; nó không huấn luyện lại mô hình mỗi lần mở trang.

## 2. Cây thư mục

```text
votatility-forecasting-test/
├── app/                 # Dashboard Streamlit
├── data/                # Dữ liệu raw và processed (cần có CSV đầu vào)
├── notebooks/           # Notebook khám phá và thử nghiệm
├── scripts/             # Lệnh CLI cho train, predict, evaluate, tải dữ liệu
├── src/                 # Mã nguồn xử lý dữ liệu, mô hình và pipeline
├── tests/               # Kiểm tra hợp đồng dữ liệu và hành vi các module
├── artifacts/           # Đầu ra huấn luyện; bị Git ignore, tạo khi chạy
├── config.yaml          # Cấu hình dùng chung
├── requirements.txt     # Thư viện Python
├── pytest.ini           # Cấu hình pytest
├── .gitignore           # Các tệp/thư mục không đưa vào Git
└── README.md            # Hướng dẫn chạy nhanh và một số hợp đồng dự án
```

Trong bản workspace được khảo sát, chưa thấy CSV dữ liệu trong `data/`; các lệnh train/predict mặc định cần `data/raw/acb.csv`. Cần đặt dữ liệu ở đó hoặc truyền `--csv` trỏ tới CSV OHLCV hợp lệ.

## 3. Luồng hoạt động tổng thể

```text
CSV OHLCV
   │
   ├─ loader / validator / preprocessing
   │
   ├─ build_features ──> đặc trưng chỉ từ hiện tại và quá khứ
   ├─ add_future_volatility_target ──> nhãn t+1 ... t+horizon
   │
   ├─ prepare_dataset ──> train / validation / test theo thời gian
   │                         └─ loại các origin gần ranh giới (purge)
   ├─ train_pipeline
   │     ├─ GARCH / XGBoost / LSTM
   │     ├─ chọn tham số XGBoost trên validation
   │     ├─ đánh giá trên các ngày test chung
   │     └─ lưu model, dự báo, ngưỡng cảnh báo, metrics, metadata, SHAP
   │
   ├─ predict_pipeline ──> nạp artifact và dự báo origin mới nhất
   └─ Streamlit ──> đọc artifact để hiển thị tổng quan/so sánh/giải thích
```

### Các bước dữ liệu

1. Đọc CSV gồm `date, open, high, low, close, volume`.
2. Chuẩn hóa tên cột, đổi bí danh ngày như `time`/`datetime`/`timestamp`, kiểm tra số liệu và quan hệ OHLC. Chế độ `clean=True` loại hàng sai định dạng và ngày trùng (giữ bản ghi cuối); giá không hợp lệ, volume âm hoặc quan hệ OHLC sai vẫn gây lỗi.
3. Tạo lợi suất và đặc trưng. Những cửa sổ đầu chưa đủ lịch sử sẽ có NaN và không được dùng làm mẫu huấn luyện.
4. Tạo nhãn biến động tương lai. Các hàng cuối chưa đủ horizon được giữ lại nhưng nhãn là NaN; chúng vẫn có thể là origin dự báo mới nhất.
5. Chia các mẫu đã có đủ đặc trưng và nhãn theo thứ tự ngày. Tỷ lệ cấu hình là 70%/15%/15%; `horizon` origin trước các ranh giới bị bỏ để phần lợi suất tương lai của nhãn không chồng sang tập kế tiếp. Vì vậy kích thước thực tế nhỏ hơn tỷ lệ thô.
6. XGBoost chọn tham số trên validation. Test được dùng để báo cáo cuối cùng, không dùng để chọn tham số.

## 4. Các khái niệm dữ liệu và mô hình

### OHLCV và lợi suất

- **OHLCV:** giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng.
- **Log return một phiên:** `100 × ln(close_t / close_(t-1))`. Nhân 100 giúp đơn vị là điểm phần trăm, ví dụ `-2.6` nghĩa là khoảng -2.6 điểm phần trăm lợi suất log.
- **Đặc trưng (feature):** đầu vào mô hình, được tính từ dữ liệu đã biết tại ngày dự báo hoặc trước đó.
- **Nhãn (target):** giá trị mô hình cần học. `target_volatility_5d` tại ngày `t` là độ lệch chuẩn mẫu (`ddof=1`) của return ở `t+1...t+5`, tính theo điểm phần trăm và không annualize.
- **Data leakage:** vô tình đưa thông tin tương lai vào đầu vào hoặc quá trình chọn mô hình. Dự án cấm cột `date` và `target_*` trong feature schema; chia thời gian và purge ranh giới để giảm rò rỉ do nhãn bao phủ nhiều phiên.

Các feature mặc định: `return_1d`, `return_5d`, `return_10d`, `return_20d`, `close_to_ma_5`, `close_to_ma_10`, `close_to_ma_20`, `close_to_ma_50`, `rolling_volatility_5`, `rolling_volatility_20`, `volume_change`, `volume_to_ma_20`, `high_low_range`, `open_close_range`. Rolling volatility là độ lệch chuẩn các return lịch sử; các feature `close_to_ma_*` và `volume_to_ma_20` biểu diễn độ lệch tương đối so với trung bình trượt.

### Các mô hình

- **GARCH(1,1):** ước lượng `omega`, `alpha`, `beta` chỉ trên return train; sau đó lọc các return đã quan sát đến origin để tính phương sai có điều kiện. Căn bậc hai trung bình phương sai dự báo nhiều bước được dùng làm xấp xỉ target. Nó không hoàn toàn đồng nhất về mặt toán học với độ lệch chuẩn mẫu của năm return tương lai.
- **XGBoost:** hồi quy trên đúng 14 feature theo thứ tự cấu hình. Tuning thử 15 tổ hợp mặc định hoặc 2 tổ hợp với `--fast`; chọn average precision HIGH tốt nhất trong các tổ hợp có RMSE validation không quá 5% so với tốt nhất. Dự báo hồi quy âm được giữ trong `raw_prediction` để phân tích, nhưng forecast hiển thị được chặn dưới ở 0 vì độ biến động không âm.
- **LSTM:** nhận chuỗi `sequence_length` ngày kết thúc tại origin. `StandardScaler` chỉ fit trên feature của train; validation dùng early stopping để chọn trọng số. Với cấu hình mặc định, cửa sổ dài 20 phiên. Dự báo âm cũng được chặn ở 0.
- **Seed:** cấu hình mặc định 42 để tăng khả năng lặp lại, dù một số phép tính thư viện/phần cứng có thể vẫn khác biệt nhỏ.

### Cảnh báo rủi ro và đánh giá

`WarningEngine` tính quantile 50% và 70% của target train để lập hai ngưỡng:

- `NORMAL`: dưới ngưỡng WARNING.
- `WARNING`: từ ngưỡng WARNING đến dưới ngưỡng HIGH.
- `HIGH`: bằng hoặc vượt ngưỡng HIGH.

Đây là phân loại mức biến động tương đối theo phân phối train, không phải xác suất thua lỗ hoặc tín hiệu mua/bán. Nhãn HIGH thực tế dùng ngưỡng từ train; ngưỡng trên điểm dự báo được hiệu chỉnh riêng cho từng model trên validation. Ngân sách cảnh báo XGBoost hiện là tối đa 30% ngày validation, so với mức mặc định 15% cho GARCH/LSTM.

MAE và RMSE đo sai số dự báo; R² đo mức phù hợp khi có đủ mẫu. Chỉ số precision/recall/F1 và confusion matrix đánh giá riêng việc nhận diện nhãn HIGH. Các mô hình được căn theo những ngày test chung trước khi so sánh; vì vậy một số origin test có thể bị loại nếu mô hình không tạo được dự báo tương ứng.

**SHAP** trong dự án chỉ giải thích đầu ra thô của XGBoost. `mean_abs_shap` tổng hợp độ lớn đóng góp cho mức độ quan trọng toàn cục; SHAP theo từng ngày mô tả đóng góp của feature vào đầu ra của dự báo đó. Đây là giải thích mối liên hệ trong mô hình, không chứng minh quan hệ nhân quả. GARCH và LSTM chưa có giải thích SHAP.

## 5. Chức năng từng thư mục và file Python

### `src/`

#### Cấu hình và dữ liệu

- `src/config.py` — xác định root/config mặc định; đọc YAML và kiểm tra horizon, sequence length, feature trùng/cấm, cùng tổng tỷ lệ split.
- `src/data/loader.py` — tìm và đọc `data/raw/<ticker>.csv`; kiểm tra mã ticker không rỗng và báo lỗi nếu không có file.
- `src/data/validator.py` — chuẩn hóa schema OHLCV, parse ngày/số, kiểm tra OHLC, giá dương, volume không âm, ngày trùng và thứ tự thời gian.
- `src/data/preprocessing.py` — nối loader với validator; `clean_data` làm sạch DataFrame, `preprocess_data` tùy chọn ghi CSV sạch vào `data/processed`.

#### Đặc trưng và target

- `src/features/builder.py` — tạo log return, return nhiều kỳ, moving average, rolling volatility, biến động volume, volume MA và biên độ giá.
- `src/features/target.py` — tạo độ lệch chuẩn return của các phiên tương lai làm nhãn; giữ NaN cho các ngày cuối không đủ horizon.

#### Pipeline

- `src/pipeline/dataset.py` — `Dataset` gom frame đầy đủ và ba split; `prepare_dataset` dựng lại dữ liệu từ CSV thô, kiểm tra ngày/schema, tạo split có purge; `valid_history` giữ hàng có feature hợp lệ kể cả hàng chưa có target; `feature_matrix` chỉ chọn đúng danh sách feature và từ chối NaN/Inf hoặc target/date.
- `src/pipeline/train_pipeline.py` — điều phối huấn luyện, tuning, dự báo test/latest, căn ngày, tính metrics, phân loại cảnh báo, và ghi artifact vào `artifacts/<TICKER>/`.
- `src/pipeline/predict_pipeline.py` — nạp metadata/model đã lưu, kiểm tra schema và cấu hình hiện tại có tương thích, tạo dự báo origin mới nhất từ lịch sử CSV và áp dụng ngưỡng cảnh báo đã lưu.

#### Mô hình

- `src/models/base.py` — hợp đồng output chung, đơn vị dự báo, xác thực ngày origin và tạo bảng forecast hợp lệ.
- `src/models/garch_model.py` — fit và dự báo GARCH(1,1), lưu/nạp bằng joblib.
- `src/models/xgboost_model.py` — wrapper hồi quy XGBoost; kiểm soát feature schema, raw/clipped forecast và lưu/nạp model.
- `src/models/xgboost_tuning.py` — thử các tổ hợp XGBoost trên train/validation theo thời gian và trả cấu hình có RMSE validation thấp nhất.
- `src/models/lstm_model.py` — tạo các cửa sổ chuỗi, chuẩn hóa feature từ train, huấn luyện LSTM với early stopping, dự báo, lưu/nạp.

#### Đánh giá, rủi ro, giải thích

- `src/evaluation/backtest.py` — lấy giao ngày giữa forecast các mô hình và actual để so sánh công bằng.
- `src/evaluation/regression.py` — tính MAE, RMSE, R² và số mẫu.
- `src/evaluation/warning_metrics.py` — tính precision, recall, F1, support và confusion matrix cho lớp HIGH.
- `src/risk/warning_engine.py` — fit quantile ngưỡng cảnh báo trên target train; phân loại NORMAL/WARNING/HIGH; lưu/nạp JSON.
- `src/explainability/shap_explainer.py` — tính SHAP toàn cục và theo từng mẫu cho XGBoost; kiểm tra tính cộng của base value và SHAP so với raw output.

#### Tiện ích

- `src/utils/paths.py` — hiện là file rỗng, chưa có hàm tiện ích đường dẫn.
- `src/utils/logger.py` — hiện là file rỗng, chưa có cấu hình logging tập trung.

### `scripts/`

- `scripts/train.py` — CLI gọi `run_training`; nhận `--ticker`, `--model all|garch|xgboost|lstm`, `--fast`, `--csv`.
- `scripts/predict.py` — CLI dự báo mới nhất bằng artifact đã lưu; nhận ticker/model và CSV tùy chọn.
- `scripts/evaluate.py` — in `metrics.json` đã tạo trước đó; không huấn luyện lại.
- `scripts/download_data.py` — tải lịch sử qua `vnstock` và lưu trong `data/raw`. File này là bước chủ động riêng, không được pipeline offline tự gọi. `vnstock` không thấy khai báo trong `requirements.txt`, nên muốn dùng script cần cài dependency phù hợp riêng.

### `app/`

- `app/app.py` — entry point Streamlit; đọc CSV/JSON artifacts có cache và điều hướng bốn trang. Trang tổng quan lấy dữ liệu cục bộ; các trang kết quả dựa trên artifact.
- `app/pages/overview.py` — hiển thị khoảng ngày, số phiên, giá đóng cửa và volatility lịch sử 20 phiên từ CSV.
- `app/pages/forecast.py` — hiển thị forecast latest, nhãn rủi ro/ngưỡng train và biểu đồ test của mô hình chọn.
- `app/pages/comparison.py` — bảng metrics các mô hình và biểu đồ actual so với forecast trên test chung.
- `app/pages/explanation.py` — hiển thị global importance và giải thích SHAP cục bộ nếu đã có `shap.json`.
- `app/components/charts.py` — hàm dùng chung tạo biểu đồ đường Plotly cho chuỗi và so sánh dự báo/thực tế.

### `tests/`

- `tests/conftest.py` — fixture OHLCV tổng hợp có seed, không phụ thuộc Internet hay dữ liệu thị trường.
- `tests/test_dataset_contract.py` — kiểm tra scale/alignment feature-target, cấm target làm đầu vào, purge split và đường dẫn CSV tùy chọn.
- `tests/test_features.py` — file hiện có nhưng rỗng.
- `tests/test_target.py` — file hiện có nhưng rỗng.
- `tests/test_models_and_risk.py` — kiểm tra GARCH/XGBoost/LSTM, khả năng lưu/nạp, chuỗi LSTM, scaler chỉ fit trên train và ngưỡng cảnh báo.
- `tests/test_warning_engine.py` — file hiện có nhưng rỗng.
- `tests/test_shap_and_integration.py` — kiểm tra tính cộng SHAP; có bài chạy end-to-end ACB đánh dấu `slow`.
- `tests/` còn `test_warning_engine.py` và `test_target.py`; hai file này hiện rỗng, các kiểm tra liên quan nằm trong các file test khác.

### `notebooks/`

- `01_eda.ipynb` — notebook khám phá dữ liệu (EDA).
- `02_garch_experiment.ipynb` — thử nghiệm GARCH.
- `03_xgboost_experiment.ipynb` — thử nghiệm XGBoost.
- `04_lstm_experiment.ipynb` — thử nghiệm LSTM.

Notebook phục vụ phân tích/thử nghiệm tương tác. Luồng chạy tái lập chính được đóng gói trong `src/` và `scripts/`. Tên notebook cho biết chủ đề; chúng không phải entry point của CLI.

### File cấu hình và tài liệu

- `config.yaml` — ticker mặc định, offline, horizon, sequence length, danh sách feature, tỷ lệ chia dữ liệu, quantile cảnh báo, seed, tham số mô hình và đường dẫn raw/artifact.
- `requirements.txt` — thư viện chính gồm pandas/numpy, scikit-learn, arch, xgboost, torch, shap, Streamlit, Plotly, PyYAML, joblib và pytest.
- `pytest.ini` — cấu hình pytest; trong đó có marker cho kiểm thử chậm.
- `.gitignore` — bỏ qua môi trường ảo, cache, artifacts và dữ liệu processed sạch. Nó không tự xóa file đã được Git track trước đó.
- `README.md` — hướng dẫn cài đặt/chạy và ghi lại các hợp đồng quan trọng về đơn vị, split, artifact và giới hạn prototype.
- `overview.md` — tài liệu giải thích cấu trúc và luồng dự án này.

## 6. Artifact sau khi train

`artifacts/<TICKER>/` được tạo khi chạy train và bị Git ignore. Các file chính:

| File | Mục đích |
|---|---|
| `<model>.joblib` | Mô hình đã fit và thông tin cần để dự báo |
| `metadata.json` | Ticker, target, horizon, feature schema, đơn vị, cấu hình, các khoảng/số mẫu split, số ngày test khớp |
| `warning.json` | Quantile và ngưỡng WARNING/HIGH fit trên train |
| `train_predictions.csv` | Dự báo test và latest được tạo trong pipeline train, actual nếu có, nhãn rủi ro và partition |
| `metrics.json` | Metrics hồi quy và phân loại HIGH theo từng model |
| `shap.json` | Global/local SHAP cho XGBoost khi XGBoost được train |

`partition=test` có actual để đánh giá; `partition=latest` chưa có actual vì các phiên tương lai chưa diễn ra. Artifact gắn với ticker/schema/cấu hình; pipeline predict từ chối model đã lưu nếu metadata hiện tại không khớp.

## 7. Cách bắt đầu đọc và chạy

Nên lần lượt đọc `config.yaml`, `src/data/validator.py`, `src/features/builder.py`, `src/features/target.py`, `src/pipeline/dataset.py`, rồi `src/pipeline/train_pipeline.py`. Sau đó đọc từng model và dashboard.

Từ thư mục gốc repository, README hướng dẫn các lệnh:

```powershell
python -m scripts.train --ticker ACB --model all --fast
python -m scripts.evaluate --ticker ACB
python -m scripts.predict --ticker ACB --model xgboost
python -m streamlit run app/app.py
```

Trước khi chạy, cần cài `requirements.txt` và cung cấp CSV OHLCV phù hợp. Có thể dùng `--csv <đường-dẫn>` cho train/predict. `--fast` rút tìm kiếm XGBoost còn hai cấu hình và giới hạn LSTM tối đa ba epoch; kết quả dùng để thử nhanh, không thay cho đánh giá nghiêm túc.

## 8. Giới hạn cần nhớ khi diễn giải

- Dữ liệu đầu vào, ticker, quy ước giao dịch và chất lượng CSV quyết định kết quả; pipeline không tự chứng minh dữ liệu mới đáng tin.
- Độ biến động tương lai là độ phân tán return, không phải hướng tăng/giảm hay dự báo giá.
- GARCH forecast là xấp xỉ cho target độ lệch chuẩn mẫu; so sánh metric cần nhớ hai cách đo không hoàn toàn đồng nhất.
- Ngưỡng rủi ro dựa trên phân phối train và có thể không còn đại diện nếu phân phối thị trường đổi.
- Dashboard và artifact là kết quả offline, không cập nhật thời gian thực.
- SHAP mô tả cách mô hình XGBoost tạo đầu ra, không xác lập nguyên nhân ngoài đời.
- Script tải dữ liệu dùng provider bên ngoài; pipeline huấn luyện/đánh giá/dự báo không tải dữ liệu tự động.
