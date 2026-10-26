# XGBoost: quá trình cải tiến và kết quả hiện tại

Tài liệu này ghi lại các cải tiến đang có trong mã nguồn XGBoost, cách chọn tham số và kết quả trong artifact hiện tại. Mục tiêu là dự báo độ biến động thực tế trong 5 phiên giao dịch tiếp theo. Target được tính bằng độ lệch chuẩn của lợi suất ngày từ `t+1` đến `t+5`. Các con số dưới đây là snapshot của dữ liệu và artifact trong repo, không bảo đảm lặp lại ở lần chạy hoặc giai đoạn khác.

## 1. Bản XGBoost ban đầu và vấn đề

- XGBoost hồi quy dự báo một giá trị liên tục `target_volatility_5d` từ các đặc trưng kỹ thuật tại ngày dự báo.
- Chọn tham số thủ công hoặc qua một lần chia train/validation dễ phụ thuộc vào một khoảng thời gian cụ thể. Dữ liệu chuỗi thời gian cần được chia theo thứ tự thời gian.
- Cây quá sâu, nhiều cây hoặc lấy mẫu quá rộng có thể khiến mô hình bám sát nhiễu của tập train. Cần chọn cấu hình trên nhiều fold thời gian và giữ riêng giai đoạn test.

## 2. Những cải tiến đã thực hiện

### Tuning đồng thời trên các fold theo thời gian

`src/models/xgboost_tuning.py` dùng `GridSearchCV` trên các fold thời gian đã tạo trong dataset. Hàm ánh xạ index của từng fold về đúng vị trí trong ma trận đặc trưng và từ chối fold rỗng, index không hợp lệ hoặc train/validation bị chồng lấn.

Tuning tối ưu MSE (`neg_mean_squared_error`) và ghi RMSE từng fold, RMSE gộp theo MSE, cấu hình tốt nhất, số ứng viên và baseline. Baseline gồm dự báo bằng trung bình target của fold train và persistence bằng `rolling_volatility_5` nếu feature này có trong schema.

### Không dùng test để chọn tham số

Grid search chạy với `refit=False`. Sau khi chọn tham số, pipeline fit mô hình cuối trên phần dữ liệu trước test; test cuối chỉ dùng để đánh giá. Dataset hiện chia fold mở rộng theo thời gian, chừa `gap=5` tại ranh giới để hạn chế target 5 phiên tương lai chồng lấn, và giữ test cuối theo thời gian.

### Thu hẹp độ phức tạp và kiểm soát tài nguyên

Grid hiện tại trong `config.yaml` gồm:

- `n_estimators`: 1–5
- `max_depth`: 1–3
- `learning_rate`: 0.01, 0.03, 0.05, 0.07, 0.09, 0.11
- `subsample`: 0.40–0.65, bước 0.05
- `colsample_bytree`: 0.1, 0.2, 0.3

Grid đầy đủ có 1.620 cấu hình và 5 fold. `fast_param_grid` chỉ có 2 cấu hình để chạy nhanh. Song song hóa ở tầng `GridSearchCV` (`search_n_jobs=2`), còn mỗi estimator mặc định dùng một luồng (`n_jobs=1`) để tránh tranh chấp CPU.

Grid giới hạn số cây và độ sâu, đồng thời thử lấy mẫu một phần dữ liệu và đặc trưng. `min_child_weight=1` và `reg_lambda=1.0` được giữ từ cấu hình cơ sở, không nằm trong grid tìm kiếm.

### Kiểm soát feature và dữ liệu

`feature_matrix` chỉ lấy đúng danh sách feature đã cấu hình, từ chối cột thiếu, NaN/vô cực, và không cho phép `date` hoặc target làm đầu vào. Target được tạo từ lợi suất tương lai, không được đưa vào feature tại cùng origin. Wrapper lưu schema feature, target và horizon cùng mô hình.

### Metadata để truy vết

Pipeline lưu cấu hình XGBoost cuối cùng, số fold và ứng viên, điểm validation từng fold, baseline và kết quả tuning trong metadata. Kết quả test lưu riêng trong `metrics.json`; lần train mới có thể ghi đè artifact.

## 3. Kết quả trong artifact hiện tại

Bảng dưới đây lấy từ `artifacts/<MÃ>/metrics.json` và `metadata.json`. RMSE đo sai số dự báo độ biến động. R² gần 0 hoặc âm cho thấy mô hình chưa giải thích tốt biến thiên target trong giai đoạn tương ứng. Precision/recall/F1 HIGH phụ thuộc ngưỡng cảnh báo hiệu chỉnh trên validation, là mục tiêu bổ sung chứ không thay thế metric hồi quy.

| Mã | CV RMSE trung bình* | Test RMSE | Test R² | HIGH precision | HIGH recall | HIGH F1 |
|---|---:|---:|---:|---:|---:|---:|
| ACB | 0.7648 | 0.8798 | 0.0086 | 0.2389 | 0.3034 | 0.2673 |
| FPT | 0.7419 | 0.8254 | 0.0133 | 0.5758 | 0.3958 | 0.4691 |
| HPG | 0.9042 | 0.8177 | -0.0417 | 0.4286 | 0.0682 | 0.1176 |
| VIC | 1.1239 | 1.6354 | -0.5882 | 0.7895 | 0.3191 | 0.4545 |

\* `mean_validation_rmse` là trung bình số học RMSE của từng fold; `validation_rmse` trong metadata là căn bậc hai của MSE trung bình gộp. Đơn vị target phụ thuộc cách scale lợi suất trong pipeline.

CV RMSE của XGBoost thấp hơn baseline dự báo trung bình target trên cả bốn mã. So với persistence `rolling_volatility_5`, XGBoost tốt hơn ở ACB và VIC, nhưng kém hơn nhẹ ở FPT và HPG. Kết quả test không đồng đều: ACB/FPT có R² dương rất nhỏ, HPG/VIC có R² âm; VIC có RMSE test cao. Quy trình tuning đã chuẩn hóa cách chọn tham số theo thời gian, nhưng kết quả hiện tại chưa chứng minh khả năng tổng quát hóa tốt trên mọi mã.

Cả bốn artifact chọn `n_estimators=5`, `learning_rate=0.11`, `colsample_bytree=0.3`; `max_depth` là 2 hoặc 3 và `subsample` nằm trong khoảng 0.40–0.55. Nghiệm ở biên trên của một số khoảng tìm kiếm; cần mở rộng grid có kiểm soát và kiểm tra trên giai đoạn mới trước khi kết luận cấu hình tối ưu.

## 4. Giới hạn và hướng cải tiến tiếp theo

- `src/pipeline/dataset.py` hiện dùng expanding chronological folds (`max_train_size=None`) và test holdout cuối; chưa triển khai rolling window giới hạn 12 tháng hoặc Moving Block Bootstrap. Nếu yêu cầu rolling 12 tháng/MBB vẫn còn hiệu lực, cần triển khai trong bộ tạo dataset trước khi mô tả đó là hành vi hiện tại.
- Target có horizon 5 phiên nên các origin gần nhau có thể có nhãn chồng lấn. `gap=5` xử lý ranh giới fold/test và cần được giữ khi thay đổi cách chia dữ liệu.
- R² âm ở HPG/VIC và chênh lệch CV/test cho thấy cần đánh giá trên nhiều giai đoạn walk-forward mới. Không nên chọn model chỉ theo một giá trị validation hoặc test đã dùng nhiều lần để thử nghiệm.
- Feature importance bằng 0 chỉ cho biết feature không được dùng trong cây ở lần fit đó. Với `colsample_bytree` thấp và tối đa 5 cây, điều này có thể xảy ra tự nhiên. Chỉ loại feature sau khi so sánh có/không feature trên cùng fold thời gian và xác nhận trên giai đoạn độc lập.
- Grid hiện chỉ thử 1–5 cây. Nên so sánh thêm một grid mở rộng về `n_estimators` và regularization trên validation theo thời gian; không mở rộng chỉ để giảm training error.
- Ngưỡng cảnh báo HIGH được chọn trên validation; tỷ lệ cảnh báo và precision/recall trên test có thể khác. Cần báo cáo cả regression metrics, baseline, số cảnh báo và chất lượng phát hiện HIGH.

## 5. Chạy lại

Từ thư mục gốc repo:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --ticker ACB --model xgboost
```

Thay `ACB` bằng `FPT`, `HPG` hoặc `VIC` để chạy mã khác. Cấu hình đầy đủ đánh giá 1.620 ứng viên trên 5 fold; dùng `--fast` để chạy grid nhanh 2 cấu hình. Kết quả nằm trong `artifacts/<MÃ>/`: `metrics.json` chứa đánh giá test, `metadata.json` chứa cấu hình và kết quả tuning. Artifact và dữ liệu nguồn có thể bị Git ignore.
