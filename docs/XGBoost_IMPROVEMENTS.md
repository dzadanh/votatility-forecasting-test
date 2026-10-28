# XGBoost: cải tiến dự báo độ biến động và cảnh báo HIGH

Tài liệu này ghi lại thay đổi của XGBoost trên bốn mã **ACB, FPT, HPG, VIC** và so sánh với metrics trước khi cải tiến. Bài toán vẫn là dự báo độ lệch chuẩn mẫu của return trong **5 phiên giao dịch kế tiếp**; HIGH là mức độ biến động tương đối, không phải tín hiệu giao dịch. Các kết quả dưới đây là thử nghiệm lịch sử trên dữ liệu cục bộ.

## 1. Mốc so sánh và cách kiểm tra

- **Bản cũ:** XGBoost được chọn theo MSE trên validation; ngân sách cảnh báo HIGH tối đa 15% số origin validation. Trước khi người dùng train lại, metrics cũ nằm ở `artifacts/<TICKER>/`. Sau khi thư mục đó được ghi bằng bản mới, baseline đã được **dựng lại** tại `artifacts/xgboost_baseline_reproduced/<TICKER>/` bằng đúng tham số XGBoost đã chọn trước đây và ngân sách 15%. Bốn file `config_<TICKER>.yaml` trong thư mục này ghi cấu hình dựng lại. Các metrics dựng lại khớp với số đã đọc từ artifact cũ trước khi ghi đè; đây không phải bản sao nguyên trạng của model cũ.
- **Chỉ đổi cách chọn XGBoost:** artifact ở `artifacts/xgboost_high_trial/<TICKER>/` dùng mô hình mới nhưng vẫn giữ ngân sách 15% trên validation.
- **Cấu hình hiện tại:** `artifacts/<TICKER>/` là lần người dùng train lại với mô hình mới và ngân sách XGBoost 30% trên validation. Artifact thử nghiệm độc lập ở `artifacts/xgboost_high_budget30_trial/<TICKER>/` cho cùng metrics XGBoost.

Đã đối chiếu `date`, `actual_volatility` và `actual_risk_level` của từng hàng XGBoost trên test giữa baseline dựng lại và artifact chính: **khớp cả bốn mã**. ACB/FPT có 291 origin test từ 11/08/2025 đến 12/10/2026; HPG có 260 origin từ 01/10/2025 đến 16/10/2026; VIC có 259 origin từ 01/10/2025 đến 15/10/2026. Cả hai mốc dùng cùng định nghĩa HIGH thực tế: quantile **0,70** của target trên train. `mota.txt` và một số tài liệu cũ mô tả quantile 0,90; cấu hình và artifact của phép so sánh này dùng 0,70.

## 2. Những cải tiến đã áp dụng

### Mở rộng khả năng học của XGBoost

Lưới cũ chỉ thử **1–5 cây**; cả bốn mã đều chọn **5 cây**, độ sâu 3 và learning rate 0,11. Lưới mới có 15 cấu hình: ba cấu hình 5 cây cũ để làm mốc và 12 cấu hình dùng **50, 150 hoặc 300 cây**, độ sâu 2 hoặc 3, learning rate 0,03 hoặc 0,10. Tỷ lệ lấy hàng/cột của nhóm mới là 0,8. Năm fold thời gian tạo 75 lượt fit để chọn tham số, thay vì 8.100 lượt fit của lưới cũ. Feature đầu vào vẫn là 14 feature trong `config.yaml`; XGBoost vẫn hồi quy độ biến động bằng `reg:squarederror`, không có đầu phân loại HIGH riêng.

### Chọn tham số theo mục tiêu HIGH và giữ sai số dự báo

`src/models/xgboost_tuning.py` đo **average precision (AP)** của HIGH và RMSE trên năm fold thời gian. Ngưỡng xác định HIGH cho việc chọn tham số được tính từ train của fold đầu tiên, trước các cửa sổ validation. Bộ chọn lấy cấu hình có AP cao nhất trong các cấu hình có RMSE validation không quá **5%** so với RMSE tốt nhất. Mỗi cấu hình và hai điểm số được lưu trong `metadata.json`; mô hình được fit lại trên toàn bộ phần trước test sau khi chọn tham số.

Tham số được chọn lần này: **ACB/HPG** dùng 300 cây, độ sâu 3, learning rate 0,03; **FPT** dùng 50 cây, độ sâu 2, learning rate 0,10; **VIC** dùng 50 cây, độ sâu 3, learning rate 0,10. Cả bốn dùng `subsample=0.8` và `colsample_bytree=0.8`. Việc ACB/HPG chọn 300 cây, biên trên của lưới mới, là lý do hợp lý để thử thêm cấu hình trong nghiên cứu tiếp theo; kết quả hiện tại chưa chứng minh thêm cây sẽ tiếp tục tốt hơn.

### Hiệu chỉnh riêng ngân sách cảnh báo

`alert_policy.model_overrides.xgboost.max_alert_rate` cho phép XGBoost dùng tối đa **30%** origin validation làm cảnh báo HIGH, thay cho mức mặc định **15%** của GARCH/LSTM. Điều kiện precision tối thiểu trên validation vẫn là **25%**. Bộ hiệu chỉnh chọn ngưỡng dự báo theo recall trong các ràng buộc đó. Đây là ngân sách **trên validation**, không phải giới hạn cứng trên test; tỷ lệ HIGH trên test có thể cao hơn 30%. `app/pages/comparison.py` hiển thị chính sách riêng của mô hình, và `scripts.train`/`scripts.evaluate` có `--artifact-dir` để lưu, đọc thử nghiệm mà không ghi đè artifact chính.

## 3. Kết quả trên cùng ngày test: bản cũ → cấu hình hiện tại

`Tỷ lệ cảnh báo` = số origin test được dự báo HIGH / tổng origin test. `Đúng` là HIGH thực tế được phát hiện (TP); `sai` là cảnh báo HIGH khi thực tế không HIGH (FP). Precision, recall và F1 đều xét riêng lớp HIGH. MAE/RMSE tính trên độ biến động liên tục; **thấp hơn là tốt hơn**. R² cao hơn là tốt hơn.

- **ACB — 291 origin, 89 HIGH thực tế:** MAE **0,6132 → 0,6342**; RMSE **0,8687 → 0,8501**; R² **0,0334 → 0,0744**. Cảnh báo HIGH **58 → 141** (**19,9% → 48,5%**), gồm **16 đúng / 42 sai → 37 đúng / 104 sai**. Precision **27,6% → 26,2%**; recall **18,0% → 41,6%**; F1 **0,2177 → 0,3217**.
- **FPT — 291 origin, 144 HIGH thực tế:** MAE **0,6195 → 0,6427**; RMSE **0,8230 → 0,8121**; R² **0,0190 → 0,0448**. Cảnh báo HIGH **76 → 169** (**26,1% → 58,1%**), gồm **53 đúng / 23 sai → 99 đúng / 70 sai**. Precision **69,7% → 58,6%**; recall **36,8% → 68,8%**; F1 **0,4818 → 0,6326**.
- **HPG — 260 origin, 44 HIGH thực tế:** MAE **0,6442 → 0,6009**; RMSE **0,8147 → 0,8271**; R² **-0,0340 → -0,0659**. Cảnh báo HIGH **10 → 53** (**3,8% → 20,4%**), gồm **3 đúng / 7 sai → 15 đúng / 38 sai**. Precision **30,0% → 28,3%**; recall **6,8% → 34,1%**; F1 **0,1111 → 0,3093**.
- **VIC — 259 origin, 188 HIGH thực tế:** MAE **1,2308 → 1,1253**; RMSE **1,5768 → 1,4201**; R² **-0,4763 → -0,1976**. Cảnh báo HIGH **130 → 189** (**50,2% → 73,0%**), gồm **107 đúng / 23 sai → 142 đúng / 47 sai**. Precision **82,3% → 75,1%**; recall **56,9% → 75,5%**; F1 **0,6730 → 0,7533**.

Tỷ lệ cảnh báo, recall và F1 HIGH tăng ở cả bốn mã. **Precision giảm ở cả bốn**, và số cảnh báo sai tăng rõ, nhất là ACB (**+62**) và FPT (**+47**). RMSE giảm ở ACB, FPT, VIC nhưng tăng nhẹ ở HPG; MAE tăng ở ACB và FPT. Vì vậy không thể kết luận mọi mặt của XGBoost đều tốt hơn.

### Tách tác động của mô hình và ngân sách cảnh báo

Với **mô hình mới nhưng vẫn giữ ngân sách 15%**, kết quả test trung gian là:

- **ACB:** 63 cảnh báo, 27 đúng / 36 sai; recall **30,3%**, F1 **0,3553**. F1 cao hơn bản 30% (**0,3217**) dù phát hiện ít HIGH hơn.
- **FPT:** 117 cảnh báo, 73 đúng / 44 sai; recall **50,7%**, F1 **0,5594**.
- **HPG:** 26 cảnh báo, 7 đúng / 19 sai; recall **15,9%**, F1 **0,2000**.
- **VIC:** 91 cảnh báo, 76 đúng / 15 sai; recall **40,4%**, F1 **0,5448**. Ở mức 15%, F1 này thấp hơn baseline cũ (**0,6730**); mức 30% làm recall và F1 tăng nhưng cũng thêm cảnh báo sai.

RMSE của mốc trung gian bằng cấu hình hiện tại vì chỉ thay **ngưỡng cảnh báo**, không thay dự báo độ biến động. Những số này giúp phân biệt tác động của mô hình mới với tác động của việc cho phép nhiều cảnh báo hơn.

## 4. Giới hạn và cách chạy lại

Các giai đoạn test đã được xem nhiều lần trong quá trình phát triển, bao gồm việc so ngân sách 15% và 30%. Vì vậy đây là **so sánh lịch sử**, không phải đánh giá trên holdout chưa từng xem. Cần dữ liệu giao dịch đến sau giai đoạn test để kiểm tra khả năng lặp lại, đặc biệt là số cảnh báo sai. `alert_count` đếm các **ngày/origin HIGH**; `high_alert_event` đếm lần mới chuyển sang HIGH và có thể ít hơn nhiều.

Tại thư mục gốc repository, có thể chạy lại vào một thư mục riêng:

```powershell
python -m scripts.train --ticker ACB --model xgboost --artifact-dir artifacts/xgboost_repeat
python -m scripts.evaluate --ticker ACB --artifact-dir artifacts/xgboost_repeat
```

Thay `ACB` bằng `FPT`, `HPG` hoặc `VIC` để chạy các mã còn lại; **không dùng `--fast` để đánh giá chất lượng**. Các artifact trong `artifacts/` thường bị Git ignore. Baseline dựng lại phụ thuộc vào dữ liệu thô hiện có và các file cấu hình lưu ở `artifacts/xgboost_baseline_reproduced/`; khi chia sẻ kết quả, cần giữ cả provenance này hoặc tạo bản xuất riêng vì Git không tự đưa artifact lên repository.
