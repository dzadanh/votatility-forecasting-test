# LSTM: quá trình cải tiến từ bản đầu

Tài liệu này ghi lại những thay đổi trong **mã nguồn LSTM** và kết quả đã đo trên dữ liệu ACB, FPT, HPG, VIC. Mục tiêu dự án vẫn là dự báo độ biến động của **5 phiên giao dịch kế tiếp**; mục tiêu đánh giá bổ sung là phát hiện những phiên có nhãn **HIGH**. Các số dưới đây là kết quả thử nghiệm lịch sử, không phải cam kết trên dữ liệu mới.

## 1. Bản LSTM lúc đầu

- Một LSTM nhỏ với `hidden_size=32`, đọc 20 phiên đặc trưng kết thúc ở ngày dự báo, xuất một số duy nhất: độ biến động dự báo. Ví dụ, chuỗi 20 phiên ở ngày `t` dùng dữ liệu từ `t-19` đến `t`, rồi dự báo độ biến động của `t+1` đến `t+5`.
- Feature có MA giá và MA khối lượng ở **mức tuyệt đối** (`MA_5`, `MA_10`, `MA_20`, `MA_50`, `volume_ma_20`). Mức giá của các mã khác nhau có thể chênh lệch lớn; scaler được fit trên train nhưng feature tương đối vẫn dễ diễn giải hơn khi dữ liệu thay đổi thang đo.
- Train một lần trên tập train theo thời gian, dùng validation để dừng sớm theo MSE, tối đa 25 epoch và `patience=5`. Không lưu diễn biến train/validation qua từng epoch hay epoch tốt nhất trong metadata.
- Cảnh báo HIGH dựa trên việc độ biến động dự báo vượt một ngưỡng được chọn trên validation. Model không có đầu ra riêng để xếp hạng khả năng HIGH.

## 2. Những thay đổi đã thực hiện

### Feature đầu vào tương đối

Đổi MA giá thành `close_to_ma_N = close / MA_N - 1` và MA khối lượng thành `volume_to_ma_20 = volume / volume_MA_20 - 1`. Đây là thay đổi của bộ feature **dùng chung cho các model**, không phải thay kiến trúc LSTM. Kiểm thử xác nhận nhân toàn bộ giá hoặc khối lượng với một hằng số không làm đổi các feature này. Artifact cũ có schema khác phải được train lại trước khi dùng với config mới.

### Theo dõi quá trình train và thử tham số

LSTM ghi `train_mse`, `validation_mse` theo epoch, `best_epoch` và loss validation tốt nhất vào metadata. `epochs=500` trong config là **giới hạn tối đa**; nếu validation không cải thiện trong `patience=5` epoch liên tiếp thì dừng và lấy trọng số của epoch tốt nhất. Vì vậy chạy khoảng 17–31 epoch khi đặt 500 không có nghĩa là cấu hình bị bỏ qua.

Đã thêm thử nghiệm các độ dài chuỗi và tham số trên những fold theo thời gian có khoảng cách 5 origin tại ranh giới. `sequence_length=10` là 10 phiên **đầu vào**, còn `sequence_length=20` là 20 phiên đầu vào; cả hai vẫn dự báo cùng horizon 5 phiên. Một thử nghiệm trước đây cho ACB cho RMSE trung bình trên fold giảm từ khoảng 0,739 xuống 0,698 với chuỗi 10, nhưng ở bài test riêng RMSE gần như không đổi và số cảnh báo sai tăng. Vì vậy không mặc định chọn 10 phiên chỉ từ kết quả fold.

### Walk-forward và mục tiêu HIGH

Quy trình thử nghiệm mới chọn cấu hình trên ba fold **trước test** theo average precision (AP) của nhãn HIGH, thay vì chỉ dùng RMSE. Nó thử mô hình hồi quy cũ và LSTM có thêm đầu HIGH với ba trọng số loss; đầu HIGH chỉ được chọn nếu AP trung bình hơn mô hình hồi quy ít nhất 0,01. Đầu HIGH xuất điểm sigmoid trong khoảng 0–1 để xếp hạng và đặt ngưỡng; điểm này **chưa được hiệu chuẩn thành xác suất thực**.

Test được chia thành các khối 60 phiên. Trước mỗi khối, LSTM được train lại từ dữ liệu quá khứ có nhãn, giữ 120 phiên gần nhất để dừng sớm và hiệu chỉnh ngưỡng; ranh giới chừa 5 origin vì target sử dụng 5 phiên tương lai. Nhãn HIGH thực tế được giữ theo ngưỡng train gốc để so sánh với kết quả cũ. Ngân sách cảnh báo trên validation của thử nghiệm là tối đa 30%; tỷ lệ trên test có thể khác. Nếu validation không có ngưỡng đạt precision tối thiểu, quy trình dùng ngưỡng theo ngân sách cảnh báo và ghi rõ cách chọn trong metadata.

Đầu HIGH riêng **chưa chứng tỏ cải thiện ổn định** trên bốn mã. Trong các lần chạy cuối, bộ chọn đều giữ LSTM hồi quy. Vì vậy `models.lstm.high_loss_weight` mặc định là `0.0`; mã nguồn vẫn cho phép thử đầu HIGH khi có dữ liệu mới.

## 3. Kết quả quan sát được

### So với các lần train rất sớm

Trong các lần trao đổi trước khi đổi feature, số đã ghi lại là ACB RMSE khoảng **0,9473**, HIGH F1 khoảng **0,0889**; FPT RMSE khoảng **1,0053**, HIGH F1 khoảng **0,2335**. Ở bản train một lần gần nhất trước thử walk-forward, ACB là RMSE **0,8534**, F1 **0,4025**; FPT là RMSE **0,7919**, F1 **0,4779**. Đây là cải thiện *giữa các lần chạy*, nhưng artifact rất sớm đã bị ghi đè và dữ liệu nguồn có thể đã thay đổi. **Không thể quy toàn bộ chênh lệch này cho riêng feature hoặc LSTM.** HPG và VIC không có mốc rất sớm tương đương để kết luận.

### So sánh có cùng ngày test và nhãn thực tế

Bản trước walk-forward đã được sao lưu ở `artifacts/baseline_before_high_walkforward_20261026/`; code so sánh kiểm tra ngày test, target và nhãn HIGH khớp với artifact mới. Các số dưới đây viết theo thứ tự **bản trước → walk-forward**:

- **ACB:** RMSE **0,8534 → 0,8969**; HIGH recall **40,5% → 32,9%**; HIGH F1 **0,4025 → 0,3421**. Walk-forward kém hơn, nên giữ bản trước nếu mục tiêu là HIGH.
- **FPT:** RMSE **0,7919 → 0,7764**; HIGH recall **36,2% → 36,9%**; HIGH F1 **0,4779 → 0,4803**. Phát hiện thêm một phiên HIGH với hai báo động sai thêm: cải thiện nhỏ, chưa phải bước nhảy lớn.
- **HPG:** RMSE **0,8219 → 0,8463**; HIGH recall **3,75% → 20,0%**; HIGH F1 **0,0706 → 0,1711**. Tìm thêm 13 phiên HIGH nhưng báo động sai tăng **2 → 91**. Đây là đánh đổi lớn giữa recall và số cảnh báo sai.
- **VIC:** RMSE **1,5332 → 1,4461**; HIGH recall **30,4% → 29,4%**; HIGH F1 **0,4436 → 0,4335**. Dự báo độ biến động tốt hơn nhưng phát hiện HIGH kém hơn nhẹ.

Không có cải thiện đồng đều trên cả bốn mã. FPT là kết quả khích lệ nhất cho cả hai mục tiêu, còn HPG chỉ cải thiện recall nếu chấp nhận nhiều cảnh báo sai. Các giai đoạn test này đã được xem trong quá trình phát triển, nên cần dữ liệu mới để xác nhận khả năng lặp lại.

## 4. Cách chạy lại mà không tải dữ liệu

Tại thư mục gốc repo, sử dụng CSV hiện có trong `data/raw/`:

```powershell
.\.venv\Scripts\python.exe -m scripts.lstm_walk_forward --ticker ACB
.\.venv\Scripts\python.exe -m scripts.lstm_walk_forward --ticker FPT
.\.venv\Scripts\python.exe -m scripts.lstm_walk_forward --ticker HPG
.\.venv\Scripts\python.exe -m scripts.lstm_walk_forward --ticker VIC
.\.venv\Scripts\python.exe -m scripts.compare_lstm_runs
```

Artifact thử nghiệm nằm ở `artifacts/lstm_high_walkforward/<TICKER>/`, còn artifact chính `artifacts/<TICKER>/` vẫn giữ nguyên. Lệnh so sánh cần các artifact baseline đã sao lưu ở máy này. Báo cáo chi tiết từng bước nằm ở `reports/lstm_high_walkforward_2026-10-26.md`. Các artifact và CSV thường bị Git ignore; khi push mã nguồn, tài liệu này giữ số liệu tổng hợp nhưng không tự đưa dữ liệu hay model đã train lên Git.
