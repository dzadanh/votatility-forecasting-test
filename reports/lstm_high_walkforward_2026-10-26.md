# Thử nghiệm LSTM walk-forward và cảnh báo HIGH — 26/10/2026

## Dữ liệu và cách so sánh

- Chỉ đọc bốn CSV cục bộ ACB, FPT, HPG, VIC. Không tải lại dữ liệu.
- Baseline là artifacts đã có trước khi sửa, được sao lưu nguyên vẹn tại `artifacts/baseline_before_high_walkforward_20261026/`.
- Artifact mới nằm ở `artifacts/lstm_high_walkforward/<TICKER>/`; kết quả so sánh chi tiết nằm ở `artifacts/lstm_high_walkforward/comparison.json` và có thể tạo lại bằng `python -m scripts.compare_lstm_runs`.
- Script so sánh đã xác nhận ngày test và giá trị target thực tế của baseline và lần chạy mới khớp nhau cho cả bốn mã.
- HIGH thực tế luôn dùng ngưỡng lượng vị 70% từ train ban đầu. Recall = số phiên HIGH được báo / tổng phiên HIGH; precision = báo HIGH đúng / tổng lần báo; AP đo chất lượng xếp hạng trước khi chọn ngưỡng. RMSE thấp hơn là tốt hơn.

## Các bước đã thử

1. **Baseline:** LSTM hồi quy một đầu ra độ biến động, train một lần trên train, dừng sớm theo validation MSE và chọn ngưỡng HIGH trên validation.
2. **Walk-forward + đầu HIGH:** Tìm năm cấu hình bằng ba fold chỉ nằm trước test. LSTM có thể học thêm đầu HIGH với binary cross entropy có trọng số; chọn cấu hình theo AP HIGH. Từng khối test 60 phiên được dự báo bằng model fit từ quá khứ đã có nhãn, với khoảng cách năm origin ở hai ranh giới. Thử ban đầu vẫn dùng ngân sách cảnh báo validation 15%; recall giảm ở phần lớn mã: ACB 27,8%, FPT 18,1%, HPG 2,5%, VIC 14,9%.
3. **Hiệu chỉnh theo mục tiêu HIGH:** Đặt ngân sách cảnh báo validation 30% riêng cho thử nghiệm LSTM. Khi không đạt precision tối thiểu 25%, ngưỡng dự phòng dựa trên ngân sách cảnh báo thay vì ngưỡng độ biến động cũ. Đây là giới hạn trên validation, không phải giới hạn chắc chắn trên test.
4. **Chống chọn model do nhiễu:** Đầu HIGH chỉ được chọn nếu AP trung bình trước test hơn hồi quy ít nhất 0,01. HPG chỉ hơn khoảng 0,001 AP nên quay về mô hình hồi quy. Bản HPG dùng đầu HIGH trước bước này còn lưu tại `artifacts/lstm_high_walkforward_hpg_head/`; bản thử ngân sách 15% của bốn mã ở `artifacts/lstm_high_walkforward_initial_15pct/`.

## Kết quả cuối trên cùng ngày test

- **ACB — 291 ngày, 79 HIGH:** Baseline RMSE **0,8534**, AP **0,4191**, precision **40,0%**, recall **40,5%**, **32 đúng / 48 sai** trong 80 cảnh báo. Walk-forward RMSE **0,8969**, AP **0,3632**, precision **35,6%**, recall **32,9%**, **26 đúng / 47 sai** trong 73 cảnh báo. **Không cải thiện**; giữ baseline cho mục tiêu HIGH.
- **FPT — 291 ngày, 149 HIGH:** Baseline RMSE **0,7919**, AP **0,6842**, precision **70,1%**, recall **36,2%**, **54 đúng / 23 sai** trong 77 cảnh báo. Walk-forward RMSE **0,7764**, AP **0,6902**, precision **68,8%**, recall **36,9%**, **55 đúng / 25 sai** trong 80 cảnh báo. Cải thiện **nhỏ** về sai số và số HIGH phát hiện, kèm hai báo động sai thêm.
- **HPG — 670 ngày, 80 HIGH:** Baseline RMSE **0,8219**, AP **0,2247**, precision **60,0%**, recall **3,75%**, **3 đúng / 2 sai** trong 5 cảnh báo. Walk-forward RMSE **0,8463**, AP **0,1907**, precision **15,0%**, recall **20,0%**, **16 đúng / 91 sai** trong 107 cảnh báo. Phát hiện thêm 13 phiên HIGH nhưng gánh nặng cảnh báo sai tăng rất mạnh; đây là **đánh đổi**, không phải cải thiện đồng đều.
- **VIC — 260 ngày, 194 HIGH:** Baseline RMSE **1,5332**, AP **0,7919**, precision **81,9%**, recall **30,4%**, **59 đúng / 13 sai** trong 72 cảnh báo. Walk-forward RMSE **1,4461**, AP **0,7805**, precision **82,6%**, recall **29,4%**, **57 đúng / 12 sai** trong 69 cảnh báo. Sai số độ biến động giảm, nhưng khả năng phát hiện HIGH giảm nhẹ; nếu HIGH là ưu tiên chính, baseline nhỉnh hơn.

## Kết luận sử dụng

Không thay các artifact chính trong `artifacts/<TICKER>/`. FPT là mã đáng theo dõi bằng thử nghiệm walk-forward; ACB và VIC nên giữ baseline nếu tiêu chí chính là HIGH. HPG cần quyết định rõ có chấp nhận khoảng 91 báo động sai để tìm thêm 13 phiên HIGH hay không. Đầu HIGH riêng **không chứng tỏ cải thiện ổn định** trên bốn mã; vì vậy `models.lstm.high_loss_weight` mặc định vẫn là 0.0 và chỉ bật khi quá trình chọn cấu hình trước test có bằng chứng đủ mạnh.

Các giai đoạn test này đã được xem trong nhiều lần thử trước. Kết quả hiện tại là so sánh lịch sử; cần dữ liệu đến sau 26/10/2026 hoặc một giai đoạn chưa được dùng để ra quyết định để kiểm tra khả năng lặp lại. AP cuối được tính trên toàn bộ test dù model được fit lại từng khối, nên điểm số giữa các khối có thể đổi thang; đọc cùng precision, recall và số cảnh báo.
