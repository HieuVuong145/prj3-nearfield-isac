# Các quyết định thiết kế kiến trúc (DECISIONS.md)

Theo quy định tại **Phần 4 — Agent Spec**:

1. **Mô hình sinh dữ liệu mặc định:**
   - Sử dụng mô hình sóng cầu chính xác (`mode="exact"`), không biến thiên biên độ ($|a_n| = 1$).
   - Biến thiên biên độ (`amplitude=True`) chỉ được kích hoạt trong thí nghiệm kiểm tra độ bền vững E8.

2. **Số lượng mục tiêu $K$:**
   - Được coi là biết trước ($K$ known) trong các thí nghiệm chính E1–E6 và E8–E10 để cô lập hiệu năng ước lượng $(r, \theta)$.
   - Thuật toán ước lượng số nguồn MDL chỉ được đánh giá trong thí nghiệm riêng E7.

3. **Tiêu chí tách được mục tiêu (`is_resolved`):**
   - Tuân thủ nghiêm ngặt mục 1.12:
     (i) Thuật toán tìm đủ $K$ đỉnh cục bộ;
     (ii) Sau khi ghép cặp bằng thuật toán Hungarian, sai số của từng mục tiêu: $|\Delta r| < \Delta r_{\mathrm{sep}} / 2$ và $|\Delta \theta| < \Delta \theta_{\mathrm{sep}} / 2$.
     (Nếu một trục có khoảng cách bằng 0 thì dùng ngưỡng cố định $0.5\text{ m}$ hoặc $0.5^\circ$).

4. **Lưới tham chiếu của Baseline B1 (2D-MUSIC):**
   - Bắt buộc phải là lưới mịn với độ kết hợp $\mu_{\mathrm{target}} = 0.95$.
   - Không được làm thưa lưới B1 để tránh làm yếu baseline giả tạo, đảm bảo tính công bằng khi so sánh với thuật toán đề xuất P1 và P2.

