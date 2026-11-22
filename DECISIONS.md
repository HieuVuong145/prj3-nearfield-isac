# Các quyết định thiết kế (DECISIONS.md)

Cập nhật 22/11/2026 theo `KE_HOACH_4_TUAN.md`. Các mục (sửa)/(mới) thay thế bản trước.

1. **(giữ) Mô hình sinh dữ liệu mặc định:** sóng cầu chính xác (`mode="exact"`), $|a_n| = 1$.
   Biến thiên biên độ, Fresnel, ghép nối anten, sai số vị trí chỉ bật trong thí nghiệm độ bền vững (F8)
   qua các tham số của `simulate_echo`.

2. **(giữ) K biết trước** trong mọi hình. MDL chỉ có 1 bảng nhỏ.

3. **(sửa) Tiêu chí tách được (`is_resolved`) tính theo đơn vị ô** trong toạ độ chuẩn hoá
   $u=\cos\theta$, $w = 1/r$; ô góc $2/N$, ô khoảng cách `cell_w(90°)` (coherence 0.5, đo bằng số):
   (i) đúng K ước lượng; (ii) sau ghép Hungarian (khoảng cách Euclid theo ô), với mỗi mục tiêu k:
   $|\hat u-u_k| < \mathrm{sep}_{u,k}/2$ và $|\hat w-w_k| < \mathrm{sep}_{w,k}/2$,
   với sep là độ tách tới láng giềng gần nhất theo trục đó; trục có độ tách 0 dùng ngưỡng 0.5 ô.
   B0 (không có khoảng cách) được chấm theo trục u (`axes=("u",)`) để công bằng.

4. **(sửa) Lưới B1 theo ô với hệ số lấy mẫu dày ov = 8** trên cả hai trục (bỏ quy tắc μ = 0.95:
   ở N = 129 nó cho bước ≈ 3 m tại 10 m, sàn lưới lớn hơn √CRB hàng chục lần).
   Thêm **B1-r** = B1 + tinh chỉnh liên tục hàm MUSIC (Nelder–Mead trong (u, w)), để so sánh với P1
   cô lập hiệu ứng của *hàm chi phí* chứ không phải của *lưới*. Trong thí nghiệm, lưới được dịch
   lệch khỏi mục tiêu (mục tiêu không nằm trên điểm lưới).

5. **(mới) Cấu hình thí nghiệm phân giải:** N = 257, **T = 512** (≈ 2N; T = 256 < N không hợp lệ),
   X = `orth` (XXᴴ/T = (P/N)I chính xác), `configs/resolution.yaml`.
   SNR = SNR mỗi phần tử, mỗi snapshot của echo (spec §1.3). Báo cáo nêu kèm
   SNR tích luỹ ≈ SNR + 10·log10(N·T) (≈ +51 dB ở N = 257, T = 512).

6. **(mới) Pha β ngẫu nhiên đều, rút mới mỗi lần thử.** Trị riêng tín hiệu nhỏ nhất và CRB của cặp
   gần nhau phụ thuộc mạnh vào pha tương đối (SRL thay đổi tới ~16 dB theo pha). SRL báo cáo trung vị theo pha.

7. **(mới) Quy ước góc giữ theo trục mảng** ($\theta=90°$ là broadside). Khi so với B3/khảo sát
   (góc tính từ pháp tuyến $\theta_b$): $\theta = 90° - \theta_b$, $u = \cos\theta = \sin\theta_b$.
   B3 dùng M = 512 (chẵn) → ta dùng N = 513.

8. **(mới) P1 = khởi tạo kiểu B3 → AP-DML → joint DML dùng X đã biết** (β tách đóng; thống kê đủ
   $Q = YX^H$). Cổng quyết định baseline (việc #15) chọn **đi tiếp** hướng này: nguyên mẫu ML known-X tách được
   cặp cùng góc 0.5 ô ở −20 dB (P_res = 1.0) trong khi B1-r cần ≈ +10 dB.

9. **(mới) Dải quét F4/F5:** độ tách Δ ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2} ô;
   SNR ∈ {−30, −10, +10} dB; r₀ = 10 m; θ ∈ {90°, 60°} (F4), θ = 90° (F5).
   Căn cứ: `results/srl_calibration.csv` và `results/gate_baseline.json`.

10. **(mới) Monte-Carlo:** seed theo `SeedSequence(seed).spawn(n)` (bit-exact với mọi `n_jobs`);
    chạy nhanh 40–50 lần, bản cuối 200–300 lần kèm khoảng tin cậy Wilson 95%.
