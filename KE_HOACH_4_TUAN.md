# Kế hoạch còn lại (hạn chót 20/12/2026)

Thay thế lịch trong `AGENT_SPEC_*.md` §5 và `plan.xlsx`. Phần kiến thức (§1) của spec vẫn dùng được, trừ các điểm sửa ở mục 2 dưới đây.

---

## 1. Hiện trạng (review ngày 22/11)

| Hạng mục | Trạng thái | Đánh giá |
|---|---|---|
| T0 repo, config, pyproject, ruff | Xong | Tốt, giữ nguyên |
| T1 `geometry.py` | Xong, 9 test | Đúng công thức §1.2, đạo hàm khớp FD. Giữ |
| T2 `signal.py` | Xong, 7 test | Đúng định nghĩa SNR. Thiếu các hook lệch mô hình (xem 2.4) |
| T3 `crb.py` | Xong, 6 test | Đúng (β phức là tham số phụ nên CRB đã tự là phần bù Schur). Thiếu vài chỉ số phụ (xem 2.5) |
| T4–T12 | Mới là file rỗng 1 dòng | Toàn bộ phần thuật toán, thí nghiệm và báo cáo còn phải làm |

`pytest`: 31/31 xanh, `ruff`: sạch.

**Kết luận:** nền móng (mô hình, CRB) tốt và giữ lại toàn bộ. Khoảng 75% khối lượng còn lại nằm ở thuật toán, Monte-Carlo, hình và báo cáo, nên phải cắt phạm vi mạnh.

---

## 2. Những gì cần sửa so với spec và tài liệu khảo sát

### 2.1 Lưới B1 với μ = 0.95 quá thô theo khoảng cách (lỗi trong DECISIONS #4)

Đã đo bằng vector lái chính xác, θ = 90°, r = 10 m:

| N | Δw giữa 2 điểm kề khi μ=0.95 | "Ô" Δw khi μ=0.5 | Bước lưới quy ra mét tại 10 m |
|---|---|---|---|
| 129 | 0.031 m⁻¹ | 0.109 m⁻¹ | ≈ 3 m |
| 257 | 0.0077 m⁻¹ | 0.0275 m⁻¹ | ≈ 0.8 m |
| 513 | 0.0020 m⁻¹ | 0.0070 m⁻¹ | ≈ 0.2 m |

(w = 1/r. Ô μ=0.5 khớp con số 0.116 m⁻¹ của Cui–Dai trong tài liệu khảo sát.)

Sàn RMSE do lưới ≈ bước/√12, tức khoảng 0.9 m ở N=129, trong khi √CRB(r) ở SNR 10 dB chỉ cỡ 0.03–0.1 m. Nếu giữ μ=0.95 thì B1 thua chỉ vì lưới, và hội đồng sẽ bác so sánh đó.

**Sửa:**
- Lưới định nghĩa theo **ô** (đo bằng số qua coherence chính xác): `step_u = (2/N)/ov_u`, `step_w = cell_w(θ)/ov_w`.
- B1 dùng ov = 8. Thêm biến thể **B1-r**: B1 cộng tinh chỉnh cục bộ hàm MUSIC (Nelder–Mead hoặc parabol). Nhờ đó phần lợi của P1 nằm ở **hàm chi phí** (DML so với MUSIC), không phải ở lưới.

### 2.2 N=129 gần như không phân giải được theo khoảng cách

Ở N=129, một ô tại 10 m trải từ khoảng 4.8 m đến 10 m. Vùng 5–40 m gần như nằm gọn trong 1–2 ô theo w.

**Sửa:** thí nghiệm phân giải dùng **N = 257 làm mặc định** (theo khảo sát); N=129 dùng cho sanity check và CRB nhanh; N=513 chỉ cho độ phức tạp và một điểm kiểm tra. **Dùng T = 512 ≈ 2N** (bản đầu ghi nhầm T = 256, nhỏ hơn N = 257). Không theo đề xuất T=64 của khảo sát: T<N làm R̂ suy biến và không dùng được tiền xử lý Ĝ (chính khảo sát cũng ghi Ĝ cần T ≥ N).

### 2.3 Hiệu chỉnh dải quét trước khi chạy Monte-Carlo

CRB 2 mục tiêu (N=129, SNR 10 dB, T=256, cùng θ=90°, r₁=10 m) cho √CRB(r) ≈ 0.22 m ở Δr=0.5 m. Nghĩa là theo thống kê thì vẫn tách được dù Δr chỉ khoảng 0.1 ô. Nếu quét như spec (SNR 10–30 dB) thì mọi thuật toán tốt sẽ có P_res ≈ 1 và đường cong sẽ phẳng.

**Sửa:** trước mỗi thí nghiệm P_res, dùng CRB để tính giới hạn phân giải thống kê (SRL, δ ≥ √CRB_δ) và đặt dải quét Δ và SNR quanh vùng chuyển tiếp (gồm cả SNR −10…0 dB). Mỗi lần mất khoảng 1 phút, nhưng tránh được việc lãng phí cả ngày chạy MC.

### 2.4 `signal.py` cần thêm (nhỏ)
- `make_waveform(kind="orth")`: QPSK trực giao, `XXᴴ/T = (P/N)I` chính xác khi T ≥ N.
- `kind="streams"`: U luồng truyền thông cộng luồng dò công suất ρ, dùng cho thí nghiệm hạng R_x.
- `simulate_echo(..., amplitude=False, coupling=None, pos_err_std=0.0, mode="exact")`: các hook lệch mô hình cho thí nghiệm độ bền vững.
- Sửa nhỏ trong `steering()`: mảng numpy 0-d hiện trả về (N,1) thay vì (N,).

### 2.5 `crb.py` cần thêm (nhỏ)
- `crb_separation(...)`: CRB của δ = r₂−r₁ (hoặc u₂−u₁) bằng `eᵀCe`, dùng cho SRL.
- `crb_blockdiag(...)`: CRB khi bỏ khối chéo giữa các mục tiêu; báo cáo tỉ số đầy đủ/chéo như một chỉ số ghép nối.
- Hàm đổi sang (u, w): `C_uw = J_g C J_gᵀ`.

### 2.6 Định vị lại đóng góp (theo tài liệu khảo sát, đồng ý)

P1 không còn là "MUSIC phân cấp + DML", vì ý tưởng thô–tinh đã có ở B3 (arXiv:2411.15532) và các bài khác. **P1 mới = khởi tạo kiểu B3 → AP-DML → joint DML dùng X đã biết** (β tách đóng):

- Thống kê đủ là `Q = Y Xᴴ` (N×N). Với tham số η, ma trận Φ có các phần tử
  `ΦᴴΦ[k,l] = T·(a_kᴴa_l)·(a_lᵀ R_x a_k*)` và `Φᴴy[k] = a_kᴴ Q a_k*`.
  Hàm chi phí `f(η) = −(Φᴴy)ᴴ(ΦᴴΦ)⁻¹(Φᴴy)` chỉ tốn O(N²K) mỗi lần đánh giá, không phụ thuộc T.
- Cấu trúc khứ hồi a aᵀ cho coherence bằng μ² (ví dụ μ=0.985 thành 0.970), nên búp chính hẹp hơn MUSIC một chiều. Đây là lợi thế vật lý thật, nhưng **là giả thuyết, cần đo**.
- **Tầng khởi tạo của P1 chính là B3.** Cài một lần, dùng cho hai vai: B3 (baseline) và P1-init. So sánh B3 với P1 cô lập đúng phần đóng góp của tầng DML. Đây là thí nghiệm rẻ nhất mà vẫn chặt chẽ nhất.

### 2.7 Quy ước góc: giữ nguyên

Code dùng θ tính từ trục mảng (u = cosθ). Tài liệu khảo sát và B3 dùng θ_b tính từ pháp tuyến. Hai cách tương đương: `u = cosθ = sinθ_b`, `θ = 90° − θ_b`. Miền [30°,150°] của ta chính là [−60°,60°] của khảo sát. **Không đổi code**, chỉ ghi phép đổi này trong báo cáo và trong test tái tạo B3. B3 dùng M=512 (chẵn) nên ta dùng N=513.

---

## 3. Phạm vi: giữ / cắt

| Hạng mục | Quyết định |
|---|---|
| B0 far-field MUSIC 1D | **Giữ** (khoảng 20 dòng) |
| B1 2D-MUSIC lưới mịn + B1-r | **Giữ**, lưới ov=8 |
| B2 Capon | Giữ nếu còn thời gian (dùng chung `spectrum.py`, khoảng 30 phút) |
| B3 FFT-MUSIC | **Giữ**, đồng thời là tầng khởi tạo P1 |
| P1 = B3-init → AP-DML → joint known-X DML (L-BFGS trong (u,w)) | **Bắt buộc**, đóng góp chính |
| Tiền xử lý Ĝ = YXᴴ(XXᴴ)⁻¹ | **Giữ** (vài dòng), dùng trong thí nghiệm hạng |
| P2: từ điển (u,w) chính xác + SOMP + ℓ1-SVD + refine dùng chung DML | **Giữ** |
| M-SBL (EM, Woodbury) | **Giữ** (khoảng 40 dòng), đối chứng mạnh ở SNR thấp |
| FISTA group-LASSO | Cắt (SOMP + SBL là đủ) |
| Sub-array / Root-MUSIC, mảng ảo T9b, EDoF | Cắt (EDoF chỉ làm nếu dư 1 giờ) |
| MDL | Hàm + 1 bảng nhỏ, không có thí nghiệm riêng |
| SS/FBSS, trade-off rate–P_res, E3c, E9, E10 | Cắt, chuyển sang "hướng phát triển" |
| Học sâu, atomic norm | Chỉ viết trong tổng quan |
| `mc.py` | Bản đơn giản: SeedSequence theo từng trial, ProcessPool, cache `.npz` theo hash cấu hình. Bỏ tính năng resume |
| Số lần MC | Chạy nhanh 50; bản cuối 200–300, kèm khoảng tin cậy Wilson (không làm 500) |

---

## 4. Bộ hình cuối cùng (8 hình, theo 3 luận điểm bảo vệ)

| Hình | Nội dung | Cấu hình | Luận điểm |
|---|---|---|---|
| F1 | Phổ MUSIC 1 mục tiêu + tái tạo ca N=65 (20 m, 45°) | N=65/129 | Kiểm chứng mô hình |
| F2 | RMSE(r), RMSE(θ) theo SNR, 1 mục tiêu: B1, B1-r, B3, P1, SBL; kèm √CRB và sàn lưới | N=257 | P1 chạm CRB, on-grid dừng ở sàn |
| F3 | 3 mục tiêu cùng góc: phổ B0, B1 và ước lượng P1 | N=257 | Minh hoạ near-field |
| F4 | **P_res theo Δw (đơn vị ô), cùng θ**, 3 mức SNR, có đường SRL từ CRB | N=257 | **Luận điểm 1: ngưỡng phân giải** |
| F5 | P_res theo Δu (ô), cùng r | N=257 | Luận điểm 1 |
| F6 | RMSE 2 mục tiêu gần nhau so với CRB đầy đủ và CRB chéo khối | N=257 | Luận điểm 2 |
| F7 | Thời gian chạy theo N ∈ {129, 257, 513}: B1, B3, P1, SOMP, SBL | — | Luận điểm 3: chi phí P1 ≈ B3 + một lượng nhỏ |
| F8 | Độ bền vững (2×2 ô con): hạng R_x (beam/streams/iso, có và không Ĝ); tỉ số biên độ 0/−10/−20 dB; T ∈ {N, 2N, 4N}; lệch mô hình (Fresnel/biên độ/ghép nối) | N=257 | Luận điểm phụ |

---

## 5. Danh sách công việc (theo thứ tự thực hiện)

Làm lần lượt từ trên xuống; việc sau dựa trên việc trước. Tick `[x]` khi xong và test xanh.
Hạn chót cuối cùng: 20/12/2026.

**Thư viện lõi**
- [x] 1. T0–T3: repo, `geometry`, `signal`, `crb` (đã có từ trước)
- [x] 2. Bổ sung `geometry`/`signal`: sửa lỗi mảng 0-d, vị trí anten tuỳ chọn, dạng sóng `orth` và `streams`, hook lệch mô hình (Fresnel, biên độ, ghép nối, sai số vị trí)
- [x] 3. Bổ sung `crb`: CRB theo (u, w), CRB của độ tách (SRL), CRB chéo khối
- [x] 4. `grids.py`: `cell_u`, `cell_w(θ)` đo bằng số, lưới (u, w) theo ô, lưới cục bộ, `chunk_iter`
- [x] 5. `spectrum.py`: phổ MUSIC và Capon theo khối, cache vector lái
- [x] 6. `peaks.py`: tìm K đỉnh 1D/2D
- [x] 7. `mdl.py`
- [x] 8. `metrics.py`: Hungarian theo ô, `is_resolved`, P_res + Wilson CI, RMSE
- [x] 9. `mc.py`: song song, seed theo trial, cache `.npz`

**Baseline và kiểm chứng**
- [x] 10. B0 MUSIC trường xa 1D
- [x] 11. B1 2D-MUSIC lưới mịn (ov = 8) và B1-r (tinh chỉnh liên tục)
- [x] 12. B2 Capon
- [x] 13. F1: phổ MUSIC + tái tạo ca N=65 (20 m, 45°)
- [x] 14. Hiệu chỉnh SRL từ CRB → chốt dải quét F4/F5
- [x] 15. **Cổng quyết định baseline:** B0 không tách, B1 tách được ca cùng góc; dò P_res của MUSIC; nguyên mẫu ML known-X → đi tiếp P1
- [x] 16. F2 bản nháp (B1, B1-r so với √CRB và sàn lưới)
- [x] 17. Bản thảo Chương 2 (cơ sở lý thuyết)

**Đề xuất P1**
- [ ] 18. `estimators/b3_fft.py`: FFT góc → cụm → beamforming theo r ở biên cụm → MUSIC cục bộ; test tái tạo kịch bản K=4 của B3 (ghi chú mục tiêu 3 m nằm dưới 1.2D)
- [ ] 19. `estimators/refine_dml.py`: hàm chi phí known-X (β tách đóng, thống kê đủ Q = YXᴴ) + gradient giải tích + L-BFGS trong (u, w); test gradient khớp FD, chi phí giảm đơn điệu
- [ ] 20. `estimators/ap_dml.py`: AP-DML (tối ưu lần lượt từng mục tiêu)
- [ ] 21. `estimators/p1.py`: ghép B3-init → AP-DML → joint DML
- [ ] 22. Tiền xử lý Ĝ = YXᴴ(XXᴴ)⁻¹
- [ ] 23. **Cổng quyết định P1:** RMSE của P1 ≤ 1.5·√CRB ở vùng tiệm cận; P_res của P1 > B1-r ở ít nhất một vùng Δ. Nếu không đạt: chuyển trọng tâm sang "đạt CRB + nghiên cứu ngưỡng phân giải + giải pháp hạng bằng Ĝ", ghi kết quả thật
- [ ] 24. F2 đầy đủ (thêm B3, P1)
- [ ] 25. F3: 3 mục tiêu cùng góc (B0, B1, P1)
- [ ] 26. Bản thảo Chương 3 (mô hình, baseline, tiêu chí đánh giá)

**Đề xuất P2 (CS / SBL)**
- [ ] 27. Từ điển (u, w) chính xác với `cs_mu`, kiểm coherence bằng số
- [ ] 28. SOMP + ℓ1-SVD, tinh chỉnh dùng chung `refine_dml`; test khôi phục đúng hỗ trợ khi mục tiêu nằm trên lưới
- [ ] 29. M-SBL (EM, Woodbury); test hội tụ

**Thí nghiệm chính** (200–300 lần MC, khoảng tin cậy Wilson)
- [ ] 30. F4: P_res theo Δw (cùng góc), SNR ∈ {−30, −10, +10} dB, có đường SRL
- [ ] 31. F5: P_res theo Δu (cùng khoảng cách)
- [ ] 32. F6: RMSE 2 mục tiêu gần so với CRB đầy đủ và CRB chéo khối
- [ ] 33. F7: thời gian chạy theo N ∈ {129, 257, 513}
- [ ] 34. F8: độ bền vững (hạng R_x / Ĝ; tỉ số biên độ; T; lệch mô hình)
- [ ] 35. Code freeze: chỉ sửa lỗi, không thêm tính năng
- [ ] 36. Bản thảo Chương 4 (P1, P2, độ phức tạp)

**Đóng gói và báo cáo**
- [ ] 37. `experiments/run_all.py --quick` (< 30 phút), `results/summary.csv`, `results/figs/F*.png`
- [ ] 38. README: bảng kết quả chính, cách tái lập
- [ ] 39. Chương 5 (kết quả)
- [ ] 40. Chương 6 (kết luận, hạn chế, hướng phát triển)
- [ ] 41. Chương 1 (giới thiệu, định vị đóng góp theo mục 2.6)
- [ ] 42. Slide bảo vệ (3 luận điểm ↔ F4, F2/F6, F7)

---

## 5b. Kết quả các việc đã xong (#1–#17)

Code: `grids`, `spectrum`, `peaks`, `mdl`, `metrics`, `mc`, B0/B1/B1-r/B2, các bổ sung 2.4–2.5. `pytest`: 60/60 xanh.

| Mốc | Kết quả | File |
|---|---|---|
| Tái tạo ca N=65 (20 m, 45°) | B1-r: 19.93–19.98 m, 45.00°. **Chỉ đúng khi miền góc loại hướng búp lưới**; nếu quét toàn [30°,150°] thì B1 bắt nhầm búp lưới ở 131.5° (Δu = λ/d) | `F1_sanity.png`, `tests/test_music.py` |
| Cổng near-field (N=257, 10 m & 25 m cùng góc) | B0: P_res = 0; B1, B1-r: P_res = 1.0 ở −20 dB | `gate_baseline.json` |
| F2 bản nháp | B1 dừng ở sàn lưới (0.24 m); **B1-r ≈ 1.38·√CRB** ở vùng tiệm cận; ngưỡng ≈ −25 dB | `F2_rmse_snr.png` |
| SRL | 0.5 ô (Δw, θ=90°) cần −43 dB theo CRB; B1/B1-r thực tế cần ≈ +10 dB | `SRL_calibration.png`, `gate_baseline.json` |

**Phát hiện đưa vào báo cáo:**
1. **MUSIC mù cách CRB (X đã biết) khoảng √2.** MUSIC chỉ dùng vector lái phía thu; CRB dùng cả hai số hạng của ∂(a aᵀ). Đây là luận điểm định lượng cho P1.
2. **Tương quan nguồn bị bình phương trong echo khứ hồi.** Với R_x đẳng hướng, trị riêng tín hiệu nhỏ nhất của cặp mục tiêu nằm giữa N(1−μ)² và N(1−μ²), tuỳ pha arg(β₁β₂*c*²). Ở 0.5 ô (μ = 0.85) giá trị này chỉ ≈ 6 so với ≈ 870 của trị riêng lớn. Đó là lý do MUSIC cần SNR cao để tách dưới một ô.
3. **Khoảng cách giữa MUSIC và giới hạn thống kê rất lớn (~50 dB ở 0.5 ô).** Nguyên mẫu ML known-X (biết θ, quét cặp theo w) đạt P_res = 1.0 ở −20 dB, B1-r đạt 0 ở cùng SNR. Kết luận: **đi tiếp hướng P1**.
4. Pha β ảnh hưởng mạnh tới SRL (tới ~16 dB), nên mọi trial rút pha ngẫu nhiên.

**Ảnh hưởng tới các việc tiếp theo:** F4/F5 quét SNR ∈ {−30, −10, +10} dB, Δ ∈ {0.05…2} ô (DECISIONS #9). P1 phải tự ước lượng cả θ; con số thật sẽ kém nguyên mẫu (biết θ).

---

## 6. Cập nhật đề xuất cho `DECISIONS.md` (đã áp dụng, xem DECISIONS.md)

1. (giữ) Mô hình exact, không biến thiên biên độ; biên độ chỉ dùng trong F8.
2. (giữ) K biết trước trong mọi hình; MDL chỉ có 1 bảng.
3. (sửa) `is_resolved` tính trong (u, w): |Δû| < |Δu_sep|/2 và |Δŵ| < |Δw_sep|/2. Trục nào có độ tách bằng 0 thì dùng ngưỡng 0.5 ô.
4. (sửa) Lưới B1 theo ô với ov = 8, kèm biến thể B1-r có tinh chỉnh cục bộ. Bỏ quy tắc μ=0.95.
5. (mới) N=257, T=512, X = `orth` cho thí nghiệm phân giải. SNR tính theo phần tử, theo snapshot, sau echo (như §1.3).
6. (mới) Quy ước góc giữ theo trục mảng; khi so với B3 và khảo sát thì đổi `θ = 90° − θ_b`.
7. (mới) P1 = B3-init → AP-DML → joint DML known-X.

## 7. Rủi ro

| Rủi ro | Dự phòng |
|---|---|
| Báo cáo chưa viết chương nào | Viết chương xen kẽ với code (việc #17, #26, #36); cắt F8 xuống 2 ô con |
| P1 không thắng B1-r | Phương án dự phòng ở việc #23; trình bày kết quả thật |
| MC ở N=513 chậm | N=513 chỉ dùng cho F7 (vài trial để đo thời gian) |
| B3 có nhiều ngưỡng heuristic khó tái tạo | Cài đúng 3 bước, ghi rõ ngưỡng đã chọn; không cần khớp tuyệt đối với bảng thời gian của bài |
| Trễ tiến độ ở nhánh P2 | Cắt SOMP, giữ SBL (hoặc ngược lại); bỏ F5 |
