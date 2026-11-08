# Kế hoạch đồ án: Ước lượng vị trí và định vị mục tiêu trong hệ thống near-field ISAC dùng mảng anten cực lớn (XL-MIMO)

**Hướng nghiên cứu:** Ước lượng tham số (góc và khoảng cách) cho hệ thống ISAC dựa trên mảng anten cực lớn.
**Sản phẩm dự kiến:** Hiện thực hoá thuật toán siêu phân giải (MUSIC cải tiến hoặc phương pháp dựa trên nén tín hiệu - Compressive Sensing) để phân tách nhiều mục tiêu nằm gần nhau.

---

## 1. Các bài báo liên quan đến đề tài

| File | Bài báo | Mức liên quan | Dùng vào đâu |
|---|---|---|---|
| **baibao2** | Wang, Mu, Liu, *Near-Field Integrated Sensing and Communications* | **Rất cao** | Mô hình kênh near-field cho ULA với khoảng cách đúng r_n, 2D-MUSIC (Phụ lục A), CRB góc và khoảng cách (Phụ lục B). Là **bài gốc để tái tạo và làm baseline**. |
| **baibao1** | Qu, Guo, Saeed, *NF-ISAC: Performance Analysis and Beamforming Design* | **Rất cao** | 2D-MUSIC, so sánh RMSE với CRB theo SNR, xác suất phát hiện, so sánh với far-field. Công thức CRB (41)–(43) dùng để kiểm chứng. |
| **2407_17237** | Hua, Xu, Zhang, *Near-Field ISAC with Extremely Large-Scale Antenna Array* | **Cao** | FIM và CRB **đa mục tiêu 3D** (Proposition 1), mô hình có cả biến thiên biên độ. Dùng làm cận dưới cho kịch bản nhiều mục tiêu gần nhau. |
| **chude2** | Xue et al., *The Impact of Uniform Circular Array on Near-field ISAC* | Trung bình | CRB dạng đóng của góc và khoảng cách; nhận xét FIM xấp xỉ chéo (góc và khoảng cách gần như độc lập). Có thể mở rộng sang mảng tròn. |
| **nearfieldLT** | Liu et al., *Near-Field Communications: What Will Be Different?* | Trung bình (động lực) | Hình 5(a): 512 anten, MUSIC tách 3 mục tiêu **cùng hướng, khác khoảng cách** (10/25/40 m). Mô-típ trực tiếp cho đề tài. |
| **baibao4, baibao5** | Tutorial (Lu et al.) và Survey (Liu et al.) về near-field | Nền tảng | Mô hình NUSW, EDoF, codebook miền cực, ước lượng kênh dựa trên thưa. Dùng cho Chương 2 và phần CS. |
| baibao8 | RIS-aided near-field communications | Thấp | Chỉ lấy **ý tưởng hai giai đoạn** (sub-array cho góc thô, sau đó tìm chung góc và khoảng cách) để giảm độ phức tạp tìm kiếm. |
| baibao7 | RIS-assisted NF-ISAC, PEB | Thấp | Chỉ dùng phép đổi tọa độ cực sang Descartes và PEB. |
| baibao6 | Device-free drone localization, RIS mmWave | Thấp | Far-field, định vị bằng tam giác đạc. Chỉ tham khảo phần ánh xạ góc sang vị trí. |

> Lưu ý: baibao4 và baibao5 là tài liệu dài (42 và 41 trang), chỉ mới đọc phần tóm tắt và mở đầu.

### Khoảng trống nghiên cứu

Không bài nào trong bộ tài liệu có thuật toán siêu phân giải cho nhiều mục tiêu, cũng không có CS. Các bài chỉ làm **1 mục tiêu với 2D-MUSIC thường**. Đây là đóng góp của đồ án.

Tài liệu cần tìm thêm (có trong danh mục tham khảo của các bài trên):

- Huang & Barkat 1991 (baibao2 [9]).
- Swindlehurst & Kailath 1988 (baibao1 [21]).
- Friedlander 2019 (baibao1 [40]).
- Reduced-dimension MUSIC, Zhang et al. 2018 (baibao1 [29]).
- Cui & Dai, polar-domain sparse channel (baibao1 [24]).
- Zhang, Zhang, Eldar, near-field sparse representation (2407 [15]).
- Hua, Xu, Eldar, arXiv:2308.16130 (2407 [7]): thiết kế bộ ước lượng near-field 3D.

---

## 2. Kế hoạch thực hiện (16 tuần)

### Nội dung 1: Tìm hiểu tổng quan và cơ sở lý thuyết, từ Tuần 1 đến Tuần 3

Chi tiết:

- **Tìm hiểu bối cảnh ISAC:** kiến trúc monostatic dual-function waveform (cùng tín hiệu phát vừa truyền thông vừa cảm biến), trade-off sensing–communication qua thiết kế R_x, lý do tại sao bài toán ước lượng vị trí có thể tách ra nghiên cứu độc lập (BS biết X).
- Đọc kỹ baibao2 và baibao1: mô hình kênh near-field (khoảng cách Rayleigh, vùng Fresnel), mô hình echo khứ hồi, 2D-MUSIC và CRB.
- Tìm hiểu 2407_17237 (FIM/CRB đa mục tiêu) và chude2 (CRB góc/khoảng cách dạng đóng).
- Tìm hiểu các phương pháp siêu phân giải: MUSIC, RD-MUSIC, MDL/AIC ước lượng số mục tiêu, phương pháp dựa trên thưa (OMP/SOMP, group-LASSO, SBL), tinh chỉnh ngoài lưới.
- Tìm hiểu từ điển miền cực (lấy mẫu theo 1/r) và độ kết hợp (coherence) của từ điển.
- Tìm hiểu khái niệm Effective Degrees of Freedom (EDoF) trong XL-MIMO near-field: giải thích tại sao mảng lớn cho phép phân giải khoảng cách.
- Khảo sát các nghiên cứu về phân giải nhiều mục tiêu gần nhau trong near-field.
- **Nêu rõ giả thiết băng hẹp (narrowband)** trong cơ sở lý thuyết: điều kiện hợp lệ là B·D/c ≪ 1 (không phải B/f_c nhỏ); với D ≈ 0.686 m thì cần B ≲ 40 MHz. Mở rộng OFDM/wideband là hướng phát triển.
- Hoàn thiện Chương 1: Giới thiệu đề tài (bối cảnh 6G/ISAC, XL-MIMO, phạm vi và giả thiết); Chương 2: Cơ sở lý thuyết (hệ thống ISAC monostatic, near-field, steering vector, CRB, EDoF).

### Nội dung 2: Xây dựng mô hình hệ thống và mô phỏng đối chứng, từ Tuần 3 đến Tuần 6

Chi tiết:

- Xây dựng mô hình hệ thống: trạm gốc ISAC đơn tĩnh (monostatic), ULA N anten (129–513), băng tần 28 GHz, K mục tiêu trong vùng near-field.
- Xây dựng bộ sinh tín hiệu echo bằng Python (NumPy/PyTorch): Y = Σ β_k a(r_k,θ_k) aᵀ(r_k,θ_k) X + Z.
- Cài đặt thuật toán đối chứng: 2D-MUSIC lưới mịn (baibao2/baibao1), MUSIC far-field 1 chiều, Capon/MVDR near-field.
- Cài đặt bộ tính CRB (đa mục tiêu, bằng FIM số) và các chỉ số đánh giá: RMSE, xác suất phân giải, thời gian chạy.
- Kiểm chứng bộ mô phỏng bằng cách tái tạo phổ MUSIC và đường RMSE–CRB của baibao2, baibao1.
- Hoàn thiện Chương 3: Mô hình hệ thống và phát biểu bài toán.

### Nội dung 3: Thiết kế giải pháp đề xuất, từ Tuần 5 đến Tuần 7

Chi tiết:

- Thiết kế MUSIC cải tiến hai giai đoạn: góc thô bằng sub-array, sau đó tìm chung (r,θ) trên lưới thích ứng và tinh chỉnh Newton.
- Thiết kế phương pháp CS: từ điển miền cực không đều, SOMP/group-LASSO cho nhiều snapshot, tinh chỉnh ngoài lưới.
- Thiết kế xử lý nguồn tương quan: chọn dạng sóng R_x hạng đầy đủ, diagonal loading, thử các biến thể giải tương quan.
- Thiết kế quy trình ước lượng số mục tiêu (MDL) và tiêu chí "tách được" hai mục tiêu.
- Hoàn thiện Chương 4: Giải pháp đề xuất.

### Nội dung 4: Xây dựng chương trình, từ Tuần 7 đến Tuần 14

Chi tiết:

- Xây dựng thư viện: kênh/steering, dạng sóng, các bộ ước lượng, CRB, chỉ số, kịch bản thí nghiệm (xem mục 3).
- Cài đặt và tối ưu hóa MUSIC cải tiến (dùng aᴴPₙa = N − ‖Uₛᴴa‖², vector hóa theo lưới).
- Cài đặt các thuật toán CS (SOMP, group-LASSO/FISTA, tùy chọn SBL) và bước tinh chỉnh ngoài lưới.
- Cài đặt MDL và quy trình chọn đỉnh 2D (non-maximum suppression).
- Chạy Monte-Carlo trên nhiều cấu hình SNR, số snapshot L, số anten N, khoảng cách giữa hai mục tiêu.
- Tích hợp thành một pipeline hoàn chỉnh có file cấu hình, tự động lưu kết quả và vẽ hình.

### Nội dung 5: Thực nghiệm và đánh giá, từ Tuần 13 đến Tuần 16

Chi tiết:

- Đánh giá độ chính xác: RMSE của r và θ so với CRB theo SNR và theo L.
- Đánh giá khả năng phân giải: xác suất tách được và ngưỡng phân giải tối thiểu (Δr, Δθ) theo SNR, cho ba trường hợp (cùng góc khác khoảng cách, cùng khoảng cách khác góc, cả hai gần nhau).
- Đánh giá độ phức tạp tính toán và thời gian chạy theo N và kích thước lưới.
- Đánh giá độ bền vững: lệch mô hình (xấp xỉ Fresnel, ghép nối, sai số vị trí anten), nguồn tương quan, sai số ước lượng số mục tiêu.
- Hoàn thiện Chương 5: Thực nghiệm và đánh giá; Chương 6: Kết luận và hướng phát triển.

---

## 3. Cách triển khai chi tiết sản phẩm

**Sản phẩm:** một thư viện Python kèm bộ thí nghiệm, tạo ra đường cong phân giải của các thuật toán.

### 3.1 Cấu trúc mã nguồn

```
nfisac/
  channel.py          # a(r,θ) chính xác; xấp xỉ Fresnel; tạo echo Y
  waveform.py         # R_x = (P/N)I; beam đơn (hạng 1); beam + tín hiệu cảm biến
  crb.py              # FIM/CRB đa mục tiêu bằng đạo hàm số (autograd)
  estimators/
    music2d.py        # baseline (baibao2)
    music_2stage.py   # đề xuất 1
    omp_cs.py         # từ điển miền cực + SOMP / group-LASSO (đề xuất 2)
    refine.py         # tinh chỉnh Newton ngoài lưới
    mdl.py            # ước lượng số mục tiêu
  metrics.py          # RMSE, P_res, thời gian
  experiments/        # mỗi file = một hình trong báo cáo
```

### 3.2 Mô hình

- Vị trí anten s_n = [nd, 0], n ∈ {−Ñ..Ñ}. Mục tiêu tại (r,θ): r_n = √(r² + n²d² − 2·r·n·d·cosθ); a_n = e^{−j(2π/λ)(r_n − r)} (theo baibao2).
- Echo: y[t] = Σ_k β_k a_k a_kᵀ x[t] + z[t]. Đặt s_k[t] = β_k a_kᵀ x[t] thì y[t] = A s[t] + z[t], tức là bài toán MUSIC/MMV chuẩn.
- Tham số khởi điểm: f_c = 28 GHz, N = 129 (Rayleigh ≈ 88 m với d = λ/2), mục tiêu trong khoảng 5–40 m, L ≥ 2N snapshot.

### 3.3 Thuật toán

1. **Baseline 2D-MUSIC**
   - R̂ = YYᴴ/L, phân tích riêng, ma trận tín hiệu Uₛ (K cột).
   - Phổ P(r,θ) = 1/(N − ‖Uₛᴴ a(r,θ)‖²), tìm K đỉnh.

2. **MUSIC cải tiến hai giai đoạn** (ý tưởng từ baibao8)
   - (a) Dùng sub-array nhỏ ở giữa (xấp xỉ far-field) để quét góc thô.
   - (b) Với mỗi góc ứng viên, quét khoảng cách trên lưới đều theo **1/r**, mịn dần.
   - (c) Tinh chỉnh bằng Newton/Nelder-Mead trên hàm chi phí MUSIC.
   - (d) MDL để chọn K.

3. **CS (nén tín hiệu)**
   - Từ điển Ψ (N × G) lấy mẫu đều theo sinθ/cosθ và theo 1/r, sao cho coherence giữa hai nguyên tử kề nhau ≤ ngưỡng (theo Cui & Dai).
   - Giải min ‖S‖₂,₁ s.t. ‖Y − ΨS‖_F ≤ ε bằng SOMP (nhanh) hoặc FISTA (chính xác hơn), sau đó tinh chỉnh ngoài lưới.

4. **Tùy chọn nâng cao:** mô hình khứ hồi cho phép vector hóa aaᵀ để tạo "mảng ảo" có khẩu độ hiệu dụng lớn hơn. Chỉ làm nếu còn thời gian.

### 3.4 Thí nghiệm và hình báo cáo

| Hình | Nội dung |
|---|---|
| H1 | Tái tạo phổ MUSIC của baibao2 (N=65, mục tiêu tại (20 m, 45°), ước lượng ≈ 19.95 m) |
| H2 | Tái tạo RMSE–CRB theo SNR của baibao1 (đối chứng mô phỏng) |
| H3 | 3 mục tiêu cùng hướng, khác khoảng cách (như nearfieldLT Fig. 5a) so với far-field MUSIC |
| H4 | **P_res theo khoảng cách Δr** (cùng góc) cho các thuật toán |
| H5 | P_res theo Δθ (cùng khoảng cách) |
| H6 | RMSE của hai mục tiêu gần nhau so với CRB đa mục tiêu |
| H7 | Thời gian chạy theo N |
| H8 | Độ bền vững: sai lệch mô hình, L nhỏ, nguồn tương quan |

Tiêu chí "tách được": số đỉnh đúng bằng K và mỗi sai số nằm trong nửa khoảng cách giữa hai mục tiêu.

### 3.5 Những điểm cần lưu ý

- **Quy ước góc khác nhau:** baibao2 dùng cosθ (góc tính từ trục mảng), baibao1 dùng sinθ (từ pháp tuyến). Cần chốt một quy ước từ đầu.
- **Khoảng cách anten trong baibao2:** N=65 trên khẩu độ 0.5 m cho d ≈ 0.73λ (không phải λ/2). Nếu tái tạo hình, cần dùng đúng thông số này.
- **Hạng của R_x:** nếu phát một beam duy nhất (R_x hạng 1) thì các nguồn s_k hoàn toàn tương quan và MUSIC không tách được nhiều mục tiêu. Dùng R_x hạng đầy đủ hoặc có thêm tín hiệu cảm biến chuyên dụng (như R_s trong baibao2).
- **Mục tiêu gần nhau thì tương quan cao:** hệ số tương quan giữa hai nguồn xấp xỉ |a_kᴴ a_l|/N, tiến về 1 khi chúng gần nhau. Đó là giới hạn bản chất của bài toán.
- **Spatial smoothing:** không áp dụng được nguyên dạng vì phase bậc hai trong near-field làm manifold mất tính dịch chuyển. Cần kiểm chứng thực nghiệm thay vì mặc định.
- **Hiệu chuẩn:** nếu L < N thì R̂ thiếu hạng, cần L ≥ 2N hoặc diagonal loading. Cần định nghĩa SNR thống nhất giữa các thuật toán.
- **CRB:** tính FIM số bằng autograd thay vì dùng công thức dạng đóng của từng bài.

### 3.6 Mốc nghiệm thu

- **Tuần 6:** H1–H2 khớp bài gốc.
- **Tuần 10:** MUSIC hai giai đoạn chạy được, đạt cùng RMSE với 2D-MUSIC lưới mịn nhưng nhanh hơn đáng kể.
- **Tuần 14:** CS và các thí nghiệm H3–H8 hoàn tất.
- **Tuần 16:** báo cáo và mã nguồn đóng gói.
