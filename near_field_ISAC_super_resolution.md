# Siêu phân giải nhiều mục tiêu gần nhau trong near-field ISAC (XL-MIMO)

> Tài liệu này **tự đủ** (self-contained): mọi khái niệm, tính chất, công thức cần để code đều nằm ở Phần 1. Không cần đọc bài báo gốc.
> Các công thức đã được suy dẫn lại và có kiểm thử số trong Phần 4. Agent **không được tin mù** công thức nào chưa qua test.

---

## 0. Mục tiêu sản phẩm

Thư viện Python + bộ thí nghiệm để **ước lượng (r, θ) của K mục tiêu nằm gần nhau** từ tín hiệu echo của một mảng ULA cực lớn (near-field), và chứng minh:

1. Thuật toán đề xuất tách được mục tiêu gần nhau hơn (ngưỡng phân giải nhỏ hơn) so với baseline 2D-MUSIC.
2. Sai số ước lượng tiệm cận CRB.
3. Thời gian chạy thấp hơn baseline lưới mịn.

**Thuật toán so sánh:**

| Mã | Thuật toán | Vai trò |
|---|---|---|
| B0 | MUSIC far-field 1D (chỉ góc) | Đối chứng: không tách được mục tiêu cùng hướng |
| B1 | 2D-MUSIC lưới mịn (r, θ) | Baseline chính |
| B2 | Capon/MVDR near-field | Baseline phụ |
| P1 | MUSIC phân cấp (coarse-to-fine) + tinh chỉnh DML cục bộ | Đề xuất 1 |
| P2 | CS: từ điển miền cực + SOMP/group-LASSO + tinh chỉnh ngoài lưới | Đề xuất 2 |
| O1 | AP-DML (alternating projection) | Tùy chọn |

---

## 1. Knowledge base (kiến thức cần thiết)

### 1.0 Bối cảnh ISAC và giả thiết phạm vi

- **Kiến trúc:** một mảng ULA phát tín hiệu dual-function `x[t] = Σ_k p_k·c_k[t] + s[t]` (p_k: beam cho user k, c_k: ký hiệu dữ liệu, s: tín hiệu cảm biến chuyên dụng). Hiệp phương sai `R_x = Σ_k p_k p_kᴴ + R_s`. Cùng một x vừa mang dữ liệu vừa là "mẫu phát" cho sensing.
- **Vì sao tách được bài toán sensing:** BS biết X, nên phần ước lượng (r,θ) chỉ phụ thuộc X qua R_x; có thể nghiên cứu riêng thuật toán ước lượng với R_x cho trước.
- **Trade-off sensing–communication:** dồn công suất vào beam giao tiếp làm R_x hạng thấp, có hướng, bất lợi cho phân giải nhiều mục tiêu (xem 1.3). Cơ chế điều khiển trong dự án: `waveform="mix"`, `R_x = (1−ρ)·p pᴴ + ρ·(P/N)·I`.
- **Chỉ số giao tiếp tối thiểu (dùng ở E8):** user CU tại (r_c, θ_c) với kênh một chiều `h = β_c·a(r_c,θ_c)`; `SINR_c = |hᵀp|² / (hᵀ(R_x − p pᴴ)h* + σ_c²)`, `rate = log2(1+SINR_c)`. Báo cáo đường cong (rate, P_res) khi quét ρ.
- **Giả thiết phạm vi (phải nêu trong báo cáo):**
  1. LOS thuần, không clutter/đa đường từ môi trường (hướng mở).
  2. Monostatic, một mảng ULA (bistatic là hướng mở).
  3. **Băng hẹp.** Điều kiện hợp lệ cho mảng cực lớn là **B·D/c ≪ 1** (thời gian tín hiệu quét qua khẩu độ nhỏ hơn nhiều so với 1/B), *không phải* B/f_c nhỏ. Với D ≈ 0.686 m thì D/c ≈ 2.3 ns, nên B = 400 MHz cho B·D/c ≈ 0.9 (**không** hợp lệ, có hiệu ứng beam squint/wideband không gian). Mặc định coi B → 0; nếu cần số, giữ B·D/c ≤ 0.1 (B ≲ 40 MHz). Mở rộng OFDM/wideband là hướng phát triển.
  4. Vị trí mục tiêu đứng yên trong T snapshot.


### 1.1 Near-field và far-field

- Khẩu độ mảng D = (N−1)·d. Bước sóng λ = c/f_c.
- **Khoảng cách Rayleigh** d_Ray = 2D²/λ. Gần hơn d_Ray là near-field (sóng cầu), xa hơn là far-field (sóng phẳng).
- Cận dưới vùng Fresnel (bên trong thì không dùng mô hình sóng cầu thông thường): r ≳ 1.2·D.
- Ví dụ chuẩn: f_c = 28 GHz (λ ≈ 0.010714 m), N = 129, d = λ/2 → D ≈ 0.686 m, d_Ray ≈ 88 m, cận dưới ≈ 0.82 m.
- Ý nghĩa: mảng càng lớn và tần số càng cao thì d_Ray càng lớn, nhiều mục tiêu rơi vào near-field. Trong near-field, pha phụ thuộc đồng thời **góc và khoảng cách**, nên ước lượng chung được cả hai.

### 1.2 Hình học và vector lái (steering vector)

Quy ước **duy nhất** cho toàn dự án (tránh lẫn quy ước góc):

```
Mảng ULA nằm trên trục x, tâm tại gốc, N = 2Ñ+1 anten (N lẻ), chỉ số n ∈ {-Ñ,...,Ñ}
Anten n tại s_n = [n·d, 0]
Mục tiêu tại (r, θ):  p = [r·cosθ, r·sinθ],  θ ∈ (0, π) tính từ trục mảng
                      θ = 90° là hướng pháp tuyến (broadside)
Khoảng cách chính xác:  r_n = sqrt(r² + n²d² − 2·r·n·d·cosθ)
Vector lái chính xác:   a_n(r,θ) = exp(−j·2π/λ·(r_n − r))          (|a_n| = 1, ‖a‖² = N)
```

Các xấp xỉ:

```
Far-field (bậc 1):  r_n − r ≈ −n·d·cosθ
                    a_n ≈ exp(+j·2π·n·d·cosθ/λ)                      (chỉ phụ thuộc θ)

Fresnel (bậc 2):    r_n − r ≈ −n·d·cosθ + n²·d²·sin²θ/(2r)
                    a_n ≈ exp(j·(2π/λ)·(n·d·cosθ − n²·d²·sin²θ/(2r)))
```

Tính chất quan trọng:

- Pha của a có **thành phần tuyến tính theo n (mã hóa góc)** và **thành phần bậc hai theo n (mã hóa khoảng cách)**, hệ số bậc hai ∝ sin²θ/r.
- Ở θ → 0 hoặc π (endfire), sin²θ → 0 nên **khoảng cách gần như không quan sát được**. Quan sát tốt nhất ở broadside. Mặc định giới hạn θ ∈ [30°, 150°].
- Thông tin khoảng cách ∝ 1/r, nên **lấy mẫu lưới theo 1/r, không theo r**.
- Biên độ từng link là λ/(4π r_n) (khác nhau giữa các anten). Mặc định bỏ qua (|a_n| = 1); có cờ bật `amplitude=True` để kiểm tra độ bền vững.

**Đạo hàm chính xác** (dùng cho CRB và tinh chỉnh):

```
∂r_n/∂r = (r − n·d·cosθ)/r_n
∂r_n/∂θ = r·n·d·sinθ / r_n
∂a_n/∂r = −j·(2π/λ)·( (r − n·d·cosθ)/r_n − 1 ) · a_n
∂a_n/∂θ = −j·(2π/λ)·( r·n·d·sinθ / r_n ) · a_n
```

### 1.3 Mô hình echo đơn tĩnh (monostatic)

Cùng một mảng phát và thu. Kênh khứ hồi của mục tiêu k: G_k = β_k · a_k · a_kᵀ (lưu ý **aᵀ**, không phải aᴴ).

```
Y = Σ_k β_k · a_k · a_kᵀ · X + Z            Y ∈ C^{N×T},  X ∈ C^{N×T} (tín hiệu phát),  Z ~ CN(0, σ² I)
R_x = X·Xᴴ / T                              hiệp phương sai tín hiệu phát
```

Viết lại thành dạng MUSIC chuẩn:

```
y[t] = A·s[t] + z[t],   A = [a_1 ... a_K] ∈ C^{N×K},   s_k[t] = β_k · (a_kᵀ x[t])
R_s = E[s sᴴ] = B · Aᵀ · R_x · A* · Bᴴ       B = diag(β)
R_Y = A · R_s · Aᴴ + σ² I
```

**Hệ quả quan trọng nhất cho dự án:**

1. **Hạng của R_x quyết định có tách được nhiều mục tiêu hay không.**
   - Nếu R_x = w·wᴴ (một beam duy nhất, hạng 1): s_k[t] = β_k (a_kᵀ w) c[t], tức mọi nguồn **hoàn toàn tương quan** (coherent). R_s hạng 1, không gian tín hiệu chỉ có 1 chiều, MUSIC thất bại và định danh không đảm bảo.
   - Cần rank(R_x) ≥ K. Thực tế: R_x hạng đầy đủ, ví dụ R_x = (P/N)·I (đẳng hướng).
2. **Mục tiêu gần nhau thì tương quan cao ngay cả khi R_x = (P/N)·I.**
   - Với R_x = (P/N) I: R_s = (P/N) · B · (AᴴA)* · Bᴴ.
   - Hệ số tương quan nguồn: ρ_kl = |a_kᴴ a_l| / N. Gần mục tiêu → ρ → 1 → R_s gần suy biến. **Đây là rào cản phân giải cơ bản.**
3. Mô hình cho phép dùng thêm thông tin X biết trước (BS biết X): `Y·Xᴴ·(X·Xᴴ)⁻¹ ≈ Σ β_k a_k a_kᵀ` (cần T ≥ N). Phần tử (n,m) có pha −2π(r_n + r_m − 2r)/λ → **mảng ảo** (tùy chọn nâng cao, Task T9b).

**Định nghĩa SNR (dùng thống nhất toàn dự án):**

```
Chuẩn hóa P = 1 (tổng công suất phát), R_x = (P/N)·I → E|a_kᵀ x|² = P
SNR_k = |β_k|² · P / σ²        (công suất tín hiệu trên mỗi anten thu / công suất nhiễu mỗi anten)
Mặc định |β_k| = 1, pha ngẫu nhiên đều;  σ² = 10^(−SNR_dB/10)
```

### 1.4 Thuật toán MUSIC 2D

```
1. R̂ = Y·Yᴴ / T                        (cần T ≥ N để R̂ đủ hạng; khuyến nghị T ≥ 2N, hoặc diagonal loading)
2. eigh(R̂) → trị riêng giảm dần; U_s = K vector riêng đầu, U_n = N−K vector còn lại
3. P(r,θ) = 1 / (aᴴ U_n U_nᴴ a) = 1 / (N − ‖U_sᴴ a‖²)      (vì ‖a‖² = N)
4. Ước lượng = K đỉnh cục bộ lớn nhất của P(r,θ) trên lưới
```

Kỹ thuật tính hiệu quả:

- Dùng dạng `N − ‖U_sᴴ a‖²`, chi phí O(N·K) mỗi điểm lưới thay vì O(N²).
- Vector hóa: `G = U_sᴴ @ A_grid` (K×G), `P = 1/(N − sum(|G|², axis=0))`. **Xử lý theo từng khối** (chunk) để không tràn RAM (lưới 150k điểm × N=129 phức ≈ 300 MB).
- Phổ MUSIC chỉ có ý nghĩa khi đã biết K. Dùng MDL (1.5).

### 1.5 Ước lượng số mục tiêu K (MDL)

Với trị riêng λ_1 ≥ … ≥ λ_N của R̂, T snapshot, giả thiết k nguồn:

```
g_k = (∏_{i=k+1..N} λ_i)^(1/(N−k))         trung bình nhân
a_k = (1/(N−k)) · Σ_{i=k+1..N} λ_i         trung bình cộng
MDL(k) = −T·(N−k)·log(g_k / a_k) + 0.5·k·(2N−k)·log(T)
K̂ = argmin_{k=0..N−1} MDL(k)
```

Lưu ý: khi mục tiêu rất gần nhau, MDL có xu hướng đếm thiếu. Thí nghiệm chính dùng K biết trước. Thí nghiệm riêng (E7) mới dùng MDL.

### 1.6 Tính kết hợp (coherence) và thiết kế lưới

- Độ kết hợp giữa hai điểm lưới: `μ(p_i, p_j) = |a(p_i)ᴴ a(p_j)| / N ∈ [0,1]`.
- Lưới MUSIC cần **mịn**: bước sao cho μ giữa hai điểm kề ≥ ~0.95 để không bỏ sót đỉnh.
- Từ điển CS cần **thưa hơn**: μ giữa hai nguyên tử kề ≤ ngưỡng ρ₀ (mặc định 0.5–0.7) để giảm tương quan, nếu không SOMP/LASSO không ổn định.
- **Quy tắc lập lưới**:
  - Góc: đều theo u = cosθ (không đều theo θ), bước cơ bản ≈ 2/N (DFT), mịn hơn thì chia 2–4.
  - Khoảng cách: với mỗi u, bắt đầu từ 1/r_max đến 1/r_min, **đi từng bước theo 1/r** sao cho μ(r_i, r_{i+1}) = μ_target. Giải số bằng bisection (hoặc dùng độ lệch pha bậc hai Fresnel làm xấp xỉ ban đầu: bước Δ(1/r) ∝ λ/(D²·sin²θ)).
  - Hệ quả: lưới khoảng cách **dày ở gần, thưa ở xa**; số điểm giảm mạnh khi θ gần endfire.

### 1.7 Giới hạn phân giải (định tính, cần kiểm chứng bằng thí nghiệm)

- Hai mục tiêu cùng góc, khoảng cách khác nhau: độ lệch pha bậc hai giữa chúng ∝ (1/r₁ − 1/r₂)·D²·sin²θ/λ. Tách được khi đại lượng này đủ lớn.
- Do đó **ngưỡng phân giải khoảng cách Δr tăng khoảng theo r²** (≈ r²·λ/(D²·sin²θ) nhân hằng số) và tốt nhất ở broadside.
- Khi r → d_Ray trở đi, thông tin khoảng cách mất dần, mô hình dần thành far-field (chỉ phân giải được góc).
- Hai mục tiêu cùng khoảng cách, khác góc: phân giải góc ≈ vài phần của 2/N trong miền cosθ (Rayleigh của mảng).

**Công thức định lượng (quy mô Rayleigh, suy từ pha bậc hai Fresnel):**

```
Pha bậc hai tại mép mảng (n·d = D/2):   φ_edge(r) = π·D²·sin²θ / (4·λ·r)
Quy mô phân giải theo 1/r:              Δ(1/r)_R = c·λ / (D²·sin²θ),   c ∈ [1, 4]  (hằng số cần HIỆU CHUẨN bằng thí nghiệm)
Khi Δr ≪ r:                             Δr_R ≈ c·λ·r² / (D²·sin²θ)
Δr_min giảm ∝ 1/D² ∝ 1/N² (d cố định)
```

Số minh họa (c = 1, N = 129, d = λ/2, f = 28 GHz, θ = 90°; λ/D² = 0.02279 m⁻¹), dùng dạng chính xác theo 1/r:

| r₁ (m) | r₂ tối thiểu (m) | Δr_R (m) |
|---|---|---|
| 5 | 5.64 | 0.64 |
| 10 | 12.95 | 2.95 |
| 15 | 22.79 | 7.79 |

Lưu ý: từ r ≳ 15 m, Δr_R tiến tới cỡ r, tức là phân giải khoảng cách kiểu Rayleigh gần như mất. Đây là **quy mô phân giải quy ước**; thuật toán siêu phân giải ở SNR cao được kỳ vọng tách được Δr **nhỏ hơn** Δr_R. Mục đích dùng: (1) chọn dải quét Δr cho E3 theo thang chuẩn hóa Δr/Δr_R; (2) kiểm chứng H1 và H5.

### 1.8 CRB (cận Cramér-Rao) cho K mục tiêu

Tham số: γ = [r_1..r_K, θ_1..θ_K, Re β_1..Re β_K, Im β_1..Im β_K], X biết, nhiễu trắng phức.

```
M(γ) = Σ_k β_k · a_k · a_kᵀ                      (ma trận N×N; y[t] = M·x[t] + z[t])
Ḃ_i = ∂M/∂γ_i:
   ∂M/∂r_k      = β_k · (ȧ_{k,r}·a_kᵀ + a_k·ȧ_{k,r}ᵀ)
   ∂M/∂θ_k      = β_k · (ȧ_{k,θ}·a_kᵀ + a_k·ȧ_{k,θ}ᵀ)
   ∂M/∂Reβ_k    = a_k·a_kᵀ
   ∂M/∂Imβ_k    = j·a_k·a_kᵀ
FIM:  J_ij = (2T/σ²) · Re{ trace( Ḃ_iᴴ · Ḃ_j · R_x ) }
CRB = J⁻¹ (dùng pinv/solve, kiểm tra điều kiện số)
```

Sai số vị trí (m): với Jacobian Φ = [[cosθ, −r·sinθ],[sinθ, r·cosθ]] của (r,θ) → (x,y), và C_k là khối 2×2 (r_k, θ_k) của CRB:

```
PEB²_k = trace( Φ · C_k · Φᵀ )
```

Tính chất dùng làm **kiểm thử định tính**:

- CRB ∝ 1/SNR (dốc −10 dB/decade) và ∝ 1/T.
- CRB giảm khi tăng N (θ giảm nhanh hơn r).
- CRB(r) **tăng theo r**; CRB(θ) thay đổi ít theo r và tiến về giá trị far-field.
- Khi hai mục tiêu tiến lại gần: CRB tăng mạnh (FIM gần suy biến).
- Với đối xứng mảng và 1 mục tiêu, các phần tử chéo chéo giữa r và θ trong FIM gần 0 (hai tham số gần như độc lập).
- CRB tối ưu theo R_x cho 1 mục tiêu đã biết vị trí thường tập trung công suất (hạng thấp), **mâu thuẫn với nhu cầu R_x hạng đầy đủ để tách nhiều mục tiêu**. Khi chưa biết vị trí, dùng R_x đẳng hướng.

### 1.9 Cách tiếp cận nén tín hiệu (CS) cho nhiều snapshot (MMV)

```
Từ điển:  Ψ = [a(r_g, θ_g)]_{g=1..G} ∈ C^{N×G}  (lập theo 1.6, ngưỡng ρ₀)
Mô hình:  Y ≈ Ψ·S,   S ∈ C^{G×T} chỉ có K hàng khác 0 (row-sparse)
```

Hai bộ giải:

- **SOMP** (Simultaneous OMP): lặp K lần (hoặc đến khi dư < ngưỡng):
  1. Chọn nguyên tử g* = argmax_g ‖Ψ_gᴴ · R‖₂ (R là phần dư, tính theo hàng).
  2. Thêm g* vào tập hỗ trợ Ω.
  3. Giải bình phương tối thiểu: S_Ω = Ψ_Ω⁺ Y; cập nhật R = Y − Ψ_Ω S_Ω.
- **Group-LASSO bằng FISTA**: min_S ½‖Y − ΨS‖_F² + μ·Σ_g‖S_g,:‖₂.
  - Gradient: Ψᴴ(ΨS − Y). Bước 1/L với L = ‖Ψ‖₂² (norm phổ, tính bằng power iteration).
  - Prox theo hàng: s_g ← max(0, 1 − μ/(L·‖s_g‖)) · s_g.
  - μ ≈ c·σ·sqrt(T·log G) (c ≈ 1–2, khởi điểm; quét tham số).
- **ℓ1-SVD (giảm chiều)**: thay Y bằng Y_sv = Y·V_K (V_K là K vector kỳ dị phải đầu của Y), còn K cột. Giảm mạnh chi phí mà gần như không mất thông tin.
- **Tinh chỉnh ngoài lưới**: từ các nguyên tử tìm được, tối ưu liên tục (r,θ) bằng DML (1.10).

- **Off-grid kiểu nhiễu loạn bậc 1** (tích hợp trong CS, thay thế hoặc bổ sung DML): với nguyên tử thật lệch lưới nhỏ, `a(r_g+δr, θ_g+δθ) ≈ a_g + δr·ȧ_{g,r} + δθ·ȧ_{g,θ}` (đạo hàm ở 1.2). Mô hình `Y ≈ [Ψ_Ω + D_r·diag(δr) + D_θ·diag(δθ)]·S_Ω`, giải luân phiên (S_Ω bằng bình phương tối thiểu; δr, δθ bằng bình phương tối thiểu có chặn |δ| ≤ nửa ô lưới). Chỉ hợp lệ khi độ lệch nhỏ; nên kết thúc bằng 1 vòng DML.
- **SBL (Sparse Bayesian Learning) cho MMV** (tùy chọn P3): `Y = ΨS + Z`, hàng s_g ~ CN(0, γ_g·I_T), γ = siêu tham số.
  ```
  Γ = diag(γ);  Σ = Γ − Γ Ψᴴ (σ² I_N + Ψ Γ Ψᴴ)⁻¹ Ψ Γ         (dạng Woodbury: nghịch đảo ma trận N×N, KHÔNG phải G×G)
  M = Γ Ψᴴ (σ² I_N + Ψ Γ Ψᴴ)⁻¹ Y                              (hàng M_g là trung bình hậu nghiệm của s_g)
  Cập nhật EM:  γ_g ← ‖M_g‖² / T + Σ_gg                        (Σ_gg là vô hướng; không chia thêm cho T)
  Hỗ trợ = {g : γ_g > ngưỡng}  (ví dụ 1e-3·max γ);  σ² cố định hoặc ước lượng từ trị riêng nhỏ nhất
  ```
  Chi phí mỗi vòng ≈ O(N³ + N²G + NGT); ℓ1-SVD giảm T → K. Ưu điểm: không cần chọn μ như LASSO, chịu tốt coherence cao.

### 1.10 Tinh chỉnh bằng Deterministic ML (DML)

Hàm chi phí **ghép chung** tất cả K mục tiêu (xử lý được nguồn tương quan, khác với MUSIC tách rời):

```
P_A = A (AᴴA)⁻¹ Aᴴ                      chiếu lên không gian sinh bởi K vector lái
f(r_1..r_K, θ_1..θ_K) = trace( (I − P_A) · R̂ )
```

- Khởi tạo từ MUSIC hoặc CS. Tối ưu cục bộ bằng L-BFGS (scipy) với gradient giải tích (dùng đạo hàm ở 1.2) hoặc autograd (torch).
- **Alternating Projection (AP)**: tối ưu từng mục tiêu một bằng quét lưới 1D/2D, giữ các mục tiêu còn lại cố định, lặp đến hội tụ. Dùng làm khởi tạo chắc chắn (O1).
- DML không yêu cầu nguồn không tương quan, nhưng vẫn yêu cầu A đủ hạng và R_x hạng ≥ K.

### 1.11 Tìm kiếm phân cấp (coarse-to-fine) để giảm độ phức tạp

- Giai đoạn A (thô): phổ MUSIC trên lưới thưa (μ_target ≈ 0.7) hoặc dùng sub-array giữa M₀ ≈ N/4 anten (Rayleigh của sub-array nhỏ nên coi là far-field, quét góc).
  Lấy K_c = 2K+2 đỉnh làm ứng viên.
  *Chú ý: sub-array phân giải góc kém hơn nên chỉ dùng để chọn cửa sổ ứng viên, không dùng để quyết định cuối.*
- Giai đoạn B (mịn): lưới mịn (μ_target ≈ 0.95) **chỉ trong các cửa sổ** quanh ứng viên (±2 ô thô).
- Giai đoạn C: tinh chỉnh DML.
- Chi phí: toàn lưới mịn O(G_fine·N·K) → giảm xuống O(G_coarse·N·K + K_c·G_win·N·K).

- **Phương án Root-MUSIC cho giai đoạn A** (`coarse="root"`): với sub-array far-field M₀ anten, `C = U_n U_nᴴ`, đa thức `Σ_{l=-(M₀-1)}^{M₀-1} c_l z^l = 0` với `c_l = Σ_m C_{m,m+l}`; lấy các nghiệm trong vòng tròn đơn vị gần |z|=1 nhất, `cosθ = λ·angle(z)/(2π·d)` (khớp quy ước vector lái far-field ở 1.2; kiểm tra dấu bằng test).
- **Điều kiện sub-array coi là far-field:** r ≳ 2·D_sub²/λ. Với M₀ = N/4: D_sub ≈ D/4, ngưỡng ≈ 88/16 ≈ 5.5 m (theo tiêu chí Rayleigh, nên có dư; nếu mục tiêu gần hơn thì chỉ dùng chế độ `grid`).
- **Về độ phức tạp:** lưới thô vẫn có số điểm tăng theo N³ như lưới mịn (chỉ giảm hằng số). **Chỉ `subarray`/`root` (quét góc trước, rồi quét r chỉ tại các góc ứng viên) mới giảm bậc**, vì số điểm còn K_c·N_r ∝ N².

### 1.12 Đánh giá

- **Ghép cặp**: dùng thuật toán Hungarian (`scipy.optimize.linear_sum_assignment`) trên khoảng cách Euclid (x,y) giữa ước lượng và thật.
- **RMSE**: theo r (m), θ (độ), vị trí (m). Tính trên các thử nghiệm mà K̂ = K và đã ghép được; báo cáo kèm tỷ lệ thử nghiệm bị loại.
- **Tách được** (resolved) trong một lần thử khi **cả hai** đúng:
  (i) tìm đủ K đỉnh;
  (ii) sau ghép, sai số mỗi mục tiêu: |Δr| < Δr_sep/2 **và** |Δθ| < Δθ_sep/2 (với Δr_sep, Δθ_sep là khoảng cách thật giữa các mục tiêu theo từng trục; nếu một trục có khoảng cách 0 thì dùng ngưỡng cố định, ví dụ 0.5 m hoặc 0.5°).
- **P_res** = tỷ lệ lần thử tách được trên N_mc.
- **Ngưỡng phân giải** = Δ nhỏ nhất để P_res ≥ 0.9.
- **Hiệu suất so với CRB**: `eff = RMSE / sqrt(CRB)`; tốt khi ≤ ~1.5 ở SNR đủ lớn.
- **Thời gian chạy**: đo bằng `time.perf_counter`, tách riêng thời gian dựng lưới (có thể cache) và thời gian ước lượng.

### 1.13 Các bẫy đã biết (đọc trước khi code)

| Bẫy | Cách xử lý |
|---|---|
| Quy ước góc lệch giữa các tài liệu (cosθ từ trục mảng so với sinθ từ pháp tuyến) | Chỉ dùng quy ước ở 1.2; mọi hàm nhận θ **radian**, config nhập độ |
| Khoảng cách anten d ≠ λ/2 trong một số mô phỏng gốc (ví dụ 65 anten trên 0.5 m cho d ≈ 0.73λ, sinh grating lobe trong miền góc) | Mặc định d = λ/2; test tái tạo dùng `d_over_lambda` đúng bằng 0.5/64/λ |
| R_x hạng 1 làm nguồn tương quan hoàn toàn | Mặc định R_x = (P/N)·I; test hạng của R_x |
| T < N → R̂ suy biến | Assert T ≥ N hoặc diagonal loading có log cảnh báo |
| Spatial smoothing kinh điển giả định manifold dịch chuyển (Vandermonde); near-field không thỏa | Không dùng mặc định; nếu thử thì chỉ như thí nghiệm ablation |
| Xấp xỉ Fresnel sai ở r nhỏ hoặc khẩu độ lớn | Dùng mô hình chính xác (exact) khi sinh dữ liệu; Fresnel chỉ dùng trong ước lượng nếu cần tốc độ và phải đo sai lệch |
| SNR không định nghĩa thống nhất | Dùng đúng 1.3 |
| Không có công thức CRB đóng cho đa mục tiêu | Tính số theo 1.8, kiểm chứng bằng đạo hàm hữu hạn |

### 1.14 Effective Degrees of Freedom (EDoF) của một tập điểm

```
Với K điểm {p_k}: Gram chuẩn hóa  R_ij = a(p_i)ᴴ a(p_j) / N      (K×K, đường chéo = 1)
EDoF = trace(R) / ‖R‖₂ = K / λ_max(R)                           (hạng ổn định, 1 ≤ EDoF ≤ K)
```

- K điểm cùng một vector lái (far-field, cùng góc): R toàn 1, λ_max = K, **EDoF = 1** (không phân biệt được).
- Near-field, cùng góc khác khoảng cách: EDoF > 1; tăng khi Δ(1/r) vượt Δ(1/r)_R hoặc khi N tăng.
- Dùng làm chẩn đoán trong báo cáo: vẽ EDoF theo (Δr, N). Hàm `edof(cfg, points)` trong `geometry.py`; test: far-field/cùng điểm cho 1; điểm cách xa nhiều (trực giao) cho ≈ K.
- (Khái niệm EDoF của *kênh* trong tài liệu XL-MIMO là hạng hiệu dụng của ma trận kênh; ở đây dùng bản rút gọn trên tập mục tiêu.)

### 1.15 Độ phức tạp lý thuyết (để viết Chương 4)

Số điểm lưới mịn: N_θ ∝ N; bước 1/r ∝ λ/D² ∝ 1/N² nên N_r ∝ N² → **G ∝ N³**.

| Thuật toán | Chi phí chính | Ghi chú |
|---|---|---|
| Hiệp phương sai + EVD | O(N²T) + O(N³) | Dùng chung |
| B1 (lưới mịn) | O(G·N·K) = O(N⁴K) | Phổ chiếm ưu thế khi N lớn |
| P1, `coarse="grid"` | O(G_c·N·K), G_c ∝ N³ nhưng nhỏ hơn hằng số | Chỉ giảm hằng số |
| P1, `coarse∈{subarray,root}` | O(N_θ·M₀·K) + O(K_c·N_r·N·K) ∝ N³ | **Giảm bậc** |
| DML (mỗi vòng) | O(N²K) (tính R̂A) | Số vòng I nhỏ |
| SOMP | O(K·G·N·T) | ℓ1-SVD: T → K |
| FISTA | O(I·G·N·T) mỗi lần lặp tổng | Cần ‖Ψ‖₂² |
| SBL | O(I·(N³ + N²G + NGT)) | Woodbury; ℓ1-SVD giảm T |

---

## 2. Cấu trúc repo

```
nfisac/
  pyproject.toml            # numpy, scipy, matplotlib, pyyaml, pytest, tqdm; torch (tùy chọn, cho autograd)
  configs/default.yaml
  nfisac/
    config.py               # dataclass cấu hình, nạp YAML
    geometry.py             # ArrayCfg, steering (exact|fresnel|far), steering_grad, coherence, validity check
    signal.py               # make_waveform, simulate_echo, noise_var_from_snr
    crb.py                  # fim, crb, peb
    grids.py                # polar_grid_by_coherence, u_grid, chunked grid iterator
    spectrum.py             # music_spectrum (chunked), capon_spectrum
    peaks.py                # pick_peaks_2d (K đỉnh), cửa sổ ứng viên
    mdl.py                  # mdl_estimate_K
    estimators/
      base.py               # Estimator protocol, EstResult
      music_ff.py           # B0
      music2d.py            # B1
      capon.py              # B2
      music_hier.py         # P1 (coarse-to-fine + DML refine)
      refine_dml.py         # DML cost + gradient + L-BFGS
      cs.py                 # P2: SOMP, FISTA group-LASSO, l1-SVD
      ap_dml.py             # O1
    metrics.py              # matching, RMSE, is_resolved, p_res, timing
    mc.py                   # chạy Monte-Carlo, seed, song song, cache .npz
  experiments/
    e0_sanity.py ... e8_robust.py
  tests/
    test_geometry.py test_signal.py test_crb.py test_grids.py
    test_music.py test_hier.py test_cs.py test_metrics.py
  results/                  # .npz/.csv/.png (git-ignored)
  README.md
```

### API cốt lõi (shape bắt buộc)

```python
@dataclass(frozen=True)
class ArrayCfg:
    N: int                      # lẻ
    fc: float = 28e9
    d_over_lambda: float = 0.5
    c: float = 299_792_458.0
    # property: lam, d, D, rayleigh, n (np.arange(-(N//2), N//2+1)), n_half

@dataclass
class Target:
    r: float; theta: float; beta: complex      # theta radian

def steering(cfg, r, theta, mode="exact", amplitude=False) -> np.ndarray
    # r, theta: array shape (K,) hoặc scalar -> trả về (N,K) (hoặc (N,) nếu scalar), complex128

def steering_grad(cfg, r, theta) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # (a, da_dr, da_dtheta), mỗi cái (N,K)

def make_waveform(cfg, T, P=1.0, kind="iso", focus=None, rho=0.2, rng=None) -> np.ndarray
    # (N,T). kind: "iso" (R_x=(P/N)I), "beam" (hạng 1 hội tụ tại focus=(r0,theta0)),
    # "mix" (R_x = (1-rho)*beam + rho*iso)

def simulate_echo(cfg, targets, X, sigma2, rng) -> np.ndarray      # Y (N,T)

def crb(cfg, targets, Rx, T, sigma2) -> CrbResult   # .J, .C, .C_rt[k] (2x2), .peb[k]

class Estimator(Protocol):
    name: str
    def fit(self, Y: np.ndarray, K: int | None) -> EstResult
@dataclass
class EstResult: targets: list[Target]; runtime_s: float; extra: dict
```

Quy tắc code chung:

- `complex128`, vector hóa bằng NumPy; **cấm vòng for trên điểm lưới** (vòng for chỉ ở mức khối/chunk hoặc mức mục tiêu K).
- Mọi nguồn ngẫu nhiên đi qua `np.random.Generator` truyền vào; mỗi thử nghiệm có seed ghi lại.
- Type hint đầy đủ, docstring ghi rõ shape và đơn vị.
- Không hằng số "ma thuật": đặt trong config.
- Mỗi module có test; không merge khi test đỏ.

### configs/default.yaml (giá trị khởi điểm)

```yaml
array: {N: 129, fc: 28.0e9, d_over_lambda: 0.5}
signal: {T: 256, P: 1.0, waveform: iso, snr_db: 10}
targets_region: {r_min: 3.0, r_max: 60.0, theta_deg: [30, 150]}
grid:
  music_mu: 0.95          # độ kết hợp giữa điểm lưới kề (lưới mịn)
  coarse_mu: 0.7          # lưới thô (phân cấp)
  cs_mu: 0.5              # từ điển CS
  u_oversample: 2         # bước u = 2/(N*u_oversample)
mc: {n_trials: 200, seed: 0, n_jobs: -1}
```

---

## 3. Danh sách việc giao cho Agent (từng task độc lập, có tiêu chí nghiệm thu)

> Mỗi lần chỉ giao **1 task**. Task sau chỉ bắt đầu khi test của task trước xanh. Mỗi task kết thúc bằng: code + test + chạy `pytest -q` xanh + cập nhật README mục tương ứng.

### T0. Khởi tạo repo
- Tạo cấu trúc ở Phần 2, `pyproject.toml`, cấu hình lint (ruff), pytest, load YAML thành dataclass.
- **Nghiệm thu:** `pytest -q` chạy được (có test giả), `python -c "import nfisac"` thành công.

### T1. geometry.py: vector lái, đạo hàm, kết hợp
- Cài `ArrayCfg`, `steering` (3 chế độ), `steering_grad`, `coherence(a1, a2)`, `check_fresnel_validity(cfg, r, theta)` (trả về sai pha cực đại giữa exact và Fresnel).
- **Nghiệm thu (test):**
  1. `|a_n| = 1` và `‖a‖² = N` cho mọi n.
  2. Với r = 1e6 m: `steering(exact)` khớp `steering(far)` (sai số < 1e-4).
  3. `steering_grad` khớp đạo hàm hữu hạn trung tâm (sai số tương đối < 1e-5) ở ≥ 20 điểm ngẫu nhiên.
  4. Với N=129, f=28 GHz, d=λ/2: `rayleigh ≈ 87.8 ± 0.5 m`.
  5. Fresnel khớp exact trong 5% sai pha khi r ≥ 10 m, θ = 90°; sai lệch tăng khi r giảm.
  6. Vector lái đối xứng: `a(r,θ)` với n → −n tương ứng θ → π − θ.

### T2. signal.py: dạng sóng, echo, SNR
- Cài `make_waveform` (iso, beam, mix), `simulate_echo`, `noise_var_from_snr`.
- **Nghiệm thu:**
  1. Với `iso`: `X Xᴴ/T` hội tụ về `(P/N)·I` khi T lớn (sai số Frobenius tương đối < 5% với T = 20000).
  2. Với `beam`: hạng số học của `X Xᴴ` = 1.
  3. Công suất tín hiệu trung bình mỗi anten ≈ `|β|²·P` (sai số < 5%) khi chạy bằng Monte-Carlo.
  4. Công suất nhiễu mỗi anten ≈ σ².
  5. Với 2 mục tiêu và `iso`: eigenvalue của `R̂_Y` có đúng 2 trị riêng lớn hơn σ² rõ rệt ở SNR 20 dB; với `beam`: chỉ 1 trị riêng lớn.

### T3. crb.py
- Cài `fim`, `crb`, `peb` theo 1.8; đạo hàm giải tích; hỗ trợ K ≥ 1.
- **Nghiệm thu:**
  1. FIM khớp FIM tính bằng đạo hàm hữu hạn của `M(γ)` (sai số tương đối < 1e-4).
  2. J đối xứng, xác định dương (K=1, 2) ở cấu hình mặc định.
  3. CRB giảm đúng 10 dB khi SNR tăng 10 dB (±0.1 dB), giảm 3 dB khi T gấp đôi.
  4. K = 1: CRB(r) tăng khi r tăng từ 5 đến 40 m; CRB(θ) ít thay đổi.
  5. K = 2, cùng θ, Δr giảm dần: CRB tăng đơn điệu.
  6. `peb` khớp phép đổi tọa độ (kiểm tra bằng tính tay với Jacobian Φ).

### T4. grids.py, spectrum.py, peaks.py
- Lưới theo hệ số kết hợp (1.6), `chunk_iter`, `music_spectrum(U_s, grid)`, `pick_peaks_2d`.
- **Nghiệm thu:**
  1. Với lưới `music_mu = 0.95`: μ giữa hai điểm kề theo r nằm trong [0.93, 0.97].
  2. Số điểm theo r giảm khi θ tiến về 30°/150° so với 90°.
  3. Phổ tính theo chunk khớp phổ tính một lần (với lưới nhỏ) đến 1e-10.
  4. `pick_peaks_2d` tìm đúng K đỉnh trên phổ tổng hợp có K đỉnh Gauss, bỏ qua đỉnh phụ.

### T5. Baseline B1 (2D-MUSIC), MDL, metrics
- `music2d.py`, `mdl.py`, `metrics.py` (Hungarian, `is_resolved`, `p_res`, RMSE).
- **Nghiệm thu:**
  1. 1 mục tiêu (r=15 m, θ=60°), SNR 20 dB, N=129, T=256: lỗi ≤ 1 ô lưới; RMSE ≤ 1.5·√CRB (hiệu suất) với 200 lần thử.
  2. 2 mục tiêu cách xa (Δθ = 20°): P_res ≥ 0.99 ở SNR 20 dB.
  3. **Tái tạo kiểm chứng mô hình gốc (hồi quy):** N=65, khẩu độ D=0.5 m (d ≈ 0.73λ), f=28 GHz, mục tiêu (20 m, 45°): đỉnh phổ nằm cách thật ≤ 0.2 m và ≤ 1°, ở SNR cao.
  4. `mdl_estimate_K` trả đúng K = 1, 2, 3 cho mục tiêu cách xa ở SNR 20 dB (≥ 95% số lần).
  5. `is_resolved` đúng trên các ca dựng tay (đúng, thiếu đỉnh, sai vị trí).

### T6. Baseline B0 (far-field 1D) và B2 (Capon)
- **Nghiệm thu:** 2 mục tiêu **cùng góc, khác khoảng cách 10 m và 25 m**: B0 không tách được (P_res < 0.1); B1 tách được (P_res ≥ 0.9 ở SNR 20 dB). Đây là bằng chứng near-field.

### T7. Đề xuất P1: MUSIC phân cấp + DML
- `music_hier.py`, `refine_dml.py` theo 1.10–1.11. Tùy chọn: `refine ∈ {none, music_local, dml}`, `coarse ∈ {grid, subarray, root}`.
- **Nghiệm thu:**
  1. Cùng kết quả (RMSE trong 20%) so với B1 nhưng **nhanh hơn ≥ 5×** ở N=129; ghi lại bảng tốc độ cho N ∈ {65, 129, 257, 513}.
  2. DML: hàm chi phí giảm đơn điệu qua các vòng lặp; gradient giải tích khớp đạo hàm hữu hạn.
  3. Ca 2 mục tiêu gần (cùng θ = 90°, r₁ = 15 m, Δr = Δ_thr(B1) × 0.7): P1 tách tốt hơn B1 rõ rệt (ΔP_res ≥ 0.2) ở SNR 20 dB.
  4. RMSE (r, θ) sau tinh chỉnh ≤ 1.5·√CRB ở SNR ≥ 15 dB.

### T8. Đề xuất P2: CS
- `cs.py`: từ điển, SOMP, FISTA group-LASSO, ℓ1-SVD, tinh chỉnh ngoài lưới (dùng `refine_dml`, hoặc nhiễu loạn bậc 1 ở 1.9; so sánh hai cách).
- **Nghiệm thu:**
  1. Trên dữ liệu tổng hợp trong lưới (mục tiêu đặt đúng điểm lưới), SOMP khôi phục chính xác hỗ trợ ở SNR 20 dB.
  2. Từ điển với `cs_mu=0.5`: mọi cặp nguyên tử kề có μ ≤ 0.5.
  3. FISTA: hàm mục tiêu giảm đơn điệu (có tính đến momentum) và hội tụ trong ≤ 500 vòng ở ca mặc định.
  4. ℓ1-SVD cho kết quả tương đương không dùng (khác biệt nhỏ), thời gian giảm ≥ 3×.
  5. Có tinh chỉnh ngoài lưới: RMSE tiến sát P1.

### T9. Tùy chọn
- **T9a. AP-DML (O1):** lặp từng mục tiêu, hội tụ trong ≤ 10 vòng.
- **T9b. Mảng ảo (vector hóa a ⊗ a):** dùng `Y Xᴴ (X Xᴴ)⁻¹` khi T ≥ N; ước lượng bằng MUSIC trên mảng ảo. Báo cáo có tăng khả năng tách và số mục tiêu tối đa hay không.
- **T9c. Biến thể sub-array/root** cho giai đoạn A (đã có tùy chọn ở T7, đánh giá riêng; kiểm tra đúng dấu cosθ của Root-MUSIC).
- **T9d. SBL (P3):** cài theo 1.9 (Woodbury). **Nghiệm thu:** γ_g hội tụ, hỗ trợ đúng trên dữ liệu trong lưới ở SNR 10 dB; ở ca coherence cao (`cs_mu = 0.8`) SBL tách đúng nhiều hơn SOMP.
- **T9e. EDoF:** hàm `edof` và test ở 1.14.

### T10. mc.py: chạy Monte-Carlo
- Chạy N_mc thử nghiệm song song, cache kết quả `.npz` theo hash cấu hình, ghi seed, tiến độ (tqdm), tiếp tục khi bị ngắt.
- **Nghiệm thu:** chạy lại cùng cấu hình cho kết quả giống hệt (bit-exact) khi cùng seed; thay đổi n_jobs không đổi kết quả.

### T11. Thí nghiệm và hình báo cáo

| Mã | Mô tả | Biến quét | Đầu ra |
|---|---|---|---|
| E0 | Sanity: phổ MUSIC 1 mục tiêu; tái tạo ca N=65 | — | Heatmap phổ, đánh dấu vị trí thật |
| E1 | RMSE theo SNR (1 mục tiêu) so với √CRB | SNR −10…30 dB | RMSE vs SNR cho B1, P1, P2 + đường CRB |
| E2 | 3 mục tiêu cùng góc (10, 25, 40 m), N = 513 | — | Phổ B0 so với B1 so với P1 |
| E3 | **Phân giải theo khoảng cách** (cùng θ) | **Δr/Δr_R ∈ [0.1, 2]** (Δr_R theo 1.7, c = 1; chuẩn hóa để các r₀ so sánh được); r₀ ∈ {5, 10, 15} m (r₀ lớn hơn thì Δr_R ≈ r₀, ít ý nghĩa); θ ∈ {60°, 90°}; SNR ∈ {10, 20, 30} | P_res theo Δr/Δr_R; ngưỡng phân giải theo r₀ |
| E4 | Phân giải theo góc (cùng r) | Δθ ∈ [0.2°, 10°] | P_res theo Δθ |
| E5 | RMSE 2 mục tiêu gần so với CRB 2 mục tiêu | SNR | RMSE, hiệu suất |
| E6 | Độ phức tạp | N ∈ {65, 129, 257, 513} | Thời gian, số điểm lưới |
| E7 | Ước lượng K bằng MDL | SNR, Δ | Tỷ lệ đúng K |
| E3c | 2 mục tiêu gần nhau trong không gian vật lý | r₁ = 10 m, θ₁ = 89°; r₂ = r₁ + δ, θ₂ = 91°; δ ∈ [0, 3] m | P_res theo d_phys = ‖p₁ − p₂‖ |
| E8 | Độ bền vững | T ∈ {N, 2N, 4N}; sai số vị trí anten; Fresnel thay vì exact; bật biên độ; `waveform ∈ {iso, mix, beam}`; **|β₁|/|β₂| ∈ {1, 3, 10}** (mục tiêu mạnh/yếu); ghép nối anten (Toeplitz dải: `C = toeplitz([1, c₁, c₂, 0…])`, |c₁| ≈ 0.1–0.3, |c₂| ≈ 0.02–0.1; áp dụng `a_coupled = C·a`); (rate, P_res) theo ρ | P_res, RMSE, rate |
| E9 | Ngưỡng phân giải theo N | N ∈ {65, 129, 257, 513}; 2 mục tiêu cùng θ = 90°, r₀ = 10 m; quét Δr | Ngưỡng Δr (P_res ≥ 0.9) theo N; so với dự đoán ∝ 1/N² |
| E10 | Chuyển tiếp near-field → far-field | θ = 90°, r từ 5 đến 100 m (**ghi đè r_max = 100**, vượt d_Ray ≈ 88 m); 1 mục tiêu và 2 mục tiêu cùng góc | RMSE và P_res của B0 so với B1 theo r |

**Giả thuyết cần kiểm chứng (ghi nhận kể cả khi sai):**

- H1: Ngưỡng Δr tăng ≈ theo r₀² và nhỏ nhất ở θ = 90°.
- H2: `beam` (R_x hạng 1) làm mọi thuật toán thất bại khi K ≥ 2; `mix` cải thiện dần theo ρ.
- H3: P1 (có DML) phân giải tốt hơn B1; P2 tốt ở SNR thấp và T nhỏ nhưng nhạy với độ kết hợp lưới.
- H4: P1 nhanh hơn B1 một bậc độ lớn khi N lớn (bậc giảm chỉ khi dùng `subarray`/`root`, xem 1.15).
- H5: Ngưỡng Δr giảm xấp xỉ ∝ 1/N² (E9).
- H6: Với r > d_Ray, B0 và B1 tương đương; với r < d_Ray, B1 vượt trội (E10).
- H7: Mục tiêu yếu cạnh mục tiêu mạnh (|β₁|/|β₂| = 10) làm P_res giảm rõ; DML/CS chịu tốt hơn MUSIC (E8).

### T12. Đóng gói
- `README.md`: cách cài, cách chạy từng E, bảng kết quả chính, cách tái lập.
- Xuất bảng `results/summary.csv` và hình `results/figs/*.png` đặt tên theo mã E.
- **Nghiệm thu:** một lệnh `make all` (hoặc `python -m experiments.run_all --quick`) tái tạo toàn bộ hình ở chế độ nhanh (N_mc nhỏ) trong < 30 phút.

---

## 4. Quy trình làm việc với Agent

**Prompt mẫu giao một task:**

```
Bạn là kỹ sư lập trình viên phần mềm khoa học (Python). Đọc AGENT_SPEC_near_field_ISAC_super_resolution.md.
Nhiệm vụ: thực hiện Task T<k>.
Ràng buộc:
- Chỉ sửa/tạo các file liệt kê trong task; không đổi API đã có trừ khi báo lý do.
- Dùng công thức ở Phần 1 làm chuẩn. Nếu bạn thấy công thức mâu thuẫn với test, DỪNG và báo, không tự sửa im lặng.
- Viết test theo mục "Nghiệm thu" TRƯỚC, rồi mới code.
- Cuối cùng: chạy pytest -q, dán kết quả; liệt kê giả định, sai lệch, và việc còn dở.
```

**Definition of Done cho mỗi task:**

- [ ] Code có type hint, docstring (shape, đơn vị).
- [ ] Đủ test theo mục Nghiệm thu, `pytest -q` xanh.
- [ ] Không vòng lặp Python trên điểm lưới; chạy được ở N=513 không tràn RAM.
- [ ] README cập nhật; thông số/seed ghi lại.
- [ ] Báo cáo ngắn: kết quả định lượng chính, giả định, rủi ro.

**Quyết định cần chốt sớm (ghi vào `DECISIONS.md`):**

1. Mô hình sinh dữ liệu mặc định: exact, không biên độ (đúng như tài liệu); biên độ chỉ trong E8.
2. K biết trước cho E1–E6; MDL chỉ ở E7.
3. Ngưỡng `is_resolved` như 1.12.
4. Lưới tham chiếu của B1 phải mịn (μ = 0.95); nếu không, baseline bị "yếu giả tạo" và so sánh mất ý nghĩa.

**Rủi ro và phương án dự phòng:**

| Rủi ro | Dự phòng |
|---|---|
| Lưới mịn quá lớn ở N = 513 | Chunking + float32 cho bước phổ (kiểm tra sai khác); giảm miền θ, r |
| MUSIC không tách được mục tiêu gần do tương quan | Dùng DML/AP-DML; thử R_x `mix`; thử mảng ảo (T9b) |
| CS không ổn định do từ điển tương quan | Giảm `cs_mu`, tăng ℓ1-SVD, đổi sang FISTA có warm start từ SOMP |
| CRB đa mục tiêu gần suy biến | Dùng `pinv` + báo cảnh báo; so sánh chỉ khi số điều kiện hợp lý |
| Kết quả không khớp giả thuyết | Ghi nhận thật; phân tích nguyên nhân là đóng góp, không chỉnh tham số để "khớp" |

---

## 5. Lịch thực hiện theo tuần (khớp kế hoạch 16 tuần)

Mốc tuần của kế hoạch gốc: ND1 (tuần 1–3), ND2 (3–6), ND3 (5–7), ND4 (7–14), ND5 (13–16).
Spec sắp task theo **thứ tự phụ thuộc**; bảng dưới chuyển thành lịch theo tuần.

| Tuần | Nội dung kế hoạch | Việc cụ thể | Task | Mốc nghiệm thu |
|---|---|---|---|---|
| 1 | ND1 | Đọc Phần 1.1–1.4; dựng repo, test | T0 | `pytest` chạy được |
| 2 | ND1 | Đọc 1.5–1.12; viết nháp Chương 2 | T1 (geometry) | Test vector lái, đạo hàm xanh |
| 3 | ND1 → ND2 | Hoàn thiện Chương 1–2 | T2 (signal) | Test SNR, hạng R_x xanh |
| 4 | ND2 | Kiểm chứng CRB; dựng lưới và phổ | T3 (CRB), T4 (grids/spectrum) | FIM khớp đạo hàm hữu hạn |
| 5 | ND2 + ND3 | Baseline B1, MDL, metrics; bắt đầu phác thảo P1/P2 | T5 | Tái tạo ca N=65 (20 m, 45°) |
| 6 | ND2 + ND3 | Baseline B0, B2; chạy E0, E1, E2 bản baseline; viết Chương 3 | T6, T10 (đưa lên sớm) | **Hết ND2:** B0 không tách, B1 tách được ca cùng góc |
| 7 | ND3 → ND4 | **Chốt thiết kế** P1/P2 vào `DECISIONS.md` dựa trên kết quả baseline; viết Chương 4 | Bắt đầu T7 | **Hết ND3** |
| 8–9 | ND4 | MUSIC phân cấp, DML | T7 | Nhanh hơn B1 ≥ 5× |
| 10 | ND4 | Tinh chỉnh tham số, đo tốc độ theo N | T7 (hoàn tất) | **Mốc tuần 10** |
| 11–12 | ND4 | CS: SOMP, FISTA, ℓ1-SVD, tinh chỉnh ngoài lưới | T8 | RMSE sát P1 |
| 13 | ND4 + ND5 | Tùy chọn AP-DML, mảng ảo; bắt đầu E3, E4 | T9 (tùy chọn), T11 | Có đường P_res theo Δr |
| 14 | ND4 → ND5 | Chạy E5–E8 | T11 | **Hết ND4:** pipeline hoàn chỉnh |
| 15 | ND5 | Chạy đầy đủ, vẽ hình, viết Chương 5 | T11 | Đủ hình H1–H8 |
| 16 | ND5 | Đóng gói, README, Chương 6 | T12 | **Hết ND5:** `run_all --quick` < 30 phút |

**Lưu ý khi giao Agent:**

- Lịch tính theo nhịp làm việc của người. Agent có thể xong T1–T6 trong vài ngày, nên dùng bảng này như **thứ tự kiểm duyệt**: mỗi tuần chỉ duyệt các task đã đến mốc.
- **Cổng tuần 7:** nếu baseline ở tuần 6 cho thấy 2D-MUSIC đã tách tốt ở mọi ca, cần đổi hướng đề xuất (ví dụ tập trung vào độ phức tạp hoặc mục tiêu gần hơn) trước khi tốn công T7–T8.

---

## 6. Dàn ý báo cáo

```
Chương 1: Giới thiệu — bối cảnh (6G, ISAC, XL-MIMO); vấn đề near-field và siêu phân giải; mục tiêu, phạm vi, giả thiết (1.0); đóng góp; cấu trúc báo cáo
Chương 2: Cơ sở lý thuyết — hệ thống ISAC monostatic (1.0); vùng near-field và vector lái (1.1–1.2); mô hình echo và hạng R_x (1.3); CRB (1.8); EDoF (1.14)
Chương 3: Mô hình mô phỏng và thuật toán đối chứng — B0, B1, B2, MDL, tiêu chí đánh giá (1.12)
Chương 4: Thuật toán đề xuất — P1 (phân cấp + DML), P2 (CS; SBL tùy chọn), phân tích độ phức tạp (1.15)
Chương 5: Kết quả — E0 kiểm chứng mô hình; E1, E5 so với CRB; E2, E3, E3c, E4, E9, E10 phân giải; E6 độ phức tạp; E7 MDL; E8 độ bền vững và trade-off ISAC
Chương 6: Kết luận — tóm tắt; hạn chế (LOS, băng hẹp, monostatic); hướng phát triển (wideband/OFDM, bistatic, mảng phẳng, học sâu)
```
