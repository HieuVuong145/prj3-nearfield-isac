# Siêu phân giải góc–khoảng cách cho nhiều mục tiêu gần nhau trong ISAC near-field với XL-MIMO: khảo sát, khoảng trống nghiên cứu và đề xuất điều chỉnh đồ án 16 tuần

Kết luận chính: nếu chỉ ghép "tìm kiếm thô rồi tinh" với DML thì P1 **không còn mới** về ý tưởng. Đã có các công trình làm thô–tinh hoặc tinh chỉnh bằng ML cho near-field: B3 (FFT → 2D-MUSIC trong vùng thu hẹp), Hua–Xu–Eldar (ML cyclic cho đa mục tiêu 3D), các bài coarse-to-fine MUSIC và NOMP.\[1\] Tuy vậy, đồ án vẫn có một đóng góp **bảo vệ được** nếu định vị đúng chỗ, gồm ba phần:
1. DML **ghép chung K mục tiêu** dùng **dạng sóng phát X đã biết** và cấu trúc hai chiều $a\,a^T$ của echo đơn tĩnh, với vector lái cầu chính xác.
2. Một nghiên cứu **xác suất phân giải P_res và ngưỡng phân giải** có hệ thống, so với CRB đa mục tiêu đầy đủ. Giới hạn phân giải thống kê cho hai nguồn near-field gần nhau đã có, nhưng cho mô hình thụ động: El Korso et al. (Signal Processing, 2012) với mảng tuyến tính, và Bao, El Korso, Ouslimani (Digital Signal Processing 48:137–147, 2016) với mảng tuyến tính không đều, tính cả "SNR tối thiểu để phân giải hai nguồn near-field gần nhau". Vì vậy khoảng trống phải được giới hạn lại: echo đơn tĩnh với X đã biết, XL-MIMO, joint DML. Trong phạm vi đó nhóm khảo sát của tôi không tìm thấy bài 2022–2026 nào.
3. Một thư viện mã nguồn mở tái tạo được B3.

## TL;DR

- **P1 nên được định vị lại thành "known-waveform joint DML + nghiên cứu ngưỡng phân giải", không phải "MUSIC phân cấp + DML".** B3 (arXiv:2411.15532) kết thúc bằng 2D-MUSIC trên lưới cố định (Δθ=0.1°, Δr=0.2 m).\[1\] B3 không có CRB, không có tinh chỉnh ML, không đo xác suất phân giải. Vì vậy P1 có khoảng trống thật, nhưng phải chứng minh bằng số liệu: RMSE chạm CRB, ngưỡng phân giải hẹp hơn MUSIC, chi phí tương đương B3.
- **Kịch bản "nguồn gần nhau" của B3 không thật sự là bài toán siêu phân giải.** Tôi tính từ công thức coherence của Cui & Dai. Với M=512, các mục tiêu (3 m, 6°), (4 m, 7°), (5 m, 8°) cách nhau khoảng 4,4 ô góc ($2/N$) và hơn 10 ô theo $1/r$. Đồ án cần tự định nghĩa độ tách theo đơn vị "ô phân giải" (Δsinθ=2/N, Δ(1/r)≈5,12λ/(D²cos²θ)) và quét xuống dưới một ô.
- **Bẫy lớn nhất của mô hình ISAC đơn tĩnh là hạng.** MUSIC trên $R_y$ cần ma trận nguồn tương đương $B A^T R_x A^* B^H$ có hạng K. Khi $R_x$ hạng thấp (beamforming truyền thông) hoặc T nhỏ, các echo trở thành coherent. Spatial smoothing chuẩn không còn đúng trong near-field vì mất tính bất biến dịch. DML, ℓ1-SVD/SBL và tiền xử lý bình phương tối thiểu theo X đã biết là các lối thoát nên đưa vào spec.

## Key Findings

1. **B3 (Wu, Wang, Zeng 2024): đã đọc toàn văn, kiểm chứng được các điểm sau.**
   - Bước 1: lấy FFT/IFFT của ma trận hiệp phương sai, $p_\theta=\mathrm{diag}(WRW^{-1})$ với S≥M điểm, rồi tạo các cụm góc bằng ngưỡng $\Gamma_\theta=\min\{p_\Theta\}-\delta_\theta$.
   - Bước 2: beamforming theo khoảng cách tại hai biên của cụm góc, $P_{low}(r)$ và $P_{up}(r)$, rồi lấy giao của hai tập khoảng cách.
   - Bước 3: chạy 2D-MUSIC trong vùng đã thu hẹp; nếu đếm không đủ K thì mở rộng vùng.
   - Thời gian chạy: 2D-MUSIC 22,07 s; RR-MUSIC 4,56 s; RD-MUSIC 4,25 s; đề xuất 3,38 s.
   - RD/RR-MUSIC thất bại khi d=λ/2 vì chúng cần d≤λ/4.\[1\]
   - Bài là letter v1. Không có CRB, không có thí nghiệm theo độ tách, không công bố mã. Nhãn trục ở Fig. 5 dường như bị hoán đổi (RMSE(°) gắn cho "distance").
2. **CRB near-field đơn mục tiêu (Wang, Xiao, Zeng, IEEE TSP 2024): đã đọc toàn văn.**
   - Với mô hình cầu chính xác, CRB góc của XL-MIMO radar **bão hòa** khi N→∞, tiến tới $\frac{1}{2\gamma L}\frac{\lambda^2 d\sin^2\theta}{8\pi^3 r^3\cos\theta}$. CRB khoảng cách là hữu hạn.\[2\]\[3\]
   - Xấp xỉ Taylor bậc hai (Fresnel) cho CRB góc **trùng với CRB far-field** và cho xu hướng CRB khoảng cách ngược lại.\[3\] Với M=1024, r=10 m, sai lệch trung bình khoảng 20 dB.\[2\]
   - Đây là căn cứ mạnh để đồ án dùng vector lái chính xác cho cả CRB và thuật toán.
3. **Từ điển miền cực (Cui & Dai): đã kiểm toàn văn arXiv v3 qua subagent.**
   - Góc lấy mẫu đều theo $\theta=\sin\vartheta$: $\theta_n=(2n-N+1)/N$.
   - Khoảng cách: $r_s=\frac{1}{s}Z_\Delta(1-\theta^2)$ với $Z_\Delta=\frac{N^2d^2}{2\beta_\Delta^2\lambda}$. Tức là lấy mẫu **đều theo 1/r**, với s=0 ứng với r=∞.
   - Ví dụ trong bài: Δ=0,5 cho β≈1,6.\[4\] Giá trị β dùng trong mô phỏng chưa xác nhận được.
   - Công thức dựa trên xấp xỉ Fresnel, nên với N=513 và r nhỏ cần kiểm lại coherence bằng số với vector lái chính xác.\[4\]
4. **Các bài đã làm ML hoặc tinh chỉnh cho near-field đa mục tiêu:**
   - Hua, Xu, Eldar (IEEE TSP 2024): 3D-ACO và 3D-CO-WGN theo tiêu chí ML, có clutter chưa biết, so với 3D-NOMP.\[5\]\[6\] Bài cũng chỉ ra rằng phải mô hình hóa biên độ thay đổi theo anten khi mục tiêu ở gần.\[6\]
   - Coprime arrays (Cheng et al., IEEE TCOM 2025): MUSIC hai pha.\[7\]
   - DCD-MUSIC / NF-SubspaceNet (Gast et al.): có mã GitHub, bền với nguồn coherent và với lệch hiệu chuẩn.\[8\]\[9\]\[10\]
   - Giới hạn phân giải thống kê cho hai nguồn near-field gần nhau đã được nghiên cứu trong mô hình thụ động: El Korso et al. (Signal Processing 92(2):547–552, 2012) và Bao, El Korso, Ouslimani (Digital Signal Processing 48:137–147, 2016), bài sau tính cả SNR tối thiểu để phân giải. Subagent không tìm thấy bài 2022–2026 nào báo cáo **xác suất hoặc ngưỡng phân giải hai mục tiêu cho joint ML/AP-ML** trên echo đơn tĩnh XL-MIMO với X đã biết. Đây là khoảng trống, nhưng còn phụ thuộc độ phủ của tìm kiếm.
5. **Spatial smoothing trong near-field.** Pannert (Applied Acoustics 72(11):873–883, 2011) chỉ ra rằng spatial smoothing cần các vector lái tới nguồn của mọi mảng con phải giống hệt nhau, và điều kiện này "không được thỏa mãn trong near-field". Hệ quả là "các nguồn bị nới rộng" (widening of sources), dù kỹ thuật vẫn dùng được trong một số giới hạn. Nguyên nhân là vector lái cầu không bất biến khi dịch mảng con. Do đó SS/FBSS chỉ là xấp xỉ, và sai số tăng khi mảng con lớn so với r.

## Details

### 1. Mô hình chuẩn hóa cho thư viện

**Hình học và quy ước góc** (giống B3 và Wang et al.):
- Mảng có N=2Ñ+1 phần tử, chỉ số δ_n∈{−Ñ,…,Ñ}, tọa độ δ_n d.\[11\]
- θ đo từ pháp tuyến (broadside).
- Khoảng cách từ mục tiêu k đến phần tử n: $r_{n,k}=\sqrt{r_k^2+\delta_n^2d^2-2\delta_n d\,r_k\sin\theta_k}$.
- Vector lái chính xác: $[a(r,\theta)]_n=e^{-j\frac{2\pi}{\lambda}(r_{n}-r)}$.\[1\]

**Echo đơn tĩnh:** $Y=\sum_k\beta_k a_k a_k^T X+Z = A\,S+Z$, với $S=BA^TX$ là ma trận K×T và $B=\mathrm{diag}(\beta)$.
- Hiệp phương sai thu: $R_y=A\,(B A^T R_x A^* B^H)\,A^H+\sigma^2 I$, trong đó $R_x=XX^H/T$.
- Hạng của phần tín hiệu ≤ min(K, rank(R_x), T). Đây là điều kiện để MUSIC/Capon hoạt động; phần suy luận này là của tôi, từ mô hình.

**Đạo hàm giải tích** (Wang et al., (26)–(27), dùng cho CRB và Newton):\[2\]
- $\partial a_n/\partial\theta=-j\frac{2\pi}{\lambda}a_n\,\partial r_n/\partial\theta$, với $\partial r_n/\partial\theta=-\delta_n d\cos\theta/\sqrt{1-2\delta_n\varepsilon\sin\theta+\delta_n^2\varepsilon^2}$ và ε=d/r.
- $\partial r_n/\partial r=(1-\delta_n\varepsilon\sin\theta)/\sqrt{\cdot}$.

**Các con số hình học cho cấu hình của đồ án** (tôi tự tính; f_c=28 GHz, λ≈10,71 mm, d=λ/2≈5,35 mm):

| N | Khẩu độ D | Rayleigh 2D²/λ | Ranh Fresnel 0,62√(D³/λ) | 1,2D (biên độ gần đều) | Ô góc Δsinθ=2/N | Ô 1/r tại θ=0 (Cui–Dai, Δ=0,5) |
|---|---|---|---|---|---|---|
| 129 | ≈0,685 m | ≈88 m | ≈3,4 m | ≈0,82 m | 0,0155 (~0,89°) | ≈0,116 m⁻¹ |
| 257 | ≈1,37 m | ≈351 m | ≈9,6 m | ≈1,64 m | 0,0078 (~0,45°) | ≈0,029 m⁻¹ |
| 513 | ≈2,74 m | ≈1403 m | ≈27 m | ≈3,3 m | 0,0039 (~0,22°) | ≈0,0073 m⁻¹ |

Ý nghĩa: với N=513 và θ=0, hai mục tiêu ở khoảng 10 m chỉ "tách được theo kiểu Rayleigh" khi cách nhau khoảng 0,8 m theo chiều sâu. Ở 20 m con số này là khoảng 3,4 m. Theo chiều sâu, độ phân giải xấu đi rất nhanh khi r tăng và khi |θ| lớn, vì có hệ số cos²θ. Đây là lý do nên chuẩn hóa độ tách theo (Δsinθ·N/2, Δ(1/r)/ô) thay vì theo mét và độ.

Tutorial của Lu, Zeng et al. phát biểu cùng ý: hai vị trí cùng hướng tách được khi $|1/r'-1/r|\ge 1/r_{hp}(\theta')$. Độ phân giải góc của beam focusing near-field tương đương far-field.\[12\]\[13\]

### 2. Khảo sát phương pháp siêu phân giải near-field

| Phương pháp | Ý tưởng / công thức lõi | Độ phức tạp chính | Ưu | Nhược / điều kiện | Giới hạn phân giải báo cáo | Mã công khai |
|---|---|---|---|---|---|---|
| 2D-MUSIC (Huang & Barkat 1991) | $P(r,\theta)=1/\|E_n^Ha(r,\theta)\|^2$ | $O(N^3+N^2T+n_\theta n_r(N-K)N)\approx O(N^4)$ (B3) | Không cần xấp xỉ, ghép cặp tự động | Cần hạng K, T≥K; sai số sàn do lưới; thất bại với nguồn coherent | Không có công thức đóng; phụ thuộc SNR và T | Dễ tự viết |
| RD-MUSIC (Zhang et al. 2018), RR-MUSIC | Tách ma trận lái theo Fresnel, đưa 2D về 1D | ~1 lần tìm 1D | Rất nhanh | Cần d≤λ/4; với d=λ/2 sinh nhiều đỉnh giả (B3 kiểm chứng)\[1\] | — | Không |
| FFT-enhanced MUSIC (B3) | FFT góc → cụm góc → beamforming theo r tại biên cụm → 2D-MUSIC cục bộ | $O(N^3+N^2T+2NS\log S+2Ln_rN^2+Ln'_\theta n'_r(N-K)N)$ | Tương thích d=λ/2, nhanh hơn 2D-MUSIC khoảng 6,5 lần (22,07 s → 3,38 s)\[1\] | Ngưỡng heuristic (δθ, δd); mục tiêu yếu dễ bị búp phụ che; vẫn on-grid | Bằng 2D-MUSIC (theo tác giả) | Không |
| Coprime 2 pha (Cheng et al.) | Ma trận hiệp phương sai hiệu dụng để tách góc/khoảng cách; MUSIC góc rồi MUSIC khoảng cách | Thấp hơn 2D-MUSIC | Định vị được nhiều mục tiêu hơn phương pháp mảng con; RMSE thấp hơn mảng dày\[7\] | Cần mảng thưa đối xứng; có lobe ma | CRB cho CA đối xứng\[7\] | Không thấy |
| Symmetry/ESPRIT near-field (Ramezani, Demir, Björnson) | Dùng tính đối xứng để tách tham số; ESPRIT tổng quát | Thấp | Không tìm 2D | Phần lớn dựa Fresnel/d≤λ/4 | — | Không |
| DML / AP-ML (Ziskind & Wax 1988; Hua–Xu–Eldar 2024) | $\min_\eta \mathrm{tr}(P_A^\perp(\eta)\hat R)$; AP tối ưu lần lượt từng nguồn | Mỗi vòng AP gồm K lần tìm 2D cục bộ | Hiệu quả tiệm cận, xử lý nguồn coherent, đạt CRB ở SNR cao | Không lồi, cần khởi tạo tốt | Hua et al.: gần CRB\[6\] | Không |
| NOMP / Newton | OMP trên từ điển + Newton liên tục + cyclic refinement | Thấp đến trung bình | Off-grid, đơn giản | Tham lam; dễ sai khi mục tiêu dưới 1 ô | — | Bản far-field có |
| CS miền cực: P-SOMP, ℓ1-SVD, group-LASSO/FISTA (Cui & Dai 2022) | $Y\approx \Phi\, C$, C hàng-thưa; Φ lấy mẫu đều theo sinθ và 1/r | Kích thước từ điển $N\cdot S$ cột | Xử lý tốt khi T nhỏ (cả 1 snapshot) và nguồn coherent | Coherence từ điển giới hạn phân giải; basis mismatch | Coherence Δ giữa hai cột kề nhau | Cui–Dai có mã (theo footnote) |
| SBL / off-grid SBL | Tiên nghiệm Gauss phân cấp, EM/MacKay; off-grid tuyến tính hóa | $O(N^2 G)$ mỗi vòng | Phân giải tốt ở SNR thấp, ít snapshot; bền với coherent | Chậm khi G lớn | — | Có bản far-field |
| Atomic norm / gridless ("Living Off the Grid", 2D ANM theo nghịch đảo khoảng cách; ANM với không gian con DCR) | Nâng mô hình cầu thành atom Toeplitz/Bessel–Vandermonde, giải SDP\[14\]\[15\] | SDP cỡ ~N² trở lên | Không lưới, có chứng chỉ đối ngẫu | Quá nặng với N=513; công trình 2025–2026 còn mới | Có đảm bảo khi tách đủ | Không |
| Wavenumber-domain + BiCNN | Biến đổi miền số sóng rồi CNN\[16\] | Rất thấp lúc suy luận | Nhanh | Cần huấn luyện; khó tổng quát | — | Không rõ |
| Học sâu: NF-SubspaceNet, DCD-MUSIC (Gast et al.) | Học ma trận hiệp phương sai thay thế để MUSIC/ESPRIT-MUSIC hoạt động | Huấn luyện tốn; suy luận nhanh | Bền với coherent, lệch hiệu chuẩn, ít snapshot | Phụ thuộc dữ liệu huấn luyện | — | **Có** (ShlezingerLab/AI-Subspace-Methods, Python) |

Nhận xét định hướng:
- Với N=129–513, chỉ có 2D-MUSIC trên lưới cục bộ, DML/AP, SOMP/ℓ1-SVD trên từ điển 1/r và SBL là khả thi trong 16 tuần.
- Atomic norm và học sâu nên được đưa vào phần tổng quan, không nên triển khai.

### 3. CRB đa mục tiêu, giới hạn phân giải, hạng R_x và nguồn coherent

**CRB đa mục tiêu cho echo đơn tĩnh với X đã biết.** Đây là công thức tôi đề xuất, theo khung FIM Gauss của Wang et al.
- Đặt $\mu(\psi)=\mathrm{vec}\big(\sum_k\beta_k a_k a_k^TX\big)$ với $\psi=[r_1,\theta_1,\ldots,r_K,\theta_K,\Re\beta,\Im\beta]$.
- $J=\frac{2}{\sigma^2}\Re\{(\partial\mu/\partial\psi)^H(\partial\mu/\partial\psi)\}$.
- CRB của (r,θ) là nghịch đảo của phần bù Schur sau khi khử β, có kích thước 2K×2K. Bài Near-Field ISAC (Wang, Liu et al.) ghi rõ CRB của M mục tiêu có cùng dạng nhưng kích thước 2M×2M.\[17\]
- Đạo hàm theo r có số hạng $\partial(a a^T)/\partial r=\dot a a^T+a\dot a^T$, tức pha nhân đôi. Mảng ảo hai chiều tương đương có khẩu độ pha gấp đôi.
- Hai quy ước tham chiếu pha ($e^{-j2\pi(r_n-r)/\lambda}$ và $e^{-j2\pi r_n/\lambda}$) cho cùng CRB khi β là tham số phức chưa biết, vì phần bù Schur hấp thụ pha chung.

**Về FIM xấp xỉ chéo khối** (bỏ các khối chéo giữa các mục tiêu):
- Chỉ đúng khi $|a_i^Ha_j|/N\to0$, tức là các mục tiêu cách nhau nhiều ô.
- Đây chính là chế độ đồ án **không** quan tâm. Vì vậy phải dùng FIM đầy đủ và báo cáo thêm "tỉ số CRB đầy đủ / CRB chéo" như một chỉ số ghép nối.

**Tính chất của CRB đơn mục tiêu nên tái tạo** (Wang et al.): CRB góc bão hòa theo N; CRB khoảng cách hữu hạn và tăng nhanh theo r; dạng Fresnel cho kết quả sai; chế độ phased-array có CRB bằng 2/M lần chế độ MIMO.\[2\]

**Giới hạn phân giải** — nên dùng song song ba định nghĩa:
1. Kiểu Rayleigh/coherence: ô góc 2/N và ô 1/r theo Cui–Dai/Lu et al. Có thể tính bằng $\max|a^H(\eta_1)a(\eta_2)|/N\le0{,}5$ với vector chính xác.
2. Giới hạn phân giải thống kê theo CRB (Smith 2005; cho near-field: El Korso, Boyer, Renaux, Marcos, "Conditional and unconditional Cramér–Rao bounds for near-field source localization", IEEE TSP 58(5):2901–2907, 2010, và El Korso et al., "Statistical analysis of achievable resolution limit in the near field source localization context", Signal Processing 92(2):547–552, 2012, bài này suy ra SRL để phân giải hai nguồn near-field đã biết, gần nhau, trên mảng tuyến tính): độ tách δ phân giải được nếu $\delta\ge\sqrt{\mathrm{CRB}_\delta(\delta)}$.
3. Thực nghiệm: P_res(δ) và ngưỡng δ₅₀/δ₉₀.

EDoF (Wang et al. 2024, TWC) cho thấy EDoF chủ yếu do khẩu độ quyết định, không phải số anten.\[18\] Điều này hữu ích khi so N=129/257/513 ở cùng d.

**Hạng R_x và nguồn coherent trong ISAC dual-function:**
- Từ $R_y$ ở mục 1: nếu dạng sóng ISAC chỉ có U luồng truyền thông với U<K, hoặc nếu $A^TR_xA^*$ suy biến, thì nguồn tương đương coherent và MUSIC/Capon hỏng.
- Bài MIMO ISAC với mục tiêu mở rộng (arXiv:2205.14050) chỉ ra: ma trận hiệp phương sai tối ưu cho truyền thông có thể hạng thấp và khi đó CRB→∞. Ngược lại, hiệp phương sai tối ưu cho cảm biến là hạng đầy đủ.\[19\]
- Với hai mục tiêu điểm, hiệp phương sai tối ưu theo cảm biến chỉ cần hạng 2 (arXiv:2510.08793, trích Corollary 2 của tài liệu gốc).\[20\]
- Hua, Xu, Han ("Optimal Transmit Beamforming for Integrated Sensing and Communication", arXiv:2104.11871, IEEE TVT 2023) chứng minh chặt chẽ rằng thêm tín hiệu radar chuyên dụng nhìn chung có lợi, vì khai thác được toàn bộ bậc tự do không gian. Đây là cơ sở cho việc thêm luồng dò để R_x đủ hạng.

Cách xử lý đề xuất cho thư viện (theo thứ tự ưu tiên):
- (i) Tiền xử lý LS theo X đã biết, $\hat G=YX^H(XX^H)^{-1}=\sum_k\beta_ka_ka_k^T+\tilde Z$ (cần rank(R_x)=N và T≥N). Sau đó làm MUSIC/DML trên $\hat G$, có hạng K bất kể β.
- (ii) DML/AP: bản chất không cần hạng.
- (iii) ℓ1-SVD/SBL: bền với coherent.
- (iv) SS/FBSS: chỉ xấp xỉ. Nên đưa vào như một thí nghiệm cho thấy sai số tăng theo kích thước mảng con. Pannert (Applied Acoustics 72(11):873–883, 2011) cho thấy trong near-field, vector lái tới nguồn của các mảng con không còn giống nhau, nên spatial smoothing làm nguồn bị nới rộng và chỉ dùng được trong một số giới hạn. Phiên bản đối xứng tâm (forward–backward trên mảng đối xứng) giữ được nhiều cấu trúc hơn, nhưng vẫn không khôi phục chính xác bất biến dịch.

### 4. Lưới/từ điển miền cực và off-grid; MUSIC so với CS/SBL

**Lưới:**
- Đều theo $u=\sin\theta$ với bước 2/(N·ρ_u), trong đó ρ_u là hệ số oversampling.
- Đều theo $w=1/r$ với bước rút từ ô coherence: $\Delta w=2\lambda\beta_\Delta^2/(N^2d^2(1-u^2))$ theo Cui–Dai.
- Lưới (u, w) làm hàm mục tiêu gần "đẳng hướng". Tinh chỉnh Newton trong (u, w) ổn định hơn trong (θ, r).
- Từ điển P-SOMP của Cui–Dai dựa trên Fresnel. Với N=513 và r<~27 m (dưới ranh Fresnel đã tính), cần dựng cột bằng vector chính xác và kiểm $\mu=\max_{i\ne j}|\phi_i^H\phi_j|$ bằng số.
- Các thiết kế mới hơn tối ưu β theo coherence, hoặc theo độ sâu búp (beam depth) và "effective beam-focusing Fraunhofer distance" để giới hạn r_max.\[21\]\[22\] Các thiết kế này đáng tham khảo khi chọn S, số vòng khoảng cách.

**Off-grid:** có ba lựa chọn, theo thứ tự đơn giản dần:
- Newton/Gauss–Newton cục bộ trên hàm DML. Khuyến nghị dùng chung cho P1 và P2.
- Tuyến tính hóa bậc 1 (off-grid SBL).
- Lưới thích nghi coarse-to-fine.

**MUSIC so với CS/SBL ở SNR thấp và ít snapshot:**
- MUSIC cần T≥K, cần nguồn không coherent, và có ngưỡng SNR rõ rệt khi không gian con bị "hoán đổi".
- CS/SBL làm việc được với T=1 và nguồn coherent, nhưng bị giới hạn bởi coherence từ điển và basis mismatch.
- Các tài liệu đọc được không có so sánh trực tiếp MUSIC–SBL cho near-field ULA ở cấu hình của đồ án. Vì vậy đây nên là một thí nghiệm của chính đồ án, và là thí nghiệm có giá trị.
- Tôi kỳ vọng (cần kiểm chứng): SBL/ℓ1-SVD thắng ở T nhỏ hoặc khi có coherent; MUSIC+DML thắng ở T lớn, SNR trung bình–cao; DML có ngưỡng phân giải thấp nhất khi được khởi tạo tốt.

### 5. Khoảng trống nghiên cứu và độ mới của P1

**Những gì đã có:**
- B3: thô bằng FFT/beamforming, tinh bằng 2D-MUSIC trên lưới.\[1\]
- Coarse-to-fine MUSIC dùng beam squint wideband (arXiv:2509.14850): f_c=60 GHz, N_t=256, TTD.\[23\]\[24\] Bài này cũng dùng spatial smoothing và hợp nhất phổ đa sóng mang.\[25\]
- Mixed near/far-field XL-MIMO (arXiv:2503.04681): 2D-DFT thô, rồi tìm cục bộ trong ±1/N.\[26\] Bài này cảnh báo đỉnh của mục tiêu yếu có thể bị búp phụ của mục tiêu mạnh che mất.\[26\]
- Hua–Xu–Eldar: ML cyclic đa mục tiêu 3D có clutter.\[6\]\[27\]
- NOMP/Newton cho kênh near-field.\[28\]
- Một bài wideband OFDM ISAC dùng MUSIC rồi tinh chỉnh ML (chỉ thấy qua subagent, chưa xác định ID).\[29\]

**Những gì chưa thấy** (theo hai vòng tìm kiếm; độ phủ có giới hạn):
1. Joint DML trên **mô hình echo đơn tĩnh với X đã biết**, $\min_{\eta}\|Y-\sum_k\beta_k a_ka_k^TX\|_F^2$ với β tách đóng bằng LS, khởi tạo từ B3/MUSIC cục bộ, dùng vector lái chính xác.
2. Đường cong **P_res theo độ tách chuẩn hóa** (cùng góc / cùng khoảng cách / chéo), theo SNR, T, rank(R_x) và tỉ số biên độ, cho echo đơn tĩnh XL-MIMO với d=λ/2 và X đã biết, kèm CRB đa mục tiêu đầy đủ và giới hạn phân giải thống kê. Lưu ý: SRL và SNR tối thiểu để phân giải hai nguồn near-field đã được Bao, El Korso, Ouslimani (Digital Signal Processing 48:137–147, 2016) đưa ra cho mảng tuyến tính không đều trong mô hình thụ động, nên khoảng trống chỉ nằm ở bối cảnh đơn tĩnh/X đã biết/joint DML.
3. So sánh trực tiếp B3 với DML và với SBL trong chế độ dưới một ô.

**Cách phát biểu đóng góp bảo vệ được:**
- "Chúng tôi chỉ ra rằng tầng 2D-MUSIC on-grid của B3 tạo sàn sai số và ngưỡng phân giải xa CRB."
- "Thay bằng tinh chỉnh DML ghép chung, có khai thác X và cấu trúc $aa^T$, giúp RMSE chạm CRB và hạ ngưỡng phân giải xuống X ô, với chi phí tăng Y%."
- Các con số X, Y phải lấy từ thí nghiệm.
- Không nên tuyên bố "lần đầu đề xuất coarse-to-fine + ML".

### 6. Bẫy triển khai thực tế và tham số mô phỏng chuẩn

**Bẫy:**
- **Quy ước góc.** B3 và Wang dùng $-2\delta d r\sin\theta$ (θ từ pháp tuyến). Cui–Dai dùng biến θ=sinϑ. Một số bài đo góc từ trục mảng (cos). Hãy cố định một quy ước, viết unit test so với công thức gốc của từng bài khi tái tạo.
- **Tham chiếu pha.** Dùng $r_n-r$ (B3) hay $r_n$ (Wang) đều được nếu β là tham số phức tự do. Phải nhất quán giữa CRB và thuật toán.
- **d≠λ/2.** RD/RR-MUSIC cần d≤λ/4.\[1\] Khi d>λ/2 có lobe ma theo góc. Mảng thưa/coprime cần MUSIC hai pha để loại góc giả.\[30\]
- **Vùng gần.** B3 đặt mục tiêu ở 3 m với M=512, nhỏ hơn 1,2D≈3,3 m mà tôi tính được.\[1\] Ở đó biên độ thay đổi theo anten không bỏ qua được. Hua–Xu–Eldar nhấn mạnh phải mô hình hóa biên độ.\[6\] Đồ án nên giới hạn r≥1,2D hoặc thêm tùy chọn biên độ 1/r_n.
- **T<N.** B3 dùng J=200<M=512.\[1\] $\hat R$ suy biến, nên Capon/MVDR cần diagonal loading. MUSIC vẫn chạy nếu T≥K nhưng ước lượng không gian nhiễu kém. Tiền xử lý LS theo X cần T≥N.
- **Định nghĩa SNR.** Phải ghi rõ đó là SNR mỗi anten mỗi snapshot của echo ($|\beta|^2\|a^TX\|^2/(T\sigma^2)$), hay SNR "phát" kiểu Wang ($\gamma=P|\kappa|^2/\sigma^2$, với L=BT_p), hay SNR sau tích lũy mảng. Khác biệt có thể lên tới 10log₁₀(NT) dB.
- **Số hóa lưới.** RMSE của bất kỳ phương pháp on-grid nào có sàn khoảng bước/√12. Với B3, Δθ=0,1° và Δr=0,2 m, nên sàn RMSE khoảng 0,029° và 0,058 m. Phải luôn vẽ sàn này cùng CRB.
- **Ghép nối anten / lệch hiệu chuẩn.** Mô hình hóa $a\to Ca$ với C Toeplitz dải, hoặc thêm sai pha ngẫu nhiên, để kiểm độ bền. Gast et al. và arXiv:2512.21480 cho thấy đây là nguồn suy giảm đáng kể.\[9\]\[31\]
- **Biên độ khác nhau.** Ngưỡng $\Gamma_\theta=\min\{p_\Theta\}-\delta_\theta$ của B3 và tầng DFT thô dễ bỏ sót mục tiêu yếu. Cần thí nghiệm với tỉ số công suất 0/−10/−20 dB.
- **Ghép cặp và chấm điểm.** Dùng thuật toán Hungarian theo khoảng cách chuẩn hóa (u, w) trước khi tính RMSE. Quy ước rõ trường hợp phát hiện thiếu (đưa vào P_res, không đưa vào RMSE).

**Tham số để tái tạo:**

| Nguồn | Cấu hình | Mức xác minh |
|---|---|---|
| B3 (2411.15532) | M=512, d=λ/2, f=30 GHz; K=4 tại (3 m, 6°), (4 m, 7°), (5 m, 8°), (32 m, 20°); r∈[0,40] m, θ∈[−20°,40°]; Δθ=0,1°, Δr=0,2 m (n_θ=600, n_r=200); J=200; S=1024; SNR 5–25 dB\[1\] | Toàn văn |
| Wang, Xiao, Zeng CRB (2303.05736) | f=2,37 GHz, d=0,0628 m, γ=0 dB; r=10 m, θ=π/6; quét M; ví dụ M=1024; bistatic: N=8, r=18 m, R=35 m, 500 lần Monte-Carlo cho Capon | Toàn văn\[2\] |
| Beam-squint coarse-to-fine (2509.14850) | f_c=60 GHz, ULA N_t=256, TTD\[24\] | Chỉ đoạn trích |
| Cui & Dai (2108.07581) | θ_n=(2n−N+1)/N; r_s=Z_Δ(1−θ²)/s; ví dụ Δ=0,5 → β≈1,6\[4\] | Toàn văn v3 (qua subagent) |

**Bảng mức độ xác minh các tài liệu cốt lõi** (TV = đã đọc toàn văn; TT = chỉ tóm tắt/đoạn trích; SA = subagent đọc; KĐ = kinh điển, không kiểm lại trong phiên này):
- Wu, Wang, Zeng, "FFT-Enhanced Low-Complexity Near-Field Super-Resolution Sensing", arXiv:2411.15532 (2024), https://arxiv.org/abs/2411.15532 — TV.\[1\]
- Wang, Xiao, Zeng, "Cramér-Rao Bounds for Near-Field Sensing with Extremely Large-Scale MIMO", IEEE TSP 72:701–717 (2024), arXiv:2303.05736 — TV.\[2\]\[3\]
- Cui, Dai, "Channel Estimation for Extremely Large-Scale MIMO: Far-Field or Near-Field?", IEEE TCOM (2022), arXiv:2108.07581 — SA (toàn văn v3).\[32\]
- Hua, Xu, Eldar, "Near-Field 3D Localization via MIMO Radar: Cramér-Rao Bound Analysis and Estimator Design", IEEE TSP (2024), arXiv:2308.16130 — TT + SA.\[6\]
- Cheng, You, Zhou, Yuan, Wu, "Near-Field Multi-Target Localization With Coprime Arrays", IEEE TCOM 73(12):13563–13577 (2025), DOI 10.1109/TCOMM.2025.3605448; bản hội nghị arXiv:2411.01529 — TT.\[33\]
- Gast, Le Magoarou, Shlezinger, "Near Field Localization via AI-Aided Subspace Methods", arXiv:2504.00599 (được nhận ở IEEE TCOM 2026); "DCD-MUSIC", ICASSP 2025; mã https://github.com/ShlezingerLab/AI-Subspace-Methods — TT.\[34\]\[35\]
- Wang, Mu, Liu et al., "Near-Field Integrated Sensing and Communications", IEEE Commun. Lett., arXiv:2302.01153 — TT.\[17\]
- Lu, Zeng et al., "A Tutorial on Near-Field XL-MIMO Communications Towards 6G", IEEE COMST 26(4) (2024), arXiv:2310.11044 — TT.\[12\]\[36\]
- Ramezani, Demir, Björnson, "Localization in Massive MIMO Networks: From Near-Field to Far-Field", arXiv:2402.07644 (2024) — TT.\[37\]\[38\]
- Tutorial "A Tutorial on MIMO-OFDM ISAC: From Far-Field to Near-Field", arXiv:2504.19091 — TT.\[39\]
- "Near-Field Sensing: A Low-Complexity Wavenumber-Domain Method", arXiv:2408.09592 — TT.\[16\]
- "Mixed Near-field and Far-field Localization in XL-MIMO Systems", arXiv:2503.04681 — TT.\[26\]
- "Living Off the Grid: Continuous Range-Angle Super-Resolution for Near-Field XL-MIMO", arXiv:2604.10234 — TT.\[14\]
- "Near-Field Channel Estimation and Joint Angle-Range Recovery in XL-MIMO Systems: A Gridless Super-Resolution Approach", arXiv:2511.23187 — TT.\[15\]
- "Beam Squint Assisted Joint Angle-Distance Localization for Near-Field Communications", arXiv:2509.14850 — TT.\[24\]\[25\]
- Wang et al., "Analytical Framework for Effective Degrees of Freedom in Near-Field XL-MIMO", IEEE TWC, arXiv:2401.15280 — TT.\[40\]
- "MIMO Integrated Sensing and Communication with Extended Target…", arXiv:2205.14050 — TT.\[19\]
- "On Estimation of Angles of Arrival in Monostatic ISAC Without Instantaneous Transmit CSI", arXiv:2510.08793 — TT.\[20\]
- Huang, Barkat, "Near-field multiple source localization by passive sensor array", IEEE TAP 39(7):968–975 (1991) — KĐ (trích qua B3).\[1\]
- Zhang et al., "Localization of near-field sources: A reduced-dimension MUSIC algorithm", IEEE Commun. Lett. 22(7):1422–1425 (2018) — KĐ (trích qua B3).\[1\]
- Ziskind, Wax, "Maximum likelihood localization of multiple sources by alternating projection", IEEE TASSP 36(10):1553–1560 (1988) — KĐ.\[41\]
- Smith, "Statistical resolution limits and the complexified Cramér–Rao bound", IEEE TSP (2005) — KĐ, không kiểm lại.
- Bomfin et al., "Near Field Multi-Band Localization: CRB, Efficient Estimator, and Threshold SNR", arXiv:2606.12314 — SA (đơn nguồn).\[42\]
- "Low-Complexity Near-Field Channel Estimation…" (SBL + Newton), arXiv:2609.14373 — SA.\[43\]

### 7. Hướng mở rộng

- **Wideband/OFDM:**
  - "Wideband Near-Field Sensing in ISAC" (arXiv:2603.27726): từ điển góc–khoảng cách coherence thấp, MUSIC tách hiệu ứng WB/NF.\[44\]
  - "Subspace Fitting Approach for Wideband Near-Field Localization" (arXiv:2508.04169).\[45\]
  - Beam squint (arXiv:2509.14850).
  - CRB wideband compressed-domain (arXiv:2604.08531).\[46\]
  - Lợi ích: thêm thông tin trễ làm tách theo khoảng cách dễ hơn nhiều.
- **Bistatic:** Wang et al. cho thấy với phased array, khi máy thu ở far-field thì cả hai CRB → ∞.\[2\] Đây là trường hợp thoái hóa thú vị. Ngoài ra có CRB điều kiện/không điều kiện cho bistatic MIMO radar near-field: Khamidullina, Podkurkov, Haardt, "Conditional and Unconditional Cramér-Rao Bounds for Near-Field Localization in Bistatic MIMO Radar Systems", IEEE TSP 69:3220–3234 (2021), dùng mô hình sóng cầu chính xác.
- **Mảng phẳng / 3D:** Hua–Xu–Eldar (3D, có clutter); UCA ISAC (arXiv:2604.08160); UPA không vuông (arXiv:2607.18736).\[6\]\[47\]\[48\]
- **Clutter / mục tiêu mở rộng:** Hua–Xu–Eldar (hiệp phương sai nhiễu–giao thoa chưa biết); CRB tối ưu cho mục tiêu mở rộng (arXiv:2604.18166).\[6\]\[49\]
- **Mảng module / mảng thưa:** arXiv:2406.12323 cho thấy tăng số mảng con cải thiện rõ độ phân giải khoảng cách.\[50\]\[51\] Đây là phần mở rộng tự nhiên cho thí nghiệm "khẩu độ so với số anten".

## Recommendations

### Giữ, bỏ, thêm

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| B0 MUSIC far-field 1D | Giữ (ngắn) | Minh họa thất bại khi r < ranh Fresnel |
| B1 2D-MUSIC lưới mịn | Giữ, dùng lưới (u, w) | Chuẩn tham chiếu; chỉ chạy đầy đủ cho N=129/257 |
| B2 Capon/MVDR | Giữ, có diagonal loading | Rẻ; Wang et al. dùng Capon để so với CRB\[2\] |
| B3 FFT-MUSIC | Giữ, tái tạo đúng tham số bài báo | Đối thủ chính |
| P1 | **Đổi thành**: khởi tạo (B3 hoặc MUSIC cục bộ) → AP-DML → joint Gauss–Newton DML có X đã biết, β tách đóng | Đóng góp chính |
| P2 CS | Giữ SOMP + ℓ1-SVD (hoặc group-LASSO/FISTA) trên từ điển (u, w) chính xác + Newton off-grid chung với P1 | So sánh ở T nhỏ và coherent |
| SBL | **Nâng thành bắt buộc** (M-SBL, EM) | Đối chứng mạnh ở SNR thấp |
| AP-DML độc lập | Gộp vào P1 | Tránh trùng lặp |
| Atomic norm | **Bỏ khỏi triển khai**, chỉ đưa vào tổng quan | SDP quá nặng với N=513 |
| Học sâu | Bỏ; tùy chọn chạy mã DCD-MUSIC có sẵn | Ngoài phạm vi\[34\] |
| Thêm: tiền xử lý LS theo X ($\hat G$) | Thêm | Giải quyết hạng R_x |
| Thêm: CRB đa mục tiêu đầy đủ + bản chéo khối + SRL | Thêm | Chuẩn đánh giá |
| Thêm: SS/FBSS near-field | Thêm như thí nghiệm "phản ví dụ" | Cho thấy giới hạn |

### Thí nghiệm nên chạy (theo thứ tự ưu tiên)

1. **E1 – Kiểm chứng:** một mục tiêu; RMSE theo SNR so với CRB chính xác và CRB Fresnel. Tái tạo xu hướng trong Wang et al.: CRB góc bão hòa theo N, CRB Fresnel sai.
2. **E2 – Phân giải theo chiều sâu:** hai mục tiêu cùng θ (0° và 45°), độ tách Δw từ 0,1 đến 2 ô. P_res, δ₅₀ và δ₉₀ cho tất cả thuật toán. SNR ∈ {−10, 0, 10, 20} dB; T ∈ {16, 64, 256}.
3. **E3 – Phân giải theo góc:** cùng r (10 m và 30 m), Δu từ 0,1 đến 2 ô.
4. **E4 – Biên độ lệch:** tỉ số 0/−10/−20 dB; đo tỉ lệ bỏ sót của B3 so với P1 và SBL.
5. **E5 – Hạng R_x:** dạng sóng ISAC với U∈{1, 2, K, N} luồng, cộng luồng dò; nguồn coherent.
6. **E6 – Độ phức tạp:** N∈{129, 257, 513}; thời gian chạy (CPU/GPU) và số phép tính; tái tạo xu hướng Bảng I của B3.
7. **E7 – Lệch mô hình:** từ điển Fresnel áp lên dữ liệu chính xác; ghép nối anten C; d=0,6λ.

**Chuẩn chấm điểm P_res:** sau khi ghép Hungarian, cả hai mục tiêu phải có $|\hat u_k-u_k|<|\Delta u|/2$ và $|\hat w_k-w_k|<|\Delta w|/2$, đồng thời số mục tiêu tìm được bằng K. Báo cáo kèm khoảng tin cậy Wilson. Dùng ≥500 lần Monte-Carlo, như Wang et al. dùng 500.

### Tham số mô phỏng chuẩn đề xuất cho đồ án

- f_c=28 GHz, d=λ/2, N∈{129, 257, 513}; trường hợp mặc định N=257.
- r∈[max(1,2D; 2 m), 0,5·2D²/λ], vùng quan tâm 5–40 m. θ∈[−60°, 60°].
- T mặc định 64. X mặc định là QPSK trực giao hạng N; dạng sóng ISAC có U luồng là tùy chọn.
- SNR định nghĩa theo phần tử, theo snapshot, sau echo.
- Lưới thô: Δu=2/N, Δw theo Cui–Dai (Δ=0,5). Lưới mịn: oversampling ×4.
- Hội tụ của DML: ‖Δη‖ < 10⁻⁶ theo đơn vị ô, tối đa 50 vòng.
- Kiểu dữ liệu complex128 (NumPy); batch tính phổ trên PyTorch (complex64 chỉ để tìm thô).

### Kế hoạch 16 tuần cập nhật

- **Tuần 1–2:** lõi thư viện (geometry, steering chính xác/Fresnel và đạo hàm giải tích, signal monostatic với X và R_x tùy chọn, unit test theo B3/Wang); CRB đơn và đa mục tiêu; E1.
- **Tuần 3–4:** B0, B1, B2; module metrics (Hungarian, P_res, Wilson CI, sàn lưới).
- **Tuần 5–6:** B3, tái tạo đúng kịch bản K=4 (ghi chú rằng mục tiêu 3 m nằm dưới 1,2D) và bảng thời gian chạy.
- **Tuần 7–9:** P1 (AP-DML, joint Gauss–Newton có X đã biết, khởi tạo từ B3); tiền xử lý $\hat G$; E2–E3 sơ bộ.
- **Tuần 10–11:** P2 (SOMP, ℓ1-SVD/FISTA, Newton off-grid) và M-SBL; từ điển (u, w) chính xác và kiểm coherence.
- **Tuần 12–13:** chạy đầy đủ E2–E6; vẽ P_res và RMSE/CRB.
- **Tuần 14:** E7 (bền vững) và thí nghiệm SS/FBSS.
- **Tuần 15–16:** viết báo cáo, đóng gói thư viện (pip, notebook tái tạo), bảo vệ.

### Định vị đóng góp khi bảo vệ

Có ba luận điểm, mỗi luận điểm gắn với một hình:
1. Ngưỡng phân giải P_res của P1 so với B3/MUSIC/SBL, tính theo đơn vị ô.
2. RMSE của P1 chạm CRB đa mục tiêu đầy đủ, trong khi B3 dừng ở sàn lưới.
3. Chi phí tính toán của P1 chỉ cao hơn B3 một lượng nhỏ.

Thêm một luận điểm phụ: phân tích hạng R_x trong ISAC và lời giải $\hat G$.

## Caveats

- Ngân sách tìm kiếm có hạn. Kết luận "chưa có bài nào báo cáo P_res cho joint ML near-field" dựa trên hai vòng tìm kiếm và không loại trừ được các bài hội nghị hoặc tiếng Trung. Hãy chạy thêm một vòng trên IEEE Xplore với từ khóa "statistical resolution limit near-field" trước khi viết phần độ mới.
- Một số nguồn chỉ đọc được tóm tắt hoặc đoạn trích (đánh dấu TT). Một số mã arXiv 2026 (2604.x, 2606.x, 2609.x…) chỉ thấy qua kết quả tìm kiếm, chưa mở trang gốc.
- Các con số ô phân giải, Rayleigh và ranh Fresnel trong bảng cấu hình là do tôi tính từ công thức (Cui–Dai dựa trên Fresnel). Cần kiểm lại bằng vector lái chính xác.
- Các nhận định "kỳ vọng SBL thắng ở T nhỏ" và "DML có ngưỡng thấp nhất" là giả thuyết cần thí nghiệm. Các tài liệu đã đọc không chứng minh trực tiếp.
- Giá trị β mà Cui–Dai dùng trong mô phỏng chưa được xác nhận; chỉ có ví dụ Δ=0,5 → β≈1,6.

## Sources

1. <https://arxiv.org/pdf/2411.15532>
2. [Cramér-Rao Bounds for Near-Field Sensing with Extremely ...](https://arxiv.org/pdf/2303.05736)
3. [Cramér-Rao Bounds for Near-Field Sensing With Extremely Large-Scale MIMO](https://dl.acm.org/doi/10.1109/TSP.2024.3350329)
4. <https://arxiv.org/pdf/2108.07581v3>
5. [Near-Field 3D Localization via MIMO Radar](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/publications/Near-Field_3D_Localization_via_MIMO_Radar_Cramr-Rao_Bound_Analysis_and_Estimator_Design.pdf)
6. [Near-Field 3D Localization via MIMO Radar: Cram\\'er-Rao Bound Analysis and Estimator Design](https://arxiv.org/pdf/2308.16130)
7. [Near-Field Multi-Target Localization With Coprime Arrays](https://ieeexplore.ieee.org/document/11146870/;jsessionid=99DAFD00DAC67C7C2EDE7F9F78AFC58D)
8. [Near Field Localization via AI-Aided Subspace Methods](https://www.scribd.com/document/860629922/Near-Field-Localization-via-AI-Aided-Subspace-Methods)
9. [Near Field Localization via AI-Aided Subspace Methods](https://arxiv.org/pdf/2504.00599)
10. [Near Field Localization via AI-Aided Subspace Methods](https://arxiv.org/html/2504.00599)
11. [FFT-Enhanced Low-Complexity Near-Field Super-Resolution Sensing](https://arxiv.org/html/2411.15532)
12. [IEEE COMMUNICATIONS SURVEYS & TUTORIALS, VOL. 26, NO. 4, FOURTH QUARTER 2024](https://ncrl.seu.edu.cn/_upload/article/files/bf/4b/301e465f4872a355db1b453ca9ec/995541ff-3251-4ba0-bfcd-b2abdc58d6c7.pdf)
13. [A Tutorial on Near-Field XL-MIMO Communications Towards 6G](https://arxiv.org/html/2310.11044)
14. [Living Off the Grid: Continuous Range-Angle Super-Resolution for Near-Field XL-MIMO](https://arxiv.org/html/2604.10234)
15. [Near-Field Channel Estimation and Joint Angle-Range Recovery in XL-MIMO Systems: A Gridless Super-Resolution Approach](https://arxiv.org/html/2511.23187v1)
16. [Near-Field Sensing: A Low-Complexity Wavenumber-Domain Method](https://arxiv.org/pdf/2408.09592)
17. [Near-Field Integrated Sensing and Communications](https://arxiv.org/html/2302.01153)
18. [Analytical Framework for Effective Degrees of Freedom in Near-Field XL-MIMO](https://arxiv.org/pdf/2401.15280)
19. [MIMO Integrated Sensing and Communication with](https://arxiv.org/pdf/2205.14050)
20. [On Estimation of Angles of Arrival in Monostatic ISAC Without Instantaneous Transmit CSI](https://arxiv.org/html/2510.08793)
21. [\[Literature Review\] An Enhanced Polar-Domain Dictionary Design for Elevated BSs in Near-Field U-MIMO](https://www.themoonlight.io/en/review/an-enhanced-polar-domain-dictionary-design-for-elevated-bss-in-near-field-u-mimo)
22. [(PDF) Exploring Frontiers of Polar-Domain Codebooks for Near-Field Channel Estimation and Beam Training: A comprehensive analysis, case studies, and implications for 6G](https://www.researchgate.net/publication/386215745_Exploring_Frontiers_of_Polar-Domain_Codebooks_for_Near-Field_Channel_Estimation_and_Beam_Training_A_comprehensive_analysis_case_studies_and_implications_for_6G)
23. [Beam Squint Assisted Joint Angle-Distance Localization for Near-Field Communications](https://arxiv.org/pdf/2509.14850)
24. [Beam Squint Assisted Joint Angle-Distance Localization for Near-Field Communications](https://arxiv.org/html/2509.14850)
25. [Beam Squint Assisted Joint Angle-Distance Localization for Near-Field Communications](https://arxiv.org/html/2509.14850v1)
26. [Mixed Near-field and Far-field Target Localization for Low- ...](https://arxiv.org/pdf/2503.04681)
27. [Near-Field 3D Localization via MIMO Radar: Cramér-Rao Bound Analysis and Estimator Design](https://ieeexplore.ieee.org/document/10633896/)
28. [Keypoint Detection Empowered Near-Field User Localization and Channel Reconstruction](https://arxiv.org/pdf/2501.11844)
29. [Please do not remove this page Near-Field Wideband OFDM ISAC: Sensing](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13214305230002346&download=true)
30. [Near-Field Localization With Coprime Array](https://arxiv.org/pdf/2411.01529)
31. [1 Near-field Target Localization: Effect of Hardware Impairments](https://arxiv.org/pdf/2512.21480)
32. [Channel Estimation for Extremely Large-Scale MIMO: Far-Field or Near-Field?](https://arxiv.org/abs/2108.07581)
33. [Near-Field Localization With Coprime Array](https://dl.acm.org/doi/10.1145/3636534.3698231)
34. [GitHub - ShlezingerLab/AI-Subspace-Methods: The official implementation for NF-SubspaceNet and DCD-MUSIC for near field localization.](https://github.com/ShlezingerLab/AI-Subspace-Methods)
35. [Nir Shlezinger - Publications](https://sites.google.com/view/nirshl/publications)
36. [arXiv:2310.11044v3 \[cs.IT\] 3 Apr 2024 1 A Tutorial on Near-Field XL-MIMO](https://arxiv.org/pdf/2310.11044)
37. [Localization in Massive MIMO Networks: From Near-Field to Far-Field - ADS](https://ui.adsabs.harvard.edu/abs/2024arXiv240207644R/abstract)
38. [Localization in Massive MIMO Networks: From Near-Field to Far-Field](https://arxiv.org/abs/2402.07644)
39. [A Tutorial on MIMO-OFDM ISAC: From Far-Field to Near-Field](https://arxiv.org/pdf/2504.19091)
40. [\[PDF\] Analytical Framework for Effective Degrees of Freedom in Near-Field XL-MIMO](https://www.semanticscholar.org/paper/Analytical-Framework-for-Effective-Degrees-of-in-Wang-Zhang/781d762d4e71e5c03febfb0df66c523671741671)
41. [Localization in Massive MIMO Networks: From Near-Field to Far-Field](https://arxiv.org/pdf/2402.07644)
42. [Near Field Multi-Band Localization: CRB, Efficient Estimator, and Threshold SNR](https://arxiv.org/pdf/2606.12314)
43. [1 Low-Complexity Near-Field Channel Estimation](https://arxiv.org/pdf/2609.14373)
44. [Wideband Near-Field Sensing in ISAC: Unified Algorithm Design and Decoupled Effect Analysis](https://arxiv.org/html/2603.27726)
45. [Subspace Fitting Approach for Wideband Near-Field Localization](https://arxiv.org/html/2508.04169)
46. [Wideband Compressed-Domain Cram\\'{e}r-Rao Bounds for Near-Field XL-MIMO: Data and Geometric Diversity Decomposition](https://arxiv.org/pdf/2604.08531)
47. [Non-Square UPA-Enabled XL-MIMO Systems: Anisotropic Near-Field Characterization, Fundamental Limits, and Channel Estimation](https://arxiv.org/pdf/2607.18736)
48. [Near-Field Integrated Sensing and Communication with a Uniform Circular Array](https://arxiv.org/html/2604.08160)
49. [Cram\\'{e}r-Rao Bound Optimization for Near-Field ISAC with Extended Targets](https://arxiv.org/pdf/2604.18166)
50. [Near-Field Hybrid Beamforming Design for Modular XL-MIMO ISAC Systems](https://arxiv.org/html/2406.12323)
51. [Near-Field Hybrid Beamforming Design for Modular XL-MIMO ISAC Systems — Lacuna](https://lacuna.tiptreesystems.com/work/near-field-hybrid-beamforming-design-for-modular-xl-mimo-isac-systems/wrk_9e986da461bf01af3090cf056433de21)
