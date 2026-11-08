# nfisac: Siêu phân giải nhiều mục tiêu trong Near-Field ISAC (XL-MIMO)

Thư viện Python và bộ thí nghiệm phục vụ nghiên cứu ước lượng tham số (khoảng cách $r$, góc $\theta$) cho các hệ thống ISAC dựa trên mảng anten cực lớn (XL-MIMO).

## Cài đặt

```bash
pip install -e .
# Hoặc cài thêm dev dependencies:
pip install -e ".[dev]"
```

## Kiểm thử

```bash
pytest -q
```

## Cấu trúc thư mục

- `nfisac/`: Mã nguồn thư viện chính
  - `config.py`: Định nghĩa cấu hình, dataclass (`ArrayCfg`, `Target`, `ProjectCfg`) và hàm nạp YAML
  - `geometry.py`: Vector lái (exact, Fresnel, far-field), đạo hàm, hệ số kết hợp
  - `signal.py`: Bộ sinh dạng sóng, mô phỏng echo khứ hồi monostatic
  - `crb.py`: FIM, CRB đa mục tiêu và PEB
  - `grids.py`: Lưới miền cực theo hệ số kết hợp
  - `spectrum.py`: Phổ không gian MUSIC, Capon xử lý chunked
  - `peaks.py`: Tìm cực đại cục bộ 2D
  - `mdl.py`: Ước lượng số mục tiêu bằng tiêu chuẩn MDL
  - `estimators/`: Các bộ ước lượng (B0, B1, B2, P1, P2, O1)
  - `metrics.py`: Hungarian matching, RMSE, xác suất phân giải ($P_{\mathrm{res}}$)
  - `mc.py`: Quản lý mô phỏng Monte-Carlo
- `configs/`: File cấu hình YAML (`default.yaml`)
- `tests/`: Bộ kiểm thử tự động với pytest
- `experiments/`: Kịch bản thí nghiệm (E0 - E10)
- `results/`: Kết quả mô phỏng, đồ thị và số liệu tổng hợp

## Tiến độ thực hiện (Tasks)

- [x] **T0: Khởi tạo repo** — Cấu trúc dự án, dataclass cấu hình, `pyproject.toml`, test tự động.
- [x] **T1: Geometry** — Vector lái, đạo hàm giải tích, độ kết hợp.
- [x] **T2: Signal** — Dạng sóng, echo monostatic, SNR.
- [x] **T3: CRB** — FIM đa mục tiêu, CRB, PEB.
- [ ] **T4: Grids & Spectrum** — Lưới miền cực thích ứng, phổ MUSIC chunked, 2D peak finding.
- [ ] **T5: Baseline B1 & Metrics** — 2D-MUSIC, MDL, đánh giá Hungarian, RMSE.
- [ ] **T6: Baseline B0 & B2** — Far-field MUSIC 1D, Capon near-field.
- [ ] **T7: Proposed P1** — MUSIC phân cấp + tinh chỉnh DML.
- [ ] **T8: Proposed P2** — Compressive Sensing (SOMP, group-LASSO FISTA, ℓ1-SVD).
- [ ] **T9: Tùy chọn** — AP-DML, mảng ảo, SBL, EDoF.
- [ ] **T10: Monte-Carlo runner** — Song song, cache, bit-exact reproduction.
- [ ] **T11: Thí nghiệm** — E0 tới E10.
- [ ] **T12: Đóng gói** — Báo cáo tổng hợp, script tái lập toàn bộ kết quả.
