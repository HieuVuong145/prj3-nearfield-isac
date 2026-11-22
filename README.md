# nfisac: Siêu phân giải nhiều mục tiêu trong Near-Field ISAC (XL-MIMO)

Thư viện Python và bộ thí nghiệm phục vụ nghiên cứu ước lượng tham số (khoảng cách $r$, góc $\theta$) cho các hệ thống ISAC dựa trên mảng anten cực lớn (XL-MIMO).

Kế hoạch hiện hành: [KE_HOACH_4_TUAN.md](KE_HOACH_4_TUAN.md). Quyết định thiết kế: [DECISIONS.md](DECISIONS.md).

## Cài đặt

```bash
pip install -e ".[dev]"
```

## Kiểm thử

```bash
pytest -q          # 60 test, < 10 s
ruff check .
```

## Chạy thí nghiệm

```bash
python -m experiments.e0_sanity          # F1: phổ MUSIC + tái tạo ca N=65       (~5 s)
python -m experiments.srl_calibration    # SRL từ CRB -> chọn dải quét           (~10 s)
python -m experiments.gate_baseline      # cổng baseline + dò P_res của B1/B1-r  (~2 phút, 10 lõi)
python -m experiments.e1_rmse_snr        # F2 bản nháp: RMSE theo SNR            (~40 s, 10 lõi)
```

Kết quả: `results/figs/*.png`, `results/*.csv`, `results/*.json`; cache Monte-Carlo ở `results/cache/`
(khoá theo hash cấu hình; xoá thư mục này nếu đổi code của hàm trial).

## Cấu trúc thư mục

- `nfisac/`: mã nguồn thư viện
  - `config.py`: dataclass cấu hình (`ArrayCfg`, `Target`, `ProjectCfg`), nạp YAML
  - `geometry.py`: vector lái (exact, Fresnel, far-field, vị trí anten tuỳ chọn), đạo hàm, coherence, đổi (r,θ)↔(u,w)
  - `signal.py`: dạng sóng (`iso`, `beam`, `mix`, `orth`, `streams`), echo monostatic có hook lệch mô hình
  - `crb.py`: FIM, CRB đa mục tiêu, PEB, CRB theo (u,w), CRB của độ tách (SRL), CRB chéo khối
  - `grids.py`: ô phân giải đo bằng số (`cell_u`, `cell_w`), lưới (u,w) theo ô, lưới cục bộ (ROI)
  - `spectrum.py`: phổ MUSIC/Capon theo khối, cache vector lái lưới
  - `peaks.py`: tìm K đỉnh 1D/2D
  - `mdl.py`: ước lượng số mục tiêu (MDL)
  - `estimators/`: B0 `MusicFarField`, B1/B1-r `Music2D`, B2 `Capon2D` (B3, P1, P2, SBL: việc #18–#29)
  - `metrics.py`: ghép Hungarian theo ô, `is_resolved`, P_res + Wilson CI, RMSE
  - `mc.py`: Monte-Carlo song song, seed theo trial, cache `.npz`
- `configs/`: `default.yaml` (N=129), `resolution.yaml` (N=257, T=512)
- `experiments/`: kịch bản thí nghiệm (mỗi file = một hình)
- `tests/`: pytest

## Tiến độ

Danh sách đầy đủ kèm tiêu chí nghiệm thu: [KE_HOACH_4_TUAN.md §5](KE_HOACH_4_TUAN.md). Làm lần lượt từ trên xuống.

- [x] 1. T0–T3: repo, geometry, signal, CRB
- [x] 2. Bổ sung geometry/signal (dạng sóng `orth`, `streams`, hook lệch mô hình)
- [x] 3. Bổ sung CRB (theo (u, w), SRL, chéo khối)
- [x] 4. `grids.py`
- [x] 5. `spectrum.py`
- [x] 6. `peaks.py`
- [x] 7. `mdl.py`
- [x] 8. `metrics.py`
- [x] 9. `mc.py`
- [x] 10. B0 MUSIC trường xa
- [x] 11. B1 2D-MUSIC và B1-r
- [x] 12. B2 Capon
- [x] 13. F1 sanity + tái tạo ca N=65
- [x] 14. Hiệu chỉnh SRL → dải quét
- [x] 15. Cổng quyết định baseline
- [x] 16. F2 bản nháp
- [x] 17. Bản thảo Chương 2
- [ ] 18. B3 FFT-MUSIC
- [ ] 19. `refine_dml` (hàm chi phí known-X + gradient + L-BFGS)
- [ ] 20. AP-DML
- [ ] 21. P1 = B3-init → AP-DML → joint DML
- [ ] 22. Tiền xử lý Ĝ
- [ ] 23. Cổng quyết định P1
- [ ] 24. F2 đầy đủ
- [ ] 25. F3
- [ ] 26. Bản thảo Chương 3
- [ ] 27. Từ điển CS (u, w)
- [ ] 28. SOMP + ℓ1-SVD
- [ ] 29. M-SBL
- [ ] 30. F4
- [ ] 31. F5
- [ ] 32. F6
- [ ] 33. F7
- [ ] 34. F8
- [ ] 35. Code freeze
- [ ] 36. Bản thảo Chương 4
- [ ] 37. `run_all --quick`, `summary.csv`
- [ ] 38. README kết quả
- [ ] 39. Chương 5
- [ ] 40. Chương 6
- [ ] 41. Chương 1
- [ ] 42. Slide bảo vệ
