# NN-UKF 列車運動與輪軸打滑聯合估測

| 檔案 | 用途 |
|---|---|
| `paper code` | 論文圖 4、圖 5 與消融表：EKF、基準 UKF、範圍投影／解析先驗／MLP 先驗 UKF 及偏差對照組 |
| `train_nn_gyro_prior.py` | 訓練 MLP 打滑率先驗，輸出 `nn_gyro_prior_retrained.joblib` |

`paper code` 執行時需要同目錄（或 `figure3_output/`）下的 `figure3_shared_data.npz`
（圖 3 程式產生）以及 `nn_gyro_prior_retrained.joblib`；這兩個檔案目前未放在 repo 中。

## 執行測試

```bash
pip install -r requirements-dev.txt
pytest                                   # 全部測試
pytest -m "not slow"                     # 略過需要實際資料／權重的回歸測試
pytest --cov --cov-config=.coveragerc    # 含覆蓋率（涵蓋 `paper code` 與訓練程式）
```

- 測試不會修改、也不需要改寫 `paper code`：`tests/paper_harness.py` 以 AST 解析原始檔，
  在暫存目錄中以合成資料（`tests/synthetic.py`）與假模型執行，不需要 Colab、Drive 或中文字型。
- 論文數值重現測試（`tests/test_regression.py`，標記 `slow`）需要實際資料與 MLP 權重；
  找不到時會自動 skip。可把兩個檔案放在 repo 根目錄，或以環境變數指定資料夾：

  ```bash
  PAPER_DATA_DIR=/path/to/files pytest -m slow
  ```

- 工具腳本（皆在 `tests/tools/`）：
  - `make_synthetic_golden.py`：刻意改變數值行為後，重新產生合成資料回歸基準。
  - `record_real_baseline.py`：以實際資料記錄所有方法的 RMSE，作為之後的回歸基準。
  - `nees_nis.py [--real]`：計算各濾波器的 NEES／NIS（僅供參考，不做斷言）。
