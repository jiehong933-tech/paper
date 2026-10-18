# NN-UKF 列車運動與輪軸打滑聯合估測

| 檔案 | 用途 |
|---|---|
| `figure3_simulation.py` | 圖 3：產生模擬資料與基準 UKF，輸出 `figure3_shared_data.npz` |
| `train_nn_gyro_prior.py` | 訓練 MLP 打滑率先驗，輸出 `nn_gyro_prior_retrained.joblib` |
| `paper_code.py` | 論文圖 4、圖 5 與消融表：EKF、基準 UKF、範圍投影／解析先驗／MLP 先驗 UKF 及偏差對照組 |

## 執行順序

三支程式都讀寫**程式所在的資料夾**（Colab 為目前資料夾），從哪個資料夾執行都可以；
依序執行即可串接：

```bash
python figure3_simulation.py    # 約 2 秒，輸出 figure3_shared_data.npz 與圖 3
python train_nn_gyro_prior.py   # 約 1–2 分鐘，輸出 nn_gyro_prior_retrained.joblib 與訓練圖
python paper_code.py            # 讀取上面兩個檔案，輸出圖 4、圖 5、結果 NPZ／CSV 與消融表
```

缺少 `figure3_shared_data.npz`（也可放在 `figure3_output/`）或模型檔時，
`paper_code.py` 會說明缺哪個檔、該先執行哪支程式。

## 執行測試

```bash
pip install -r requirements-dev.txt
pytest                                   # 全部測試
pytest -m "not slow"                     # 略過需要實際資料／權重的回歸測試
pytest --cov --cov-config=.coveragerc    # 含覆蓋率（涵蓋三支程式）
```

- 測試不會修改、也不需要改寫三支程式：`tests/paper_harness.py` 以 AST 解析原始檔，
  在暫存目錄中以合成資料（`tests/synthetic.py`）與假模型執行，不需要 Colab、Drive 或中文字型。
- 回歸測試（`tests/test_regression.py`，真實資料部分標記 `slow`）：
  - 真實資料由測試當場執行 `figure3_simulation.py` 產生；EKF、基準 UKF 等與 MLP 無關的方法
    不需要權重即可檢查，並與 `tests/data/figure3_regression_baseline.json` 比對。
  - 以論文數值為準：表 4、表 6 的 RMSE、錯誤平衡點等（`tests/test_thesis_settings.py` 另檢查
    論文表 2、表 3、式 (68) 與第 3.5 節的各項設定）。
  - MLP 先驗 UKF 與 MLP 本身（表 5、式 (60)、epoch、自由遞迴）需要 MLP 權重，找不到時 skip；
    可把權重放在 repo 根目錄，或以環境變數指定資料夾：

  ```bash
  PAPER_DATA_DIR=/path/to/folder pytest -m slow
  ```

- 工具腳本（皆在 `tests/tools/`）：
  - `make_synthetic_golden.py`：刻意改變數值行為後，重新產生合成資料回歸基準。
  - `record_real_baseline.py`：以圖 3 真實資料重新記錄與 MLP 無關之方法的回歸基準。
  - `nees_nis.py [--real]`：計算各濾波器的 NEES／NIS（`--real` 用圖 3 真實資料；僅供參考，不做斷言）。
