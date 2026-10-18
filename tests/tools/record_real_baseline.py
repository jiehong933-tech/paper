"""以真實圖 3 資料記錄 paper_code.py 中與 MLP 無關的方法的 RMSE，作為回歸基準。

先執行 figure3_simulation.py 產生資料，再以假 MLP（參數取自訓練程式）跑 paper_code.py，
寫入 tests/data/figure3_regression_baseline.json。只有在「刻意」改變數值行為後才需要重跑：
    python tests/tools/record_real_baseline.py

若找得到 MLP 權重（repo 根目錄或 PAPER_DATA_DIR），另外印出含 NN-UKF 的完整數值（不寫檔）。
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

import synthetic  # noqa: E402
from paper_harness import (MLP_INDEPENDENT, find_weights, run_real,  # noqa: E402
                           speed_slip_metrics, training_payload)


def show(metrics):
    for name, values in metrics.items():
        print(f"{name:<20}", {k: round(v, 3) for k, v in values.items()})


def main():
    warnings.filterwarnings("ignore")
    with tempfile.TemporaryDirectory() as workdir:
        run = run_real(workdir, payload=training_payload(synthetic.zero_model()))
        metrics = {name: speed_slip_metrics(run.namespace)[name] for name in MLP_INDEPENDENT}
    path = TESTS / "data" / "figure3_regression_baseline.json"
    out = {"description": "figure3_simulation.py（SEED=42）真實資料下，paper_code.py 中與 MLP "
                          "權重無關之方法的 RMSE（m/s 與打滑率）；記錄目前程式的數值，非論文數值。",
           "metrics": metrics}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    show(metrics)
    print(f"已寫入 {path}")

    if find_weights() is not None:
        with tempfile.TemporaryDirectory() as workdir:
            print("\n含實際 MLP 權重：")
            show(speed_slip_metrics(run_real(workdir).namespace))


if __name__ == "__main__":
    main()
