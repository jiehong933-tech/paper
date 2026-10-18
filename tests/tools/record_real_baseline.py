"""以論文實際資料與 MLP 權重執行 `paper code`，記錄所有方法的 RMSE 作為回歸基準。

需要 figure3_shared_data.npz 與 nn_gyro_prior_retrained.joblib（放在 repo 根目錄、
repo/figure3_output/，或以環境變數 PAPER_DATA_DIR 指定資料夾）：
    PAPER_DATA_DIR=/path/to/files python tests/tools/record_real_baseline.py
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

from paper_harness import run_real, speed_slip_metrics  # noqa: E402


def main():
    warnings.filterwarnings("ignore")
    with tempfile.TemporaryDirectory() as workdir:
        run = run_real(workdir)
    if run is None:
        sys.exit("找不到實際資料或權重檔，無法建立基準。")
    metrics = speed_slip_metrics(run.namespace)
    path = TESTS / "data" / "real_regression_baseline.json"
    path.write_text(json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    for name, values in metrics.items():
        print(name, {k: round(v, 3) for k, v in values.items()})
    print(f"已寫入 {path}")


if __name__ == "__main__":
    main()
