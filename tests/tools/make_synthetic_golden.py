"""依目前的 `paper code` 產生合成資料回歸基準 tests/data/synthetic_golden.json。

只有在「刻意」改變數值行為後才需要重新產生：
    python tests/tools/make_synthetic_golden.py
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

import synthetic  # noqa: E402
from paper_harness import model_payload, run_script, speed_slip_metrics, write_data  # noqa: E402


def main():
    warnings.filterwarnings("ignore")
    with tempfile.TemporaryDirectory() as workdir:
        write_data(workdir, synthetic.make_scenario())
        run = run_script(workdir, model_payload(synthetic.golden_model(),
                                                synthetic.IdentityScaler()))
    out = {
        "description": "合成資料（tests/synthetic.py，seed=%d）與固定假模型 golden_model() "
                       "下 `paper code` 的 RMSE；非論文數值。" % synthetic.SEED,
        "metrics": speed_slip_metrics(run.namespace),
    }
    path = TESTS / "data" / "synthetic_golden.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已寫入 {path}")


if __name__ == "__main__":
    main()
