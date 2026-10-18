"""記錄訓練程式 make_sequence 的輸出指紋 tests/data/training_fingerprint.json（不訓練）。

只有在「刻意」改變訓練資料的產生方式後才需要重跑：
    python tests/tools/make_training_fingerprint.py
"""

import json
import sys
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

from paper_harness import load_training  # noqa: E402


def fingerprint(make_sequence, seed=0, count=5):
    """以 default_rng(seed) 連續產生 count 條序列（augment 為 False／True 各一組）的摘要。"""
    out = {}
    for augment in (False, True):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(count):
            features, target = make_sequence(rng, augment)
            rows.append({
                "feature_sums": features.sum(axis=0).tolist(),
                "target_sum": float(target.sum()),
                "slip_steps": int((target > 0).sum()),
                "row_0": features[0].tolist(),
                "row_300": features[300].tolist(),
            })
        out[f"augment={augment}"] = rows
    return out


def main():
    path = TESTS / "data" / "training_fingerprint.json"
    data = {"description": "train_nn_gyro_prior.py 的 make_sequence 以 default_rng(0) 連續產生 5 條序列的摘要",
            "fingerprint": fingerprint(load_training()["make_sequence"])}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已寫入 {path}")


if __name__ == "__main__":
    main()
