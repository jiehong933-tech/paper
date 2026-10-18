"""計算各濾波器的 NEES／NIS（僅供參考，不做斷言）。

    python tests/tools/nees_nis.py            # 合成資料 + 固定假模型
    python tests/tools/nees_nis.py --real     # 圖 3 真實資料；有 MLP 權重時含 NN-UKF

NEES_k = e_kᵀ P_k⁻¹ e_k，e_k = x_true − x̂_k（x̂_k 為程式存下的結果，含 λ 投影），
P_k 為更新後共變異數；完整 6 維期望值為 6，(v, λ) 子空間期望值為 2。
NIS_k = y_kᵀ S_k⁻¹ y_k，y_k 為量測殘差、S_k 為殘差共變異數，期望值為 2（2 維量測）。
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

import synthetic  # noqa: E402
from paper_harness import (find_weights, model_payload, run_real, run_script,  # noqa: E402
                           training_payload, write_data)

SEGMENTS = {"全段": slice(None), "打滑": slice(200, 400), "恢復": slice(400, None)}


def _quadratic(errors, covariances):
    return np.array([e @ np.linalg.solve(P, e) for e, P in zip(errors, covariances)])


def compute(run):
    ns = run.namespace
    truth, V, SLIP = ns["truth"], ns["V"], ns["SLIP"]
    sub = [V, SLIP]
    rows = {}

    # EKF：先驗 hook 有 (k, x⁻, P⁻, H, z)，後驗 hook 有 (k, x⁺, P⁺)
    x_post = np.array([p[1] for p in run.ekf_post])
    P_post = [p[2] for p in run.ekf_post]
    y = np.array([z - ns["hx"](x) for _, x, _, _, z in run.ekf_prior])
    S = [H @ P @ H.T + ns["R"] for _, _, P, H, _ in run.ekf_prior]
    rows["EKF"] = (truth - x_post, P_post, y, S)

    names = {"baseline_check": "基準 UKF", "projection": "僅範圍投影 UKF",
             "analytic": "解析先驗 UKF", "nn": "MLP 先驗 UKF",
             "bias_known": "偏差已知 UKF", "bias_constant": "常數偏差 UKF"}
    for mode, label in names.items():
        log = run.ukf[mode].log
        rows[label] = (truth - ns["ukf_results"][mode], log["P_post"],
                       np.array(log["y"]), log["S"])

    table = {}
    for label, (e, P, y, S) in rows.items():
        nees = _quadratic(e, P)
        nees_sub = _quadratic(e[:, sub], [p[np.ix_(sub, sub)] for p in P])
        nis = _quadratic(y, S)
        table[label] = {seg: (nees[s].mean(), nees_sub[s].mean(), nis[s].mean())
                        for seg, s in SEGMENTS.items()}
    return table


def print_markdown(table, title):
    print(f"\n#### {title}\n")
    print("| 方法 | " + " | ".join(f"NEES₆ {s} | NEES(v,λ) {s} | NIS {s}" for s in SEGMENTS)
          + " |")
    print("|---|" + "---:|" * (3 * len(SEGMENTS)))
    for label, values in table.items():
        cells = []
        for seg in SEGMENTS:
            cells += [f"{v:.2f}" for v in values[seg]]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\n（期望值：NEES₆ = 6，NEES(v,λ) = 2，NIS = 2）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    with tempfile.TemporaryDirectory() as workdir:
        if args.real:
            if find_weights() is not None:
                run = run_real(workdir)
                title = "圖 3 真實資料（figure3_simulation.py，SEED=42）＋實際 MLP 權重"
            else:
                run = run_real(workdir, payload=training_payload(synthetic.zero_model()))
                title = ("圖 3 真實資料（figure3_simulation.py，SEED=42）；找不到 MLP 權重，"
                         "MLP 先驗 UKF 一列使用恆輸出 0 的假模型，不具意義")
        else:
            write_data(workdir, synthetic.make_scenario())
            run = run_script(workdir, model_payload(synthetic.golden_model(),
                                                    synthetic.IdentityScaler()))
            title = f"合成資料（seed={synthetic.SEED}，固定假模型 golden_model）"
    print_markdown(compute(run), title)


if __name__ == "__main__":
    main()
