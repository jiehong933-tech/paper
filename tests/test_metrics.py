"""測試項目 7：RMSE 與區段切分（以手算小例子驗證）。"""

import csv

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_rmse_hand_example(defs):
    # 誤差 (0, 0, 2) → sqrt(4/3)
    assert_allclose(defs["rmse"](np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])),
                    np.sqrt(4 / 3), rtol=1e-15)


def test_rmse_constant_offset_and_symmetry(defs):
    a = np.array([3.0, -1.0, 7.5, 0.0])
    assert_allclose(defs["rmse"](a + 0.5, a), 0.5, rtol=1e-15)
    assert defs["rmse"](a, a + 2) == defs["rmse"](a + 2, a)
    assert defs["rmse"](a, a) == 0.0


def test_rmse_mixed_signs(defs):
    # 誤差 (3, -4) → sqrt((9+16)/2) = sqrt(12.5)
    assert_allclose(defs["rmse"](np.array([3.0, -4.0]), np.zeros(2)), np.sqrt(12.5))


def test_segments(defs):
    assert (defs["SLIP_START"], defs["SLIP_END"]) == (200, 400)
    assert defs["slip_part"] == slice(200, 400)
    assert defs["recovery_part"] == slice(400, len(defs["step"]))


def test_segment_boundaries_hand_example(defs):
    """k=0..599：打滑區間為 200..399（含 200、不含 400），恢復區間為 400..599。"""
    index = np.arange(600)
    slip, recovery = index[defs["slip_part"]], index[defs["recovery_part"]]
    assert (slip[0], slip[-1], slip.size) == (200, 399, 200)
    assert (recovery[0], recovery[-1], recovery.size) == (400, 599, 200)
    # k < 200 不屬於任何評估區段
    assert set(range(200)).isdisjoint(set(slip) | set(recovery))


def test_segment_rmse_hand_example(defs):
    estimate = np.zeros(600)
    estimate[200:400] = 2.0       # 打滑區間誤差 2
    estimate[400:] = np.tile([1.0, -1.0], 100)   # 恢復區間誤差 ±1
    estimate[:200] = 100.0        # 不計入
    truth = np.zeros(600)
    assert defs["rmse"](estimate[defs["slip_part"]], truth[defs["slip_part"]]) == 2.0
    assert defs["rmse"](estimate[defs["recovery_part"]], truth[defs["recovery_part"]]) == 1.0


def test_ablation_table_matches_results(golden_run, defs):
    """輸出的 ablation_rmse.csv 與 rmse(結果, 真值) 一致。"""
    ns, V, SLIP = golden_run.namespace, defs["V"], defs["SLIP"]
    truth, sp, rp = ns["truth"], ns["slip_part"], ns["recovery_part"]
    with open(golden_run.workdir / "ablation_rmse.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    assert len(rows) == 6
    for (name, *values), (key, result) in zip(rows, ns["ablation_methods"].items()):
        assert name == key
        expected = [defs["rmse"](result[sp, V], truth[sp, V]),
                    defs["rmse"](result[rp, V], truth[rp, V]),
                    defs["rmse"](result[sp, SLIP], truth[sp, SLIP]),
                    defs["rmse"](result[rp, SLIP], truth[rp, SLIP])]
        assert_allclose([float(v) for v in values], expected, atol=5e-7)


def test_latex_table_rows(golden_run):
    lines = (golden_run.workdir / "ablation_rmse_latex.txt").read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 6
    for line, (name, values) in zip(lines, golden_run.namespace["ablation_metrics"].items()):
        assert line == f"{name} & " + " & ".join(f"{v:.3f}" for v in values) + r" \\"


@pytest.mark.parametrize("name", ["Raw Odometry", "Standard EKF", "Baseline UKF",
                                  "NN-prior UKF"])
def test_main_methods_printed(golden_run, defs, name):
    ns = golden_run.namespace
    speed, truth = ns["main_methods"][name], ns["truth"][:, defs["V"]]
    a = defs["rmse"](speed[ns["slip_part"]], truth[ns["slip_part"]])
    b = defs["rmse"](speed[ns["recovery_part"]], truth[ns["recovery_part"]])
    line = next(l for l in golden_run.stdout.splitlines() if l.startswith(name))
    assert line.split()[-2:] == [f"{a:.3f}", f"{b:.3f}"]
