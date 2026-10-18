"""測試項目 8：回歸測試。

A. 合成資料回歸（快速）：固定合成資料與固定假模型（不訓練）重跑 paper_code.py，
   與 tests/data/synthetic_golden.json 比對（由 tests/tools/make_synthetic_golden.py 產生）。

B. 真實圖 3 資料、與 MLP 無關的方法（slow）：先執行 figure3_simulation.py 產生資料，
   再以假 MLP 跑 paper_code.py。EKF、基準 UKF、投影、解析先驗與偏差對照組都不依賴
   MLP 權重，因此不需要權重檔即可檢查：
     - 論文數值（EKF 恢復段 3.796、基準 UKF 恢復段 5.842、兩者打滑段 > 3.0）；
       目前程式與資料重現不了的項目標記 xfail，原因中附上實際值。
     - 目前數值的回歸基準 tests/data/figure3_regression_baseline.json
       （由 tests/tools/record_real_baseline.py 產生），容許相對誤差 1e-6。

C. NN-UKF（slow）：需要 MLP 權重 nn_gyro_prior_retrained.joblib（repo 根目錄或
   PAPER_DATA_DIR），找不到時 skip。
"""

import json
from pathlib import Path

import pytest

from paper_harness import MLP_INDEPENDENT, find_weights, run_real, speed_slip_metrics

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC_GOLDEN = DATA_DIR / "synthetic_golden.json"
FIGURE3_BASELINE = DATA_DIR / "figure3_regression_baseline.json"


# ---------- A. 合成資料回歸 ----------

def _golden_cases():
    golden = json.loads(SYNTHETIC_GOLDEN.read_text(encoding="utf-8"))["metrics"]
    return [(m, k, v) for m, values in golden.items() for k, v in values.items()]


@pytest.mark.parametrize("method, metric, expected", _golden_cases())
def test_synthetic_golden_metrics(golden_run, method, metric, expected):
    actual = speed_slip_metrics(golden_run.namespace)[method][metric]
    # 容許跨平台 BLAS 的捨入差異；任何實質的數值改變都遠大於此
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_synthetic_golden_covers_all_methods(golden_run):
    golden = json.loads(SYNTHETIC_GOLDEN.read_text(encoding="utf-8"))["metrics"]
    assert set(golden) == set(speed_slip_metrics(golden_run.namespace))


# ---------- B. 真實圖 3 資料、與 MLP 無關的方法 ----------

@pytest.fixture(scope="module")
def real_metrics(real_run):
    return speed_slip_metrics(real_run.namespace)


NOT_REPRODUCED = (
    "目前的 figure3_simulation.py（SEED=42）＋ paper_code.py 重現不了論文數值；"
    "此方法不依賴 MLP 權重，差異來自資料或程式版本。實際值："
)


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=NOT_REPRODUCED + "EKF 恢復段 5.119（論文 3.796）")
def test_paper_ekf_recovery_rmse(real_metrics):
    assert real_metrics["ekf"]["speed_recovery"] == pytest.approx(3.796, abs=1e-3)


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=NOT_REPRODUCED + "基準 UKF 恢復段 4.791（論文 5.842）")
def test_paper_baseline_ukf_recovery_rmse(real_metrics):
    assert real_metrics["baseline_ukf"]["speed_recovery"] == pytest.approx(5.842, abs=1e-3)


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=NOT_REPRODUCED + "EKF 打滑段 2.830（論文 > 3.0）")
def test_paper_ekf_fails_during_slip(real_metrics):
    assert real_metrics["ekf"]["speed_slip"] > 3.0


@pytest.mark.slow
def test_paper_baseline_ukf_fails_during_slip(real_metrics):
    assert real_metrics["baseline_ukf"]["speed_slip"] > 3.0


def _figure3_cases():
    recorded = json.loads(FIGURE3_BASELINE.read_text(encoding="utf-8"))["metrics"]
    return [(m, k, v) for m, values in recorded.items() for k, v in values.items()]


@pytest.mark.slow
@pytest.mark.parametrize("method, metric, expected", _figure3_cases())
def test_figure3_regression_baseline(real_metrics, method, metric, expected):
    """真實資料下各方法（含 bias_known＝偏差已知／matched-bias UKF）目前數值的回歸基準。"""
    assert real_metrics[method][metric] == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_figure3_baseline_covers_mlp_independent_methods():
    recorded = json.loads(FIGURE3_BASELINE.read_text(encoding="utf-8"))["metrics"]
    assert set(recorded) == set(MLP_INDEPENDENT)


# ---------- C. NN-UKF（需要 MLP 權重） ----------

@pytest.fixture(scope="module")
def nn_metrics(tmp_path_factory):
    if find_weights() is None:
        pytest.skip("找不到 MLP 權重 nn_gyro_prior_retrained.joblib（repo 內未附；"
                    "可放在 repo 根目錄或以 PAPER_DATA_DIR 指定）")
    return speed_slip_metrics(run_real(tmp_path_factory.mktemp("real_mlp")).namespace)


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=(
    "目前的圖 3 資料已與論文不同（見 B 段的 EKF／基準 UKF），NN-UKF 不論權重為何都無法對上；"
    "以 train_nn_gyro_prior.py 重訓的權重（83 epoch）實際為恢復段 0.725、打滑段 0.570"
    "（論文 1.034、1.120）"))
@pytest.mark.parametrize("segment, expected", [("speed_recovery", 1.034), ("speed_slip", 1.120)])
def test_paper_nn_ukf_rmse(nn_metrics, segment, expected):
    assert nn_metrics["ukf_nn"][segment] == pytest.approx(expected, abs=1e-3)


@pytest.mark.slow
def test_nn_ukf_beats_baselines_in_both_segments(nn_metrics):
    for segment in ("speed_slip", "speed_recovery"):
        assert nn_metrics["ukf_nn"][segment] < nn_metrics["ekf"][segment]
        assert nn_metrics["ukf_nn"][segment] < nn_metrics["baseline_ukf"][segment]
