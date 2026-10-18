"""測試項目 8：回歸測試。

A. 合成資料回歸（快速）：以固定合成資料與固定假模型（不訓練）重跑整份
   `paper_code.py`，與 tests/data/synthetic_golden.json 比對。基準值由
   tests/tools/make_synthetic_golden.py 依「目前程式」產生，用來偵測日後
   修改是否改變了數值結果。

B. 論文數值重現（slow）：用 repo 內的 MLP 權重與資料重跑主模擬，確認
   恢復段速度 RMSE（NN-UKF 1.034、baseline UKF 5.842、EKF 3.796 m/s）
   與打滑區間（NN-UKF 1.120；EKF、baseline UKF > 3.0），容許 ±0.001。
   repo 目前沒有 nn_gyro_prior_retrained.joblib 與 figure3_shared_data.npz，
   因此會 skip；可設定 PAPER_DATA_DIR 指向存放兩個檔案的資料夾後再執行。
"""

import json
from pathlib import Path

import pytest

from paper_harness import find_real_inputs, run_real, speed_slip_metrics

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC_GOLDEN = DATA_DIR / "synthetic_golden.json"
REAL_BASELINE = DATA_DIR / "real_regression_baseline.json"

PAPER_SPEED_RMSE = {          # (方法, 區段) -> 論文數值（m/s）
    ("ukf_nn", "speed_recovery"): 1.034,
    ("baseline_ukf", "speed_recovery"): 5.842,
    ("ekf", "speed_recovery"): 3.796,
    ("ukf_nn", "speed_slip"): 1.120,
}


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


# ---------- B. 論文數值重現（需實際權重與資料） ----------

@pytest.fixture(scope="module")
def real_metrics(tmp_path_factory):
    data, model = find_real_inputs()
    if model is None:
        pytest.skip("找不到 MLP 權重 nn_gyro_prior_retrained.joblib（repo 內未附）")
    if data is None:
        pytest.skip("找不到 figure3_shared_data.npz（repo 內未附，需由圖 3 程式產生）")
    run = run_real(tmp_path_factory.mktemp("real"))
    return speed_slip_metrics(run.namespace), run


@pytest.mark.slow
@pytest.mark.parametrize("key, expected", list(PAPER_SPEED_RMSE.items()),
                         ids=[f"{m}-{s}" for m, s in PAPER_SPEED_RMSE])
def test_paper_speed_rmse_reproduced(real_metrics, key, expected):
    metrics, _ = real_metrics
    method, segment = key
    assert metrics[method][segment] == pytest.approx(expected, abs=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["ekf", "baseline_ukf"])
def test_paper_baselines_fail_during_slip(real_metrics, method):
    metrics, _ = real_metrics
    assert metrics[method]["speed_slip"] > 3.0


@pytest.mark.slow
def test_paper_baseline_reconstruction(real_metrics):
    _, run = real_metrics
    assert run.namespace["max_gap"] <= 1e-6


@pytest.mark.slow
def test_other_baselines_regression(real_metrics):
    """其他 baseline（含 bias_known＝matched-bias UKF、bias_constant、projection、
    analytic）與 tests/data/real_regression_baseline.json 比對；
    該檔需先以 tests/tools/record_real_baseline.py 在有資料的環境產生。"""
    if not REAL_BASELINE.exists():
        pytest.skip("尚未建立 real_regression_baseline.json（需實際資料與權重）")
    metrics, _ = real_metrics
    recorded = json.loads(REAL_BASELINE.read_text(encoding="utf-8"))["metrics"]
    for method, values in recorded.items():
        for metric, expected in values.items():
            assert metrics[method][metric] == pytest.approx(expected, abs=1e-6), \
                f"{method}.{metric}"
