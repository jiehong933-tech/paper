"""測試項目 8：回歸測試。以論文（碩士論文 PDF）的數值為準。

A. 合成資料回歸（快速）：固定合成資料與固定假模型（不訓練）重跑 paper_code.py，
   與 tests/data/synthetic_golden.json 比對（由 tests/tools/make_synthetic_golden.py 產生）。

B. 論文第四章（slow）：測試中先執行 figure3_simulation.py 產生 seed = 42 的資料，再跑
   paper_code.py。純里程計、標準 EKF 與 MLP 以外的五種 UKF 都不依賴 MLP 權重，因此以
   假 MLP 即可檢查：
     - 表 4、表 6 的 RMSE（論文四捨五入至小數第三位，容許 ±0.0005）。
     - 第 4.4.2、4.5 節描述的錯誤平衡點：標準 EKF 約 (25 m/s, −0.20)、基準 UKF 約
       (15 m/s, 0.34)，兩者皆近似滿足 v(1+λ) ≈ 20 m/s。
     - 全精度回歸基準 tests/data/figure3_regression_baseline.json（相對誤差 1e-6）。

C. MLP 先驗 UKF 與 MLP 本身（slow）：使用 repo 內附的 MLP 權重 nn_gyro_prior_retrained.joblib
   （PAPER_DATA_DIR 可指定另一組；找不到時 skip）。檢查表 5／表 6 的 MLP 列、第 4.5 節
   的暫態描述、式 (60) 的測試指標、第 83 個 epoch 停止／第 67 個 epoch 最佳，以及
   式 (61) 自由遞迴的穩態值。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from paper_harness import (MLP_INDEPENDENT, find_weights, load_training, run_real,
                           speed_slip_metrics, training_rollout)

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC_GOLDEN = DATA_DIR / "synthetic_golden.json"
FIGURE3_BASELINE = DATA_DIR / "figure3_regression_baseline.json"
ROUNDING = 5e-4 + 1e-12      # 論文數值四捨五入至小數第三位

# 論文表 4（純里程計、標準 EKF、基準 UKF）與表 6（六種 UKF）：
# (速度／打滑區間, 速度／恢復區間, 打滑率／打滑區間, 打滑率／恢復區間)
THESIS_TABLE_4 = {
    "raw_odometry": (4.004, 0.409),
    "ekf": (2.830, 5.119),
    "baseline_ukf": (3.692, 4.791),
}
THESIS_TABLE_6 = {
    "baseline_ukf": (3.692, 4.791, 0.168, 0.322),
    "ukf_baseline_check": (3.692, 4.791, 0.168, 0.322),
    "ukf_projection": (2.502, 5.342, 0.118, 0.372),
    "ukf_analytic": (0.540, 0.769, 0.032, 0.042),
    "ukf_bias_known": (0.489, 0.356, 0.032, 0.025),
    "ukf_bias_constant": (0.614, 0.888, 0.038, 0.043),
}
THESIS_MLP_ROW = (0.570, 0.725, 0.034, 0.039)     # 表 5、表 6：MLP 先驗 UKF
METRICS = ("speed_slip", "speed_recovery", "slip_slip", "slip_recovery")


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


# ---------- B. 論文第四章：與 MLP 無關的方法 ----------

@pytest.fixture(scope="module")
def real_metrics(real_run):
    return speed_slip_metrics(real_run.namespace)


def _table_cases(table):
    return [(method, metric, value) for method, values in table.items()
            for metric, value in zip(METRICS, values)]


@pytest.mark.slow
@pytest.mark.parametrize("method, metric, expected", _table_cases(THESIS_TABLE_4))
def test_thesis_table_4(real_metrics, method, metric, expected):
    assert real_metrics[method][metric] == pytest.approx(expected, abs=ROUNDING)


@pytest.mark.slow
@pytest.mark.parametrize("method, metric, expected", _table_cases(THESIS_TABLE_6))
def test_thesis_table_6_without_mlp(real_metrics, method, metric, expected):
    assert real_metrics[method][metric] == pytest.approx(expected, abs=ROUNDING)


@pytest.mark.slow
@pytest.mark.parametrize("name, speed, slip", [("ekf", 25.0, -0.20), ("baseline_ukf", 15.0, 0.34)])
def test_thesis_incorrect_equilibria(real_run, name, speed, slip):
    """第 4.4.2、4.5 節：恢復區間收斂到觀測等值集合上的錯誤平衡點（以 k = 500..599 平均判斷）。"""
    ns = real_run.namespace
    tail = ns[name][500:]
    v, lam = tail[:, ns["V"]].mean(), tail[:, ns["SLIP"]].mean()
    assert v == pytest.approx(speed, abs=0.5)
    assert lam == pytest.approx(slip, abs=0.02)
    assert v * (1 + lam) == pytest.approx(20.0, abs=0.3)


def _figure3_cases():
    recorded = json.loads(FIGURE3_BASELINE.read_text(encoding="utf-8"))["metrics"]
    return [(m, k, v) for m, values in recorded.items() for k, v in values.items()]


@pytest.mark.slow
@pytest.mark.parametrize("method, metric, expected", _figure3_cases())
def test_figure3_regression_baseline(real_metrics, method, metric, expected):
    """真實資料下各方法目前數值的全精度回歸基準（與論文表 4、表 6 一致）。"""
    assert real_metrics[method][metric] == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_figure3_baseline_covers_mlp_independent_methods():
    recorded = json.loads(FIGURE3_BASELINE.read_text(encoding="utf-8"))["metrics"]
    assert set(recorded) == set(MLP_INDEPENDENT)


def test_figure3_baseline_agrees_with_thesis_tables():
    """記錄的全精度數值四捨五入後就是論文表 4、表 6 的數字（兩份基準彼此一致）。"""
    recorded = json.loads(FIGURE3_BASELINE.read_text(encoding="utf-8"))["metrics"]
    for table in (THESIS_TABLE_4, THESIS_TABLE_6):
        for method, metric, expected in _table_cases(table):
            assert recorded[method][metric] == pytest.approx(expected, abs=ROUNDING)


# ---------- C. MLP 先驗 UKF 與 MLP 本身（需要 MLP 權重） ----------

@pytest.fixture(scope="module")
def weights():
    path = find_weights()
    if path is None:
        pytest.skip("找不到 MLP 權重 nn_gyro_prior_retrained.joblib（應位於 repo 根目錄，"
                    "或以 PAPER_DATA_DIR 指定）")
    import joblib
    return joblib.load(path)


@pytest.fixture(scope="module")
def nn_run(weights, tmp_path_factory):
    return run_real(tmp_path_factory.mktemp("real_mlp"))


@pytest.mark.slow
@pytest.mark.parametrize("metric, expected", list(zip(METRICS, THESIS_MLP_ROW)))
def test_thesis_mlp_prior_ukf(nn_run, metric, expected):
    """表 5、表 6：MLP 先驗 UKF 速度 0.570／0.725 m/s、打滑率 0.034／0.039。"""
    assert speed_slip_metrics(nn_run.namespace)["ukf_nn"][metric] == pytest.approx(expected, abs=ROUNDING)


@pytest.mark.slow
def test_thesis_mlp_prior_ukf_transients(nn_run):
    """第 4.5 節：打滑開始時速度短暫升至約 21.2 m/s；打滑率由約 0.12 逐步上升、後段最高約 0.23；
    k = 400 後速度先降至約 18.5 m/s，再回到 20 m/s 附近、打滑率趨近於零。"""
    ns = nn_run.namespace
    v, lam = ns["nn_ukf"][:, ns["V"]], ns["nn_ukf"][:, ns["SLIP"]]
    assert v[200:400].max() == pytest.approx(21.2, abs=0.2)
    assert lam[200:220].mean() == pytest.approx(0.12, abs=0.02)
    assert lam[200:400].max() == pytest.approx(0.23, abs=0.02)
    assert v[400:].min() == pytest.approx(18.5, abs=0.2)
    assert v[550:].mean() == pytest.approx(20.0, abs=0.2) and abs(lam[550:].mean()) < 0.02


@pytest.mark.slow
def test_thesis_eq60_mlp_test_metrics(weights):
    """式 (60)：MAE 0.01144、RMSE 0.02316、打滑區 RMSE 0.04285、R² 0.9100（模型檔中的記錄）。"""
    assert weights["test_mae"] == pytest.approx(0.01144, abs=5e-6)
    assert weights["test_rmse"] == pytest.approx(0.02316, abs=5e-6)
    assert weights["active_rmse"] == pytest.approx(0.04285, abs=5e-6)
    assert weights["test_r2"] == pytest.approx(0.9100, abs=5e-5)


@pytest.mark.slow
def test_thesis_training_epochs_and_size(weights):
    """第 3.5 節：第 83 個 epoch 停止、保留第 67 個 epoch 驗證集 R² 最佳的權重；共 4,481 個參數。"""
    model = weights["model"]
    assert model.n_iter_ == 83
    assert int(np.argmax(model.validation_scores_)) + 1 == 67
    assert sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_) == 4481


@pytest.mark.slow
def test_thesis_eq61_free_recursion(weights):
    """式 (61)、圖 2(b)：r_o = 0 時打滑率降至零附近；r_o = 0.1、0.2 時分別趨近約 0.10、0.21。"""
    _, final_values = training_rollout(load_training(), weights["model"], weights["scaler"])
    assert abs(final_values[0]) < 0.02
    assert final_values[1] == pytest.approx(0.10, abs=0.01)
    assert final_values[2] == pytest.approx(0.21, abs=0.01)
