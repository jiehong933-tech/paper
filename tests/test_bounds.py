"""各範圍限制（clip）確實由 paper_code.py 執行，且測試確實觸發到邊界（不空轉）。

合成資料平常只會碰到部分邊界，因此以較小的 LAMBDA_MAX、或改造過的量測強制觸發：
  - λ 投影 [0, LAMBDA_MAX]（範圍投影、解析先驗、MLP 先驗三種模式）
  - 解析先驗與 MLP 先驗的輸出 [0, LAMBDA_MAX]
  - difference 特徵 [−0.30, 0.60]
（曲率速度 [10, 30] 見 test_curve_prior.py。）
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import synthetic

SMALL_LMAX = 0.1


@pytest.fixture(scope="module")
def small_lmax_run(run_factory):
    return run_factory(synthetic.golden_model(), "small_lmax", lambda_max=SMALL_LMAX)


@pytest.fixture(scope="module")
def outlier_run(run_factory, scenario):
    """在 k=300 與 k=320 放入里程計離群值，讓 difference 超出 [−0.30, 0.60]。"""
    measurement = scenario["measurement"].copy()
    measurement[300, 0] *= 2.5     # difference ≈ 2.2 → 上限 0.60
    measurement[320, 0] *= 0.3     # difference ≈ −0.6 → 下限 −0.30
    data = dict(scenario, measurement=measurement)
    data["baseline_ukf"] = synthetic.reference_baseline_ukf(data)
    return run_factory(synthetic.golden_model(), "odometer_outliers", scenario=data)


def _slip(run, mode, defs):
    posterior = np.array(run.ukf[mode].log["x_post"])[:, defs["SLIP"]]
    result = run.namespace["ukf_results"][mode][:, defs["SLIP"]]
    return posterior, result


@pytest.mark.parametrize("mode", ["projection", "analytic", "nn"])
def test_lower_projection_bound(golden_run, defs, mode):
    posterior, result = _slip(golden_run, mode, defs)
    assert posterior.min() < 0                         # 下界確實被觸發
    assert_array_equal(result, np.clip(posterior, 0, synthetic.LAMBDA_MAX))
    assert result.min() == 0.0


@pytest.mark.parametrize("mode", ["projection", "analytic", "nn"])
def test_upper_projection_bound(small_lmax_run, defs, mode):
    posterior, result = _slip(small_lmax_run, mode, defs)
    assert posterior.max() > SMALL_LMAX                # 上界確實被觸發
    assert_array_equal(result, np.clip(posterior, 0, SMALL_LMAX))
    assert result.max() == SMALL_LMAX


def test_unprojected_modes_leave_the_range(golden_run, defs):
    """基準 UKF（未投影）的 λ 會跑到負值：投影模式的測試因此不是空轉。"""
    posterior, result = _slip(golden_run, "baseline_check", defs)
    assert posterior.min() < 0
    assert_array_equal(result, posterior)


def test_prior_upper_bound(small_lmax_run):
    ns = small_lmax_run.namespace
    difference = ns["nn_difference"]       # 只由量測決定，解析與 MLP 模式相同
    assert difference.max() > SMALL_LMAX
    assert_array_equal(ns["analytic_prior"], np.clip(difference, 0, SMALL_LMAX))
    raw = np.vstack(small_lmax_run.model.calls) @ small_lmax_run.model.weights
    assert raw.max() > SMALL_LMAX
    assert_array_equal(ns["nn_prior"], np.clip(raw, 0, SMALL_LMAX))


def test_difference_clip_bounds(outlier_run):
    ns = outlier_run.namespace
    difference = ns["nn_difference"]
    assert (difference[300], difference[320]) == (0.60, -0.30)
    others = np.delete(difference, [300, 320])
    assert others.min() > -0.30 and others.max() < 0.60
    # 解析先驗與 MLP 特徵用的都是限制後的值
    assert (ns["analytic_prior"][300], ns["analytic_prior"][320]) == (synthetic.LAMBDA_MAX, 0.0)
    features = np.vstack(outlier_run.model.calls)
    assert (features[300, 2], features[320, 2]) == (0.60, -0.30)
