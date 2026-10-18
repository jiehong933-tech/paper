"""測試項目 5：MLP 介面。

介面與論文第 3.5 節、第 4.1 節一致（式 (26)、(56)–(58)、(62)）：
  - 每個時間步在 UKF 更新「之後」呼叫一次 ``model.predict``，輸入為
    1×3 特徵 [curve_speed, clip(λ_post), difference]（經 scaler.transform），
    輸出經 clip 到 [0, LAMBDA_MAX] 後成為「下一步」的打滑率先驗 slip_prior。
  - 下一次 ``kf.predict`` 時，全部 2n+1 = 13 個 sigma point 都經過 fx，
    且收到同一個 slip_prior（在 fx 內以 BLEND 混合）。
  也就是說 MLP 不是逐 sigma point 呼叫，而是每步一次、結果套用到所有 sigma point
  （論文：「每週期僅在後驗均值處執行一次 MLP，再將共同先驗提供給 13 個 Sigma 點」）。
"""

import numpy as np
import pytest
from filterpy.kalman import MerweScaledSigmaPoints
from numpy.testing import assert_allclose, assert_array_equal

import synthetic


def test_mlp_called_once_per_step_with_three_features(golden_run):
    calls = golden_run.model.calls
    assert len(calls) == synthetic.N_STEPS
    assert all(c.shape == (1, 3) for c in calls)


def test_mlp_features_match_recorded_quantities(golden_run, defs):
    ns, calls = golden_run.namespace, np.vstack(golden_run.model.calls)
    posterior_slip = np.array(golden_run.ukf["nn"].log["x_post"])[:, defs["SLIP"]]
    assert_allclose(calls[:, 0], np.clip(ns["nn_curve_speed"], 10, 30))
    assert_allclose(calls[:, 1], np.clip(posterior_slip, 0, synthetic.LAMBDA_MAX))
    assert_allclose(calls[:, 2], ns["nn_difference"])


def test_mlp_output_clipped_into_prior(golden_run):
    calls = np.vstack(golden_run.model.calls)
    raw = calls @ golden_run.model.weights + golden_run.model.bias
    assert_allclose(golden_run.namespace["nn_prior"], np.clip(raw, 0, synthetic.LAMBDA_MAX))


@pytest.mark.parametrize("mode", ["analytic", "nn"])
def test_every_sigma_point_receives_previous_prior(golden_run, mode):
    ukf = golden_run.ukf[mode]
    priors = golden_run.namespace["prior_results"][mode]
    assert len(ukf.fx_calls) == synthetic.N_STEPS
    for k, received in enumerate(ukf.fx_calls):
        assert len(received) == 13            # 2n+1，n=6
        expected = 0.0 if k == 0 else priors[k - 1]
        assert received == [expected] * 13


@pytest.mark.parametrize("mode", ["projection", "baseline_check", "bias_known",
                                  "bias_constant"])
def test_modes_without_prior_pass_none(golden_run, mode):
    for received in golden_run.ukf[mode].fx_calls:
        assert received == [None] * 13


def test_update_sigma_points_regenerated_from_prior(golden_run):
    """更新前 sigmas_f 由含 Q 的先驗 (x⁻, P⁻) 重新產生（`kf.sigmas_f = points.sigma_points(...)`）。"""
    points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2.0, kappa=0.0)
    log = golden_run.ukf["nn"].log
    for k in (0, 1, 250, 599):
        assert_allclose(log["sigmas_update"][k],
                        points.sigma_points(log["x_prior"][k], log["P_prior"][k]),
                        rtol=1e-12, atol=1e-12)


def test_nn_equals_analytic_when_mlp_reproduces_analytic_prior(run_factory):
    """MLP 換成「輸出 = difference」的假模型、其他設定相同 → 與解析先驗 UKF 完全一致。"""
    run = run_factory(synthetic.analytic_equivalent_model(), "analytic_eq")
    ns = run.namespace
    assert_array_equal(ns["nn_ukf"], ns["analytic_ukf"])
    assert_array_equal(ns["nn_prior"], ns["analytic_prior"])


def test_nn_equals_projection_when_blend_zero(run_factory):
    """BLEND=0 時先驗不起作用 → NN-UKF 與「僅範圍投影 UKF」完全一致。"""
    run = run_factory(synthetic.golden_model(), "blend0", blend=0.0)
    assert_array_equal(run.namespace["nn_ukf"], run.namespace["projection_ukf"])


def test_zero_output_mlp_pulls_slip_toward_zero(run_factory, golden_run, defs):
    """論文式 (59)、(62)：MLP 輸出的是下一步打滑率的絕對值（不是 Δλ），並以
    (1 − η_λ)λ + η_λ·λ̃ 混入全部 Sigma 點。因此恆輸出 0 的 MLP 代表「預測無打滑」，
    會把 λ 往 0 拉，而不是等同不加先驗的基準 UKF。

    （與「僅範圍投影 UKF」相比：兩者都有範圍投影，差別只在先驗。）
    """
    run = run_factory(synthetic.zero_model(), "zero_mlp")
    ns, SLIP = run.namespace, defs["SLIP"]
    assert np.all(ns["nn_prior"] == 0.0)
    slip_part = slice(200, 400)
    assert ns["nn_ukf"][slip_part, SLIP].mean() < ns["projection_ukf"][slip_part, SLIP].mean()
    assert not np.allclose(ns["nn_ukf"], ns["baseline_ukf"])
