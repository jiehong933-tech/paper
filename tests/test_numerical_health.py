"""測試項目 6：整段模擬中 P 保持對稱正定（每步 Cholesky 成功），且無 NaN。"""

import numpy as np
import pytest

from paper_harness import ukf_mode_order


def _check_covariances(covariances, label):
    for k, P in enumerate(covariances):
        assert np.all(np.isfinite(P)), f"{label} 第 {k} 步 P 含 NaN/inf"
        scale = max(np.abs(P).max(), 1.0)
        asymmetry = np.abs(P - P.T).max() / scale
        assert asymmetry < 1e-10, f"{label} 第 {k} 步 P 不對稱（{asymmetry:.2e}）"
        try:
            np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            pytest.fail(f"{label} 第 {k} 步 Cholesky 失敗")


@pytest.mark.parametrize("mode", ukf_mode_order())
def test_ukf_covariance_spd_every_step(golden_run, mode):
    log = golden_run.ukf[mode].log
    assert len(log["P_prior"]) == len(log["P_post"]) == 600
    _check_covariances(log["P_prior"], f"{mode} 先驗")
    _check_covariances(log["P_post"], f"{mode} 後驗")


@pytest.mark.parametrize("mode", ukf_mode_order())
def test_ukf_innovation_covariance_spd(golden_run, mode):
    _check_covariances(golden_run.ukf[mode].log["S"], f"{mode} S")


def test_ekf_covariance_spd_every_step(golden_run):
    assert len(golden_run.ekf_prior) == len(golden_run.ekf_post) == 600
    _check_covariances([p[2] for p in golden_run.ekf_prior], "EKF 先驗")
    _check_covariances([p[2] for p in golden_run.ekf_post], "EKF 後驗")


def test_no_nan_in_any_result(golden_run):
    ns = golden_run.namespace
    for name in ("ekf", "baseline_ukf", "projection_ukf", "analytic_ukf", "nn_ukf",
                 "bias_known_ukf", "bias_constant_ukf", "analytic_prior", "nn_prior",
                 "nn_curve_speed", "nn_difference"):
        assert np.all(np.isfinite(ns[name])), name


def test_projected_modes_keep_slip_in_range(golden_run, defs):
    ns = golden_run.namespace
    for name in ("projection_ukf", "analytic_ukf", "nn_ukf"):
        slip = ns[name][:, defs["SLIP"]]
        assert slip.min() >= 0 and slip.max() <= ns["LAMBDA_MAX"], name


@pytest.mark.parametrize("mode", ["projection", "analytic", "nn"])
def test_projection_does_not_touch_covariance(golden_run, defs, mode):
    """λ 投影只改均值、不改 P（記錄行為）：每一步下一次 predict 前的 P 都等於更新後的 P，
    包括確實發生投影的步。"""
    log = golden_run.ukf[mode].log
    slip = np.array(log["x_post"])[:, defs["SLIP"]]
    clipped = np.flatnonzero((slip < 0) | (slip > golden_run.namespace["LAMBDA_MAX"]))
    assert clipped[clipped < len(slip) - 1].size > 0          # 確實有投影發生
    for k in range(len(slip) - 1):
        np.testing.assert_array_equal(log["P_before_predict"][k + 1], log["P_post"][k])
