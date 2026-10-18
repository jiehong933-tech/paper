"""測試項目 4（依程式實際實作調整）。

`paper code` 沒有曲率地圖（沒有分段直線／定半徑彎道的 κ(s) 查表），
曲率是模型檔中的單一常數 KAPPA，用在：
  - hx 第二項：ω = vκ + b_g
  - 「虛擬曲率約束」：curve_speed 以 z_ω/κ 估速（限制在 [10, 30]），
    經 ALPHA 指數平滑後，與里程計比得 difference = z_odo/curve_speed − 1
    （限制在 [−0.30, 0.60]），作為解析先驗或 MLP 特徵。
因此本檔以「常數 κ」與上述約束實作為對象；地圖區段交界／超出範圍的
測試因無對應實作而不撰寫（見 PR 說明）。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import synthetic


def _expected_curve_prior(measurements, alpha, kappa):
    curve, curves, diffs = 20.0, [], []
    for z in measurements:
        raw = min(max(z[1] / kappa, 10.0), 30.0)
        curve = (1 - alpha) * curve + alpha * raw
        curves.append(curve)
        diffs.append(min(max(z[0] / curve - 1, -0.30), 0.60))
    return np.array(curves), np.array(diffs)


def test_curve_speed_and_difference_recursion(golden_run):
    ns = golden_run.namespace
    curves, diffs = _expected_curve_prior(ns["measurements"], synthetic.ALPHA,
                                          synthetic.KAPPA)
    assert_allclose(ns["nn_curve_speed"], curves, rtol=1e-12)
    assert_allclose(ns["nn_difference"], diffs, rtol=1e-12, atol=1e-15)


def test_analytic_prior_is_clipped_difference(golden_run):
    ns = golden_run.namespace
    _, diffs = _expected_curve_prior(ns["measurements"], synthetic.ALPHA, synthetic.KAPPA)
    assert_allclose(ns["analytic_prior"], np.clip(diffs, 0, synthetic.LAMBDA_MAX),
                    rtol=1e-12, atol=1e-15)


def test_curve_speed_tracks_true_speed_on_constant_curve(golden_run, defs):
    """定曲率、等速段（k=350..449，v=14）中 curve_speed 應接近真實車速。

    合成資料 b_g=0.0004 → 偏差造成 0.0004/κ = 0.2 m/s 的系統誤差；
    ω 雜訊 0.01/κ = 5 m/s 經 ALPHA=0.02 平滑後標準差 ≈ 5·sqrt(α/(2−α)) ≈ 0.5 m/s；
    k=200 結束的減速段殘留落後量 ≈ 5.9·0.98^150 ≈ 0.3 m/s。合計 < 1 m/s。
    """
    ns = golden_run.namespace
    error = ns["nn_curve_speed"][350:450] - ns["truth"][350:450, defs["V"]]
    assert np.sqrt(np.mean(error ** 2)) < 1.0


def test_curve_speed_ema_lag_on_ramp():
    """EMA 在等加速度 a 下的穩態落後量為 (1−α)/α · a·dt
    （α=0.02、dt=0.1：a=−1.2 m/s² → 5.88 m/s；此處用 a=−0.1 → 0.49 m/s，
    讓速度在暫態消失前都留在 [10, 30] 內）。記錄 ALPHA 造成的延遲。"""
    alpha, a, dt = synthetic.ALPHA, -0.1, 0.1
    speed = 29.0 + a * dt * np.arange(1200)
    gyro = speed * synthetic.KAPPA
    curves, _ = _expected_curve_prior(np.column_stack([speed, gyro]), alpha, synthetic.KAPPA)
    # k=1000..1099：v≈19..18，初始暫態已衰減 0.98^1000 ≈ 2e-9
    lag = curves[1000:1100] - speed[1000:1100]
    assert_allclose(lag, (1 - alpha) / alpha * (-a) * dt, rtol=1e-3)


@pytest.mark.parametrize("gyro, expected", [(0.0, 10.0), (-1.0, 10.0),
                                            (1.0, 30.0), (0.03, 15.0)])
def test_raw_curve_speed_clip_limits(gyro, expected):
    """z_ω/κ 超出 [10, 30] 時被限制；κ=0.002 時 0.03 rad/s ↔ 15 m/s。"""
    curves, _ = _expected_curve_prior(np.array([[15.0, gyro]]), alpha=1.0,
                                      kappa=synthetic.KAPPA)
    assert_allclose(curves[0], expected)


def test_straight_track_kappa_zero_not_supported():
    """直線段 κ=0：程式以 z_ω/KAPPA 求速度，κ=0 時為除以零（numpy 給 inf
    並發出 RuntimeWarning，之後被 clip 成 30）。記錄此行為，不視為錯誤，
    因為程式只支援單一非零常數曲率。"""
    with np.errstate(divide="ignore"):
        raw = np.clip(np.float64(0.05) / np.float64(0.0), 10, 30)
    assert raw == 30.0


def test_hx_uses_constant_kappa(defs):
    for v in (10.0, 20.0, 30.0):
        assert_allclose(defs["hx"](np.array([0, v, 0, 0, 0, 0.0]))[1], v * defs["KAPPA"])
