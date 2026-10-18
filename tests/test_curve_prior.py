"""測試項目 4（依程式實際實作調整）。

`paper_code.py` 沒有曲率地圖（沒有分段直線／定半徑彎道的 κ(s) 查表），
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


def _with_gyro(scenario, gyro):
    """把合成資料的陀螺儀量測換成指定序列（並重算圖 3 基準 UKF）。"""
    measurement = scenario["measurement"].copy()
    measurement[:, 1] = gyro
    data = dict(scenario, measurement=measurement)
    data["baseline_ukf"] = synthetic.reference_baseline_ukf(data)
    return data


def test_curve_speed_ema_lag_on_ramp(run_factory, scenario):
    """EMA 在等加速度 a 下的穩態落後量為 (1−α)/α · a·dt。

    α=0.02、dt=0.1 時，a=−1.2 m/s² 會落後 5.88 m/s；此處讓陀螺儀量測對應
    a=−0.1 m/s² 的速度斜坡（25 → 19 m/s，留在 [10, 30] 內），程式的
    curve_speed 應落後 0.49 m/s。記錄 ALPHA 造成的延遲。
    """
    a, dt = -0.1, 0.1
    speed = 25.0 + a * dt * np.arange(synthetic.N_STEPS)
    run = run_factory(synthetic.golden_model(), "gyro_ramp",
                      scenario=_with_gyro(scenario, speed * synthetic.KAPPA))
    # k=400..599：起始值 20 與 25 的暫態已衰減到 5·0.98^400 ≈ 0.002
    lag = run.namespace["nn_curve_speed"][400:] - speed[400:]
    alpha = synthetic.ALPHA
    assert_allclose(lag, (1 - alpha) / alpha * (-a) * dt, rtol=1e-2)


def test_raw_curve_speed_clip_limits(run_factory, scenario):
    """z_ω/κ 超出 [10, 30] 時被限制（ALPHA=1 讓 curve_speed 直接等於換算值）。

    κ=0.002：0.03 rad/s ↔ 15 m/s、0.015 ↔ 7.5（→10）、0.07 ↔ 35（→30）、負值 → 10。
    """
    gyro = np.full(synthetic.N_STEPS, 0.03)
    gyro[100:200], gyro[200:300], gyro[300:400] = 0.015, 0.07, -0.001
    run = run_factory(synthetic.golden_model(), "gyro_clip",
                      scenario=_with_gyro(scenario, gyro), alpha=1.0)
    expected = np.full(synthetic.N_STEPS, 15.0)
    expected[100:200], expected[200:300], expected[300:400] = 10.0, 30.0, 10.0
    assert_allclose(run.namespace["nn_curve_speed"], expected, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_straight_track_kappa_zero(run_factory):
    """直線段 κ=0：程式以 z_ω/KAPPA 換算速度，κ=0 時為除以零（±inf），
    之後被 clip 成 10 或 30。記錄此行為：整段仍跑得完、沒有 NaN，
    但曲率速度不再有意義（程式只支援單一非零常數曲率）。"""
    run = run_factory(synthetic.golden_model(), "kappa_zero", kappa=0.0)
    ns = run.namespace
    curve = ns["nn_curve_speed"]
    assert np.all(np.isfinite(curve)) and curve.min() >= 10 and curve.max() <= 30
    for name in ("ekf", "nn_ukf", "analytic_ukf", "projection_ukf"):
        assert np.all(np.isfinite(ns[name])), name


def test_hx_uses_constant_kappa(defs):
    for v in (10.0, 20.0, 30.0):
        assert_allclose(defs["hx"](np.array([0, v, 0, 0, 0, 0.0]))[1], v * defs["KAPPA"])
