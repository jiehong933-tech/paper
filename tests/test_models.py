"""測試項目 1–3：狀態轉移 fx、觀測 hx、EKF Jacobian，以及各 UKF 模式的設定。

狀態順序以程式為準：x = [s, v, ψ, λ, b_a, b_g]（S, V, YAW, SLIP, BA, BG）。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from paper_harness import ekf_measurement_jacobian, load_definitions


def test_state_order_matches_thesis(defs):
    assert (defs["S"], defs["V"], defs["YAW"], defs["SLIP"], defs["BA"], defs["BG"]) \
        == (0, 1, 2, 3, 4, 5)
    assert defs["DT"] == 0.1


# ---------- 1. 狀態轉移 fx ----------

def test_fx_returns_six_states(defs):
    out = defs["fx"](np.zeros(6), defs["DT"], np.zeros(2))
    assert out.shape == (6,)


def test_fx_constant_speed_zero_bias(defs):
    dt, s, v, yaw, slip, omega = defs["DT"], 123.0, 20.0, 0.3, 0.1, 0.05
    x = np.array([s, v, yaw, slip, 0.0, 0.0])
    out = defs["fx"](x, dt, np.array([0.0, omega]))
    assert_allclose(out, [s + v * dt, v, yaw + omega * dt, slip, 0.0, 0.0],
                    rtol=0, atol=1e-12)


def test_fx_bias_states_unchanged_and_subtracted(defs):
    dt = defs["DT"]
    x = np.array([5.0, 15.0, 1.0, 0.2, 0.05, 0.002])
    u = np.array([0.7, 0.04])
    out = defs["fx"](x, dt, u)
    expected = [5.0 + 15.0 * dt, 15.0 + (0.7 - 0.05) * dt,
                1.0 + (0.04 - 0.002) * dt, 0.2, 0.05, 0.002]
    assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_fx_ignores_extra_input_columns(defs):
    x = np.array([0.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    a = defs["fx"](x, 0.1, np.array([0.1, 0.2]))
    b = defs["fx"](x, 0.1, np.array([0.1, 0.2, 99.0]))
    assert_array_equal(a, b)


def test_fx_does_not_mutate_input(defs):
    x = np.array([1.0, 2.0, 3.0, 0.1, 0.01, 0.001])
    before = x.copy()
    defs["fx"](x, 0.1, np.array([0.5, 0.5]), slip_prior=0.3)
    assert_array_equal(x, before)


@pytest.mark.parametrize("blend", [0.0, 0.3, 1.0])
def test_fx_slip_prior_blend(blend):
    d = load_definitions(blend=blend)
    x = np.array([0.0, 20.0, 0.0, 0.2, 0.0, 0.0])
    out = d["fx"](x, 0.1, np.zeros(2), slip_prior=0.4)
    assert_allclose(out[d["SLIP"]], (1 - blend) * 0.2 + blend * 0.4, atol=1e-15)
    # 其他狀態不受先驗影響
    plain = d["fx"](x, 0.1, np.zeros(2))
    assert_array_equal(np.delete(out, d["SLIP"]), np.delete(plain, d["SLIP"]))


def test_fx_slip_prior_zero_is_not_none(defs):
    """slip_prior=0.0 仍會混合（把 λ 往 0 拉），與 None（不混合）不同。"""
    x = np.array([0.0, 20.0, 0.0, 0.2, 0.0, 0.0])
    with_zero = defs["fx"](x, 0.1, np.zeros(2), slip_prior=0.0)
    without = defs["fx"](x, 0.1, np.zeros(2), slip_prior=None)
    assert_allclose(with_zero[defs["SLIP"]], (1 - defs["BLEND"]) * 0.2)
    assert without[defs["SLIP"]] == 0.2


# ---------- 2. 觀測 hx ----------

def test_hx_odometer_term(defs):
    x = np.array([0.0, 14.0, 0.0, 0.43, 0.0, 0.0])
    assert_allclose(defs["hx"](x)[0], 14.0 * 1.43, rtol=1e-12)
    assert_allclose(defs["hx"](x)[0], 20.02, rtol=0, atol=1e-9)


@pytest.mark.parametrize("v", [0.0, 5.0, 14.0, 20.0, 30.0])
def test_hx_no_slip_equals_speed(defs, v):
    x = np.array([0.0, v, 0.0, 0.0, 0.1, 0.0])
    assert defs["hx"](x)[0] == v


def test_hx_gyro_term(defs):
    x = np.array([0.0, 18.0, 0.0, 0.1, 0.0, 0.003])
    assert_allclose(defs["hx"](x)[1], 18.0 * defs["KAPPA"] + 0.003, rtol=1e-12)
    assert defs["hx"](x).shape == (2,)


# ---------- 3. EKF Jacobian 與數值微分 ----------

def _numeric_jacobian(func, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    columns = []
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        columns.append((func(x + step) - func(x - step)) / (2 * eps))
    return np.column_stack(columns)


@pytest.fixture
def random_states():
    rng = np.random.default_rng(0)
    return [np.array([rng.uniform(0, 1e3), rng.uniform(5, 30), rng.uniform(-1, 1),
                      rng.uniform(0, 0.5), rng.normal(0, 0.05), rng.normal(0, 0.005)])
            for _ in range(5)]


def test_ekf_H_matches_numeric_jacobian(defs, random_states):
    for x in random_states:
        H = ekf_measurement_jacobian(defs, x)
        assert_allclose(H, _numeric_jacobian(defs["hx"], x), rtol=1e-6, atol=1e-8)


def test_ekf_H_analytic_entries(defs):
    v, slip = 14.0, 0.43
    H = ekf_measurement_jacobian(defs, [0.0, v, 0.0, slip, 0.0, 0.0])
    assert H[0, defs["V"]] == pytest.approx(1 + slip)      # ∂h1/∂v = 1+λ
    assert H[0, defs["SLIP"]] == pytest.approx(v)          # ∂h1/∂λ = v
    assert H[1, defs["V"]] == pytest.approx(defs["KAPPA"])  # ∂h2/∂v = κ
    assert H[1, defs["BG"]] == 1                           # ∂h2/∂b_g = 1
    assert np.count_nonzero(H) == 4


def test_ekf_F_matches_numeric_jacobian(defs, random_states):
    u = np.array([0.3, 0.05])
    for x in random_states:
        J = _numeric_jacobian(lambda s: defs["fx"](s, defs["DT"], u), x)
        assert_allclose(defs["F"], J, rtol=1e-6, atol=1e-8)


# ---------- 各 UKF 模式的設定（P0、Q、範圍投影） ----------

def test_baseline_modes_use_shared_P0_Q(golden_run, defs):
    for mode in ("projection", "analytic", "nn", "baseline_check"):
        ukf = golden_run.ukf[mode]
        assert_array_equal(ukf.initial_P, defs["P0"])
        assert_array_equal(ukf.initial_Q, defs["Q"])


def test_bias_modes_only_change_bias_cells(golden_run, defs):
    P0, Q, BA, BG = defs["P0"], defs["Q"], defs["BA"], defs["BG"]
    known, constant = golden_run.ukf["bias_known"], golden_run.ukf["bias_constant"]
    assert (known.initial_P[BG, BG], known.initial_Q[BG, BG]) == (1e-10, 1e-14)
    assert np.argwhere(known.initial_P != P0).tolist() == [[BG, BG]]
    assert np.argwhere(known.initial_Q != Q).tolist() == [[BG, BG]]
    assert_array_equal(constant.initial_P, P0)
    assert np.argwhere(constant.initial_Q != Q).tolist() == [[BA, BA], [BG, BG]]
    assert constant.initial_Q[BA, BA] == constant.initial_Q[BG, BG] == 1e-14


def test_mode_settings_do_not_modify_shared_P0_Q(golden_run, defs):
    ns = golden_run.namespace
    assert_array_equal(ns["P0"], defs["P0"])
    assert_array_equal(ns["Q"], defs["Q"])


@pytest.mark.parametrize("mode, clipped", [
    ("projection", True), ("analytic", True), ("nn", True),
    ("baseline_check", False), ("bias_known", False), ("bias_constant", False),
])
def test_slip_projection_only_in_projected_modes(golden_run, defs, mode, clipped):
    """存下的結果 = 更新後狀態；投影模式的 λ 另外 clip 到 [0, LAMBDA_MAX]。"""
    SLIP = defs["SLIP"]
    posterior = np.array(golden_run.ukf[mode].log["x_post"])
    result = golden_run.namespace["ukf_results"][mode]
    expected = np.clip(posterior[:, SLIP], 0, defs["LAMBDA_MAX"]) if clipped else posterior[:, SLIP]
    assert_array_equal(result[:, SLIP], expected)
    assert_array_equal(np.delete(result, SLIP, axis=1), np.delete(posterior, SLIP, axis=1))
    if not clipped:   # 合成資料中，未投影模式的 λ 確實會跑出範圍
        assert posterior[:, SLIP].min() < 0 or posterior[:, SLIP].max() > defs["LAMBDA_MAX"]
