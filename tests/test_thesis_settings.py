"""以論文為準的設定檢查。

論文：義守大學電機工程學系碩士論文〈地下環境下基於 UKF 之列車運動與輪軸打滑聯合估測〉。
每項測試註明論文出處（表、式或節）；三支程式的設定都必須與論文一致。
論文的數值結果（表 4–6、式 (60) 等）見 tests/test_regression.py。
"""

import ast

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from paper_harness import (FIG3_FILE, PAPER_FILE, TRAIN_FILE, clip_bounds, load_training,
                           statement)

# 表 2：模擬情境與感測器參數
TABLE_2 = {"DT": 0.1, "STEPS": 600, "TRUE_SPEED": 20.0, "KAPPA": 0.002, "TRUE_SLIP": 0.2,
           "SLIP_START": 200, "SLIP_END": 400, "SIGMA_ODO": 0.4, "SIGMA_GYRO": 0.01, "SEED": 42}


@pytest.mark.parametrize("name, value", TABLE_2.items())
def test_table2_simulation_settings(fig3, name, value):
    assert fig3[0][name] == value


def test_table2_zero_true_bias_and_acceleration(fig3):
    """表 2：真實加速度、真實偏差與過程擾動皆為零；加速度計無雜訊（理想化）。"""
    ns, _ = fig3
    assert np.all(ns["truth"][:, [ns["BA"], ns["BG"]]] == 0)
    assert np.all(ns["inputs"][:, 0] == 0.0)
    assert np.all(ns["truth"][:, ns["V"]] == 20.0)


def test_eq68_initial_state_and_covariances(defs, fig3):
    """式 (68)：x̂0 = [0, 20, 0, 0, 0, 0]、P0 = 0.1 I、Q = 1e-4 I、R = diag(0.4², 0.01²)。

    圖 3 程式使用相同設定（其 P0 另由 test_figure3 的 max_gap = 0 確認）。
    """
    assert_array_equal(defs["X0"], [0, 20, 0, 0, 0, 0])
    assert_array_equal(defs["P0"], 0.1 * np.eye(6))
    assert_array_equal(defs["Q"], 1e-4 * np.eye(6))
    assert_array_equal(defs["R"], np.diag([0.4 ** 2, 0.01 ** 2]))
    ns, _ = fig3
    assert ns["INIT_SPEED"] == 20.0
    assert_array_equal(ns["kf"].Q, defs["Q"])
    assert_array_equal(ns["kf"].R, defs["R"])


@pytest.mark.parametrize("mode", ["projection", "analytic", "nn", "baseline_check"])
def test_table3_ut_parameters(golden_run, fig3, mode):
    """表 3、第 2.3 節：L = 6、α = 0.1、β = 2、κ_UT = 0，13 個 Merwe Sigma 點。"""
    for points in (golden_run.ukf[mode].points_fn, fig3[0]["points"]):
        assert (points.n, points.alpha, points.beta, points.kappa) == (6, 0.1, 2.0, 0.0)
        assert points.num_sigmas() == 13


def test_table3_prior_parameters():
    """表 3：η_v = 0.02、η_λ = 0.06、打滑率範圍 [0, 0.4]、曲率 0.002（存入模型檔供主程式使用）。"""
    train = load_training()
    assert (train["ALPHA"], train["BLEND"], train["LAMBDA_MAX"], train["KAPPA"]) == \
        (0.02, 0.06, 0.4, 0.002)


@pytest.mark.parametrize("path", [PAPER_FILE, TRAIN_FILE])
def test_table3_feature_ranges(path):
    """表 3、式 (53)、(55)：曲率速度範圍 [10, 30] m/s、r_o 範圍 [−0.30, 0.60]（b_cal = 0）。"""
    assert clip_bounds(path, "/ KAPPA") == {(10, 30)}
    assert clip_bounds(path, "/ curve_speed - 1") == {(-0.30, 0.60)}


def test_table3_initial_curve_speed_and_prior():
    """表 3、第 3.5 節：曲率參考速度初值 v̄κ,0 = 20 m/s；第一步的打滑率先驗為 0。"""
    assert statement(PAPER_FILE, "(slip_prior, curve_speed)") == "slip_prior, curve_speed = (0.0, 20.0)"


def test_figure3_output_matches_section_4_2_noise_model(fig3):
    """式 (66)–(67)：z_o = v(1+λ) + ν_o、z_κ = vκ + ν_κ（真實偏差為零）。"""
    ns, _ = fig3
    truth, z = ns["truth"], ns["measurements"]
    assert np.all(np.abs(z[:, 0] - truth[:, 1] * (1 + truth[:, 3])) < 5 * 0.4)
    assert np.all(np.abs(z[:, 1] - truth[:, 1] * 0.002) < 5 * 0.01)


# ---------- 第 3.5 節：MLP 與訓練資料 ----------

def _mlp_kwargs():
    tree = ast.parse(TRAIN_FILE.read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MLPRegressor")
    train = load_training()
    return {kw.arg: (train[kw.value.id] if isinstance(kw.value, ast.Name) else ast.literal_eval(kw.value))
            for kw in call.keywords}


def test_section_3_5_mlp_configuration():
    """3–64–64–1、Tanh、Adam、學習率 0.001、批次 2,048、最多 150 epoch、L2 係數 1e-4、
    以 15% 樣本為驗證集的 early stopping（n_iter_no_change = 15、tol = 1e-4）。

    L2 係數與 tol 程式沿用 scikit-learn 預設值；此處確認預設值與論文相同。
    """
    from sklearn.neural_network import MLPRegressor
    params = MLPRegressor(**_mlp_kwargs()).get_params()
    expected = {"hidden_layer_sizes": (64, 64), "activation": "tanh", "solver": "adam",
                "learning_rate_init": 0.001, "batch_size": 2048, "max_iter": 150,
                "alpha": 1e-4, "early_stopping": True, "validation_fraction": 0.15,
                "n_iter_no_change": 15, "tol": 1e-4, "random_state": 42}
    assert {k: params[k] for k in expected} == expected


def test_section_4_1_parameter_count():
    """3–64–64–1 共 4,481 個參數；權重乘加 3×64 + 64×64 + 64×1 = 4,352（式 (64)）。"""
    layers = [3, *_mlp_kwargs()["hidden_layer_sizes"], 1]
    weights = sum(a * b for a, b in zip(layers, layers[1:]))
    assert weights == 4352
    assert weights + sum(layers[1:]) == 4481


def test_section_3_5_dataset_sizes():
    """seed = 1 產生 800 條訓練／驗證序列（479,200 筆），seed = 99 產生 200 條測試序列（119,800 筆）。"""
    train = load_training()
    assert (train["TRAIN_SEED"], train["TEST_SEED"], train["MODEL_SEED"]) == (1, 99, 42)
    assert train["N_TRAIN"] * (train["STEPS"] - 1) == 479_200
    assert train["N_TEST"] * (train["STEPS"] - 1) == 119_800


@pytest.mark.parametrize("target, source", [
    ("speed[k]", "speed[k] = np.clip(speed[k - 1] + acceleration[k - 1] * DT, 15, 25)"),  # 15–25 m/s
    ("peak", "peak = rng.uniform(0.06, 0.3)"),                                             # 峰值 0.06–0.30
    ("odometer", "odometer = speed * (1 + slip) + rng.normal(0, SIGMA_ODO, STEPS)"),       # σ_odo
])
def test_section_3_5_training_data_settings(target, source):
    assert statement(TRAIN_FILE, target) == source


def test_section_3_5_about_15_percent_without_slip():
    """約 15% 序列不含打滑：只有 rng.random() > 0.15 時才產生打滑事件。"""
    tree = ast.parse(TRAIN_FILE.read_text(encoding="utf-8"))
    conditions = [ast.unparse(node.test) for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert "rng.random() > 0.15" in conditions


def test_eq59_label_is_absolute_next_slip():
    """式 (59)：訓練標籤是下一步打滑率的絕對值 λ_{k+1}（不是 Δλ）。"""
    tree = ast.parse(TRAIN_FILE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "make_sequence")
    assert ast.unparse(function.body[-1]) == "return (features[:-1], slip[1:])"
