"""合成資料與假模型，用來取代 repo 中沒有的資料檔與 MLP 權重。

合成資料依 `paper_code.py` 的模型產生（狀態順序同論文式 (30)）：
    x = [s, v, ψ, λ, b_a, b_g]
    u = [a_meas, ω_meas]，a_meas = a + b_a + 雜訊，ω_meas = vκ + b_g + 雜訊
    z = [v(1+λ) + 雜訊(σ=0.4), vκ + b_g + 雜訊(σ=0.01)]   （同 hx 與 R）
打滑區間與程式的 SLIP_START、SLIP_END（200、400）一致。
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

SEED = 20240601
N_STEPS = 600
DT = 0.1
# 與 train_nn_gyro_prior.py 相同的參數（論文模型檔中儲存的值）
KAPPA = 0.002
ALPHA, BLEND, LAMBDA_MAX = 0.02, 0.06, 0.4
SLIP_PEAK = 0.30   # 訓練資料打滑峰值上限
TRUE_BA, TRUE_BG = 0.05, 0.0004


def make_truth(n_steps=N_STEPS, slip_start=200, slip_end=400, kappa=KAPPA):
    step = np.arange(n_steps)
    acceleration = np.zeros(n_steps)
    acceleration[150:200] = -1.2          # 20 → 14 m/s
    acceleration[450:500] = +1.2          # 14 → 20 m/s
    speed = 20.0 + np.concatenate([[0.0], np.cumsum(acceleration[:-1]) * DT])
    slip = np.zeros(n_steps)
    k = np.arange(slip_start, slip_end)
    slip[slip_start:slip_end] = SLIP_PEAK * np.sin(np.pi * (k - slip_start)
                                              / (slip_end - slip_start))
    distance = np.concatenate([[0.0], np.cumsum(speed[:-1]) * DT])
    yaw = np.concatenate([[0.0], np.cumsum(speed[:-1] * kappa) * DT])
    truth = np.column_stack([distance, speed, yaw, slip,
                             np.full(n_steps, TRUE_BA), np.full(n_steps, TRUE_BG)])
    return step, truth, acceleration


def make_scenario(seed=SEED, n_steps=N_STEPS, kappa=KAPPA):
    rng = np.random.default_rng(seed)
    step, truth, acceleration = make_truth(n_steps, kappa=kappa)
    v, slip = truth[:, 1], truth[:, 3]
    imu_input = np.column_stack([
        acceleration + TRUE_BA + rng.normal(0, 0.02, n_steps),
        v * kappa + TRUE_BG + rng.normal(0, 0.01, n_steps),
    ])
    measurement = np.column_stack([
        v * (1 + slip) + rng.normal(0, 0.4, n_steps),
        v * kappa + TRUE_BG + rng.normal(0, 0.01, n_steps),
    ])
    scenario = {"step": step, "truth": truth, "measurement": measurement,
                "imu_input": imu_input}
    scenario["baseline_ukf"] = reference_baseline_ukf(scenario, kappa=kappa)
    return scenario


def reference_baseline_ukf(scenario, kappa=KAPPA):
    """與圖 3 基準 UKF 相同設定的獨立實作（不經過 `paper_code.py` 的迴圈）。

    用於產生合成資料中的 `ukf_result`，讓 `paper_code.py` 的
    baseline_check 有可比對的對象。
    """
    from paper_harness import load_definitions

    d = load_definitions(alpha=ALPHA, blend=BLEND, kappa=kappa,
                         lambda_max=LAMBDA_MAX, n_steps=len(scenario["step"]))
    points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2.0, kappa=0.0)
    kf = UnscentedKalmanFilter(6, 2, d["DT"], d["hx"], d["fx"], points)
    kf.x, kf.P, kf.Q, kf.R = d["X0"].copy(), d["P0"].copy(), d["Q"].copy(), d["R"].copy()
    result = np.zeros_like(scenario["truth"])
    for k, (z, u) in enumerate(zip(scenario["measurement"], scenario["imu_input"])):
        kf.predict(u=u, slip_prior=None)
        kf.sigmas_f = points.sigma_points(kf.x, kf.P)
        kf.update(z)
        result[k] = kf.x
    return result


from paper_harness import IdentityScaler  # noqa: E402,F401  （沿用同一個類別）


class RecordingModel:
    """假 MLP：輸出 = features @ weights + bias，並記錄每次 predict 的輸入。

    特徵順序同 `paper_code.py`：[curve_speed, λ_post(clip), difference]。
    """

    def __init__(self, weights=(0.0, 0.0, 0.0), bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.calls = []

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        self.calls.append(features.copy())
        return features @ self.weights + self.bias


def zero_model():
    """恆輸出 0 的假模型。"""
    return RecordingModel()


def analytic_equivalent_model():
    """輸出 = difference（第 3 個特徵），經 clip 後與解析先驗完全相同。"""
    return RecordingModel(weights=(0.0, 0.0, 1.0))


def golden_model():
    """合成回歸基準使用的固定假模型（不訓練）。"""
    return RecordingModel(weights=(0.0, 0.3, 0.7))
