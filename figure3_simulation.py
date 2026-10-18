import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


# 基本設定
DT, STEPS = 0.1, 600
KAPPA = 0.002
TRUE_SPEED = 20.0       # 真實速度，不要改
INIT_SPEED = 20.0       # UKF 初始速度

SLIP_START, SLIP_END = 200, 400
TRUE_SLIP = 0.2
SIGMA_ODO, SIGMA_GYRO = 0.4, 0.01
SEED = 42

S, V, YAW, SLIP, BA, BG = range(6)


# Colab 中文字體
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False


# 狀態模型
def fx(x, dt, u):
    acc, gyro = u
    return np.array([
        x[S] + x[V] * dt,
        x[V] + (acc - x[BA]) * dt,
        x[YAW] + (gyro - x[BG]) * dt,
        x[SLIP],
        x[BA],
        x[BG],
    ])


# 量測模型
def hx(x):
    return np.array([
        x[V] * (1 + x[SLIP]),
        x[V] * KAPPA + x[BG],
    ])


# 產生固定量測資料
rng = np.random.default_rng(SEED)
true_x = np.array([0.0, TRUE_SPEED, 0.0, 0.0, 0.0, 0.0])

truth = np.zeros((STEPS, 6))
measurements = np.zeros((STEPS, 2))
inputs = np.zeros((STEPS, 2))

for k in range(STEPS):
    true_slip = TRUE_SLIP if SLIP_START <= k < SLIP_END else 0.0
    true_gyro = true_x[V] * KAPPA
    measured_gyro = true_gyro + rng.normal(0, SIGMA_GYRO)

    true_x = fx(true_x, DT, [0.0, true_gyro])
    true_x[SLIP] = true_slip

    measured_odo = (
        true_x[V] * (1 + true_slip)
        + rng.normal(0, SIGMA_ODO)
    )

    truth[k] = true_x
    measurements[k] = [measured_odo, measured_gyro]
    inputs[k] = [0.0, measured_gyro]


# 執行基準 UKF
def run_ukf(init_speed):
    points = MerweScaledSigmaPoints(
        6, alpha=0.1, beta=2.0, kappa=0.0
    )
    kf = UKF(6, 2, DT, hx, fx, points)

    kf.x = np.array([0.0, init_speed, 0.0, 0.0, 0.0, 0.0])
    kf.P = np.eye(6) * 0.1
    kf.Q = np.eye(6) * 1e-4
    kf.R = np.diag([SIGMA_ODO**2, SIGMA_GYRO**2])

    result = np.zeros((STEPS, 6))

    for k in range(STEPS):
        kf.predict(u=inputs[k])

        # 使用已包含 Q 的先驗共變異數，重新生成觀測更新用 Sigma 點
        kf.sigmas_f = points.sigma_points(kf.x, kf.P)

        kf.update(measurements[k])
        result[k] = kf.x

    return result

ukf_result = run_ukf(INIT_SPEED)


# 畫圖 3
step = np.arange(STEPS)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(step, truth[:, V], "g-", lw=2, label="真實車速")
ax[0].plot(step, measurements[:, 0], "r.", alpha=0.3, label="純里程計")
ax[0].plot(step, ukf_result[:, V], "b--", lw=1.5, label="基準 UKF")
ax[0].set_title(r"(a) 速度估測（$v$）")
ax[0].set_ylabel("速度（m/s）")
ax[0].legend(loc="lower left")
ax[0].grid(linestyle=":", alpha=0.7)

ax[1].plot(step, truth[:, SLIP], "g-", lw=2, label="真實打滑率")
ax[1].plot(step, ukf_result[:, SLIP], "b--", lw=1.5, label="基準 UKF")
ax[1].set_title(r"(b) 打滑率估測（$\lambda$）")
ax[1].set_xlabel(r"取樣步 $k$（$\Delta t=0.1\ \mathrm{s}$）")
ax[1].set_ylabel(r"打滑率 $\lambda$")
ax[1].legend(loc="upper left")
ax[1].grid(linestyle=":", alpha=0.7)

plt.tight_layout()
fig.savefig("simulation_results_tw.pdf", bbox_inches="tight")
fig.savefig("simulation_results_tw.png", dpi=200, bbox_inches="tight")
plt.show()


# 計算 RMSE
def rmse(estimate, reference):
    return np.sqrt(np.mean((estimate - reference) ** 2))


slip_range = slice(SLIP_START, SLIP_END)
recovery_range = slice(SLIP_END, STEPS)

print(f"里程計打滑區間 RMSE："
      f"{rmse(measurements[slip_range, 0], truth[slip_range, V]):.3f}")
print(f"UKF 打滑區間 RMSE："
      f"{rmse(ukf_result[slip_range, V], truth[slip_range, V]):.3f}")
print(f"UKF 恢復區間 RMSE："
      f"{rmse(ukf_result[recovery_range, V], truth[recovery_range, V]):.3f}")


# 儲存圖 4 共用資料
np.savez_compressed(
    "figure3_shared_data.npz",
    step=step,
    truth=truth,
    measurement=measurements,
    imu_input=inputs,
    ukf_result=ukf_result,
    true_speed=TRUE_SPEED,
    ukf_init_speed=INIT_SPEED,
)

print("\n已輸出圖檔及 figure3_shared_data.npz")
