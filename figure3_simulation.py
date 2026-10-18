from pathlib import Path

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

# 輸出到程式所在資料夾（Colab 為目前資料夾），與 paper_code.py 讀取的位置相同。
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# 中文字體（Colab 有 Noto CJK；其他環境改用系統字型，避免找不到檔案而中斷）
font_file = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font_file.exists():
    fm.fontManager.addfont(str(font_file))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_file)).get_name()
else:  # 其他環境：改用已安裝的中文字型（Windows 正黑體、macOS 蘋方／黑體、Linux Noto／文泉驛）
    installed = {font.name for font in fm.fontManager.ttflist}
    chinese = [name for name in ("Microsoft JhengHei", "PingFang TC", "Heiti TC", "Noto Sans CJK TC",
                                 "Noto Sans CJK JP", "WenQuanYi Zen Hei") if name in installed]
    if chinese:
        plt.rcParams["font.family"] = chinese[0]
    else:
        print("警告：找不到中文字型，圖中的中文會顯示為方塊。")
plt.rcParams["axes.unicode_minus"] = False


# 狀態模型與量測模型
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


def hx(x):
    return np.array([x[V] * (1 + x[SLIP]), x[V] * KAPPA + x[BG]])


def rmse(estimate, reference):
    return np.sqrt(np.mean((estimate - reference) ** 2))


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
    measured_odo = true_x[V] * (1 + true_slip) + rng.normal(0, SIGMA_ODO)

    truth[k] = true_x
    measurements[k] = [measured_odo, measured_gyro]
    inputs[k] = [0.0, measured_gyro]


# 執行基準 UKF
points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2.0, kappa=0.0)
kf = UKF(6, 2, DT, hx, fx, points)
kf.x = np.array([0.0, INIT_SPEED, 0.0, 0.0, 0.0, 0.0])
kf.P = np.eye(6) * 0.1
kf.Q = np.eye(6) * 1e-4
kf.R = np.diag([SIGMA_ODO**2, SIGMA_GYRO**2])
ukf_result = np.zeros((STEPS, 6))

for k in range(STEPS):
    kf.predict(u=inputs[k])
    # 使用已包含 Q 的先驗共變異數，重新生成觀測更新用 Sigma 點
    kf.sigmas_f = points.sigma_points(kf.x, kf.P)
    kf.update(measurements[k])
    ukf_result[k] = kf.x


# 畫圖 3
step = np.arange(STEPS)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(step, truth[:, V], "g-", lw=2, label="真實車速")
ax[0].plot(step, measurements[:, 0], "r.", alpha=0.3, label="純里程計")
ax[0].plot(step, ukf_result[:, V], "b--", lw=1.5, label="基準 UKF")
ax[0].set(title=r"(a) 速度估測（$v$）", ylabel="速度（m/s）")
ax[0].legend(loc="lower left")

ax[1].plot(step, truth[:, SLIP], "g-", lw=2, label="真實打滑率")
ax[1].plot(step, ukf_result[:, SLIP], "b--", lw=1.5, label="基準 UKF")
ax[1].set(title=r"(b) 打滑率估測（$\lambda$）", ylabel=r"打滑率 $\lambda$",
          xlabel=r"取樣步 $k$（$\Delta t=0.1\ \mathrm{s}$）")
ax[1].legend(loc="upper left")
for graph in ax:
    graph.grid(linestyle=":", alpha=0.7)

plt.tight_layout()
fig.savefig(BASE_DIR / "simulation_results_tw.pdf", bbox_inches="tight")
fig.savefig(BASE_DIR / "simulation_results_tw.png", dpi=200, bbox_inches="tight")
plt.show()


# 計算 RMSE
slip_range = slice(SLIP_START, SLIP_END)
recovery_range = slice(SLIP_END, STEPS)
print(f"里程計打滑區間 RMSE：{rmse(measurements[slip_range, 0], truth[slip_range, V]):.3f}")
print(f"UKF 打滑區間 RMSE：{rmse(ukf_result[slip_range, V], truth[slip_range, V]):.3f}")
print(f"UKF 恢復區間 RMSE：{rmse(ukf_result[recovery_range, V], truth[recovery_range, V]):.3f}")


# 儲存圖 4 共用資料
np.savez_compressed(
    BASE_DIR / "figure3_shared_data.npz",
    step=step, truth=truth, measurement=measurements, imu_input=inputs,
    ukf_result=ukf_result, true_speed=TRUE_SPEED, ukf_init_speed=INIT_SPEED,
)
print(f"\n已輸出圖檔及 figure3_shared_data.npz\n輸出位置：{BASE_DIR}")
