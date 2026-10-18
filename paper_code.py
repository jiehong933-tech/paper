import csv
from pathlib import Path

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF


# 1. 讀取共用資料、模型與中文字體
#    檔案放在程式所在資料夾（Colab 為目前資料夾），圖 3 資料也可放在 figure3_output/。
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_FILE = BASE_DIR / "nn_gyro_prior_retrained.joblib"   # 由 train_nn_gyro_prior.py 產生
data_file = BASE_DIR / "figure3_shared_data.npz"            # 由 figure3_simulation.py 產生
if not data_file.exists():
    data_file = BASE_DIR / "figure3_output/figure3_shared_data.npz"
if not data_file.exists():
    raise FileNotFoundError(f"找不到 figure3_shared_data.npz，請先執行圖 3 程式 figure3_simulation.py，"
                            f"並把檔案放在 {BASE_DIR} 或其 figure3_output/ 資料夾。")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"找不到 {MODEL_FILE.name}，請先執行 train_nn_gyro_prior.py"
                            f"（模型會存在該程式所在的資料夾），並把模型檔放在 {BASE_DIR}。")

with np.load(data_file) as data:
    step, truth = data["step"], data["truth"]
    measurements, inputs = data["measurement"], data["imu_input"]
    # 此結果由修正後的圖 3 產生。
    baseline_ukf = data["ukf_result"] if "ukf_result" in data.files else data["ukf_estimate"]

saved = joblib.load(MODEL_FILE)
if saved.get("model_type") != "gyro_curve_slip_prior_v2":
    raise ValueError("模型格式不符，請確認載入的是論文使用的模型。")
model, scaler = saved["model"], saved["scaler"]
ALPHA, BLEND = saved["alpha"], saved["blend"]
KAPPA, LAMBDA_MAX = saved["kappa"], saved["lambda_max"]
print(f"使用模型：{MODEL_FILE.name}")

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
plt.rcParams.update({"axes.unicode_minus": False, "pdf.fonttype": 42})


# 2. 共用參數與必要函式（fx、hx 供濾波器呼叫）
DT = 0.1
SLIP_START, SLIP_END = 200, 400
S, V, YAW, SLIP, BA, BG = range(6)
Q = np.eye(6) * 1e-4
R = np.diag([0.4 ** 2, 0.01 ** 2])
X0 = np.array([0.0, 20.0, 0.0, 0.0, 0.0, 0.0])
P0 = np.eye(6) * 0.1


def fx(x, dt, u, slip_prior=None):
    """狀態預測；u = [加速度計, 陀螺儀]。有打滑率先驗時以 BLEND 混合。"""
    s, v, yaw, slip, ba, bg = x
    predicted = np.array([s + v * dt, v + (u[0] - ba) * dt, yaw + (u[1] - bg) * dt,
                          slip, ba, bg])
    if slip_prior is not None:
        predicted[SLIP] = (1 - BLEND) * slip + BLEND * slip_prior
    return predicted


def hx(x):
    return np.array([x[V] * (1 + x[SLIP]), x[V] * KAPPA + x[BG]])


def rmse(estimate, reference):
    return np.sqrt(np.mean((estimate - reference) ** 2))


# 3. 標準 EKF
F = np.eye(6)
F[S, V], F[V, BA], F[YAW, BG] = DT, -DT, -DT
x, P, identity = X0.copy(), P0.copy(), np.eye(6)
ekf = np.zeros_like(truth)

for k, (z, u) in enumerate(zip(measurements, inputs)):
    x = fx(x, DT, u)
    P = F @ P @ F.T + Q
    H = np.zeros((2, 6))
    H[0, V], H[0, SLIP] = 1 + x[SLIP], x[V]
    H[1, V], H[1, BG] = KAPPA, 1
    gain = np.linalg.solve(H @ P @ H.T + R, H @ P).T
    x += gain @ (z - hx(x))
    I_KH = identity - gain @ H
    P = I_KH @ P @ I_KH.T + gain @ R @ gain.T  # Joseph 共變異數更新
    ekf[k] = x


# 4. UKF 版本：範圍投影、解析先驗、MLP 先驗，以及兩種只改偏差設定的對照組。
#    baseline_check 與圖 3 的基準 UKF 設定相同，用來確認對照組與基準 UKF
#    之間確實只差 P0／Q 的偏差那幾格。
ukf_results, prior_results = {}, {}
nn_curve_speed, nn_difference = np.zeros(len(step)), np.zeros(len(step))
points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2.0, kappa=0.0)

for mode in ("projection", "analytic", "nn",
             "baseline_check", "bias_known", "bias_constant"):
    use_prior = mode in ("analytic", "nn")               # 使用打滑率先驗
    use_clip = mode in ("projection", "analytic", "nn")  # 打滑率投影到 [0, LAMBDA_MAX]
    kf = UKF(6, 2, DT, hx, fx, points)
    kf.x, kf.P, kf.Q, kf.R = X0.copy(), P0.copy(), Q.copy(), R.copy()
    if mode == "bias_known":       # 偏差已知：與先驗支路相同，視陀螺儀偏差為已知且不變
        kf.P[BG, BG], kf.Q[BG, BG] = 1e-10, 1e-14
    elif mode == "bias_constant":  # 常數偏差：兩個偏差未知（P0 不變），但不隨時間變化
        kf.Q[BA, BA] = kf.Q[BG, BG] = 1e-14
    slip_prior, curve_speed = 0.0, 20.0
    result, priors = np.zeros_like(truth), np.zeros(len(step))

    for k, (z, u) in enumerate(zip(measurements, inputs)):
        kf.predict(u=u, slip_prior=slip_prior if use_prior else None)
        # 從已包含 Q 的先驗共變異數重新生成量測更新用 Sigma 點。
        kf.sigmas_f = points.sigma_points(kf.x, kf.P)
        kf.update(z)
        if use_clip:
            kf.x[SLIP] = np.clip(kf.x[SLIP], 0, LAMBDA_MAX)
        result[k] = kf.x

        if use_prior:
            # 更新後才計算下一步的先驗；校正後 gyro bias 設為零。
            curve_speed = (1 - ALPHA) * curve_speed + ALPHA * np.clip(z[1] / KAPPA, 10, 30)
            difference = np.clip(z[0] / curve_speed - 1, -0.30, 0.60)
            if mode == "analytic":
                slip_prior = np.clip(difference, 0, LAMBDA_MAX)
            else:
                features = [[np.clip(curve_speed, 10, 30),
                             np.clip(kf.x[SLIP], 0, LAMBDA_MAX), difference]]
                slip_prior = np.clip(model.predict(scaler.transform(features))[0],
                                     0, LAMBDA_MAX)
                nn_curve_speed[k], nn_difference[k] = curve_speed, difference
            priors[k] = slip_prior

    ukf_results[mode], prior_results[mode] = result, priors

projection_ukf = ukf_results["projection"]
analytic_ukf, analytic_prior = ukf_results["analytic"], prior_results["analytic"]
nn_ukf, nn_prior = ukf_results["nn"], prior_results["nn"]
bias_known_ukf = ukf_results["bias_known"]
bias_constant_ukf = ukf_results["bias_constant"]

# 檢查：本程式重建的基準 UKF 應與圖 3 讀入的基準 UKF 相同。
max_gap = np.max(np.abs(ukf_results["baseline_check"] - baseline_ukf))
print(f"\n基準 UKF 重建檢查：與圖 3 結果之最大差異 = {max_gap:.2e}")
if max_gap > 1e-6:
    print("警告：本程式的基準 UKF 與圖 3 的結果不一致，"
          "兩個對照組的數字請先不要放進論文。")


# 5. 主要比較與消融 RMSE
slip_part, recovery_part = slice(SLIP_START, SLIP_END), slice(SLIP_END, len(step))
main_methods = {
    "Raw Odometry": measurements[:, 0], "Standard EKF": ekf[:, V],
    "Baseline UKF": baseline_ukf[:, V], "NN-prior UKF": nn_ukf[:, V]
}
ablation_methods = {
    "基準 UKF（純隨機漫步）": baseline_ukf, "僅範圍投影 UKF": projection_ukf,
    "解析先驗 UKF": analytic_ukf, "MLP 先驗 UKF": nn_ukf,
    "偏差已知 UKF": bias_known_ukf, "常數偏差 UKF": bias_constant_ukf
}
for name, result in {"EKF": ekf, **ablation_methods}.items():
    if not np.isfinite(result).all():
        raise RuntimeError(f"{name} 出現 NaN 或無限大")

print("\n主要比較：速度 RMSE（m/s）")
print(f"{'演算法':<20}{'打滑區間':>12}{'恢復區間':>12}")
for name, speed in main_methods.items():
    a = rmse(speed[slip_part], truth[slip_part, V])
    b = rmse(speed[recovery_part], truth[recovery_part, V])
    print(f"{name:<20}{a:>12.3f}{b:>12.3f}")

print("\nUKF 消融：速度 RMSE（m/s）與打滑率 RMSE")
print(f"{'演算法':<24}{'速度/打滑':>12}{'速度/恢復':>12}"
      f"{'打滑率/打滑':>14}{'打滑率/恢復':>14}")
ablation_metrics = {}
for name, result in ablation_methods.items():
    ablation_metrics[name] = (
        rmse(result[slip_part, V], truth[slip_part, V]),
        rmse(result[recovery_part, V], truth[recovery_part, V]),
        rmse(result[slip_part, SLIP], truth[slip_part, SLIP]),
        rmse(result[recovery_part, SLIP], truth[recovery_part, SLIP])
    )
    print(f"{name:<24}" + "".join(f"{value:>12.3f}" for value in ablation_metrics[name]))


# 6. 圖 4、圖 5；每列依序為標籤、結果、顏色、線型、線寬。
main_styles = [
    ("標準 EKF", ekf, "orange", "-.", 1.5),
    ("基準 UKF", baseline_ukf, "red", "--", 1.5),
    ("NN 輔助 UKF（本研究）", nn_ukf, "blue", "-", 2)
]
ablation_styles = [
    ("基準 UKF", baseline_ukf, "red", "--", 1.5),
    ("僅範圍投影 UKF", projection_ukf, "purple", ":", 1.5),
    ("解析先驗 UKF", analytic_ukf, "green", "-.", 2),
    ("MLP 先驗 UKF", nn_ukf, "blue", "-", 2)
]
for filename, styles, is_main in [("Comparison_Fig_gyro_prior", main_styles, True),
                                  ("Ablation_Fig_gyro_prior", ablation_styles, False)]:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, state, true_label in zip(axes, (V, SLIP), ("真實車速", "真實打滑率")):
        ax.axvspan(SLIP_START, SLIP_END, color="gray", alpha=0.15)
        ax.grid(linestyle=":", alpha=0.7)
        ax.plot(step, truth[:, state], "k-", lw=2, label=true_label)
        if is_main and state == V:
            ax.plot(step, measurements[:, 0], "r.", alpha=0.12, label="純里程計")
        for label, estimate, color, line_style, width in styles:
            ax.plot(step, estimate[:, state], color=color, ls=line_style, lw=width, label=label)
    prefix = "" if is_main else "UKF 消融之"
    axes[0].set(title=f"(a) {prefix}速度估測 ($v$)", ylabel="速度 (m/s)")
    axes[1].set(title=rf"(b) {prefix}打滑率估測 ($\lambda$)", ylabel=r"打滑率 $\lambda$",
                xlabel=r"取樣步 $k$（$\Delta t=0.1\ \mathrm{s}$）")
    axes[0].legend(loc="lower left")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(BASE_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(BASE_DIR / f"{filename}.png", dpi=200, bbox_inches="tight")


# 7. 匯出結果；保留原本的檔名、NPZ 欄位及 CSV 欄位順序。
np.savez_compressed(
    BASE_DIR / "figure4_gyro_results_with_ablation.npz",
    ekf=ekf, baseline_ukf=baseline_ukf, projection_ukf=projection_ukf,
    analytic_ukf=analytic_ukf, nn_ukf=nn_ukf,
    analytic_prior=analytic_prior, nn_prior=nn_prior,
    curve_speed=nn_curve_speed, normalized_difference=nn_difference
)
columns = {  # 欄位名稱: 資料（順序即 CSV 欄位順序）
    "step": step, "true_speed": truth[:, V], "true_slip": truth[:, SLIP],
    "odometer_speed": measurements[:, 0],
    "ekf_speed": ekf[:, V], "ekf_slip": ekf[:, SLIP],
    "baseline_ukf_speed": baseline_ukf[:, V], "baseline_ukf_slip": baseline_ukf[:, SLIP],
    "projection_ukf_speed": projection_ukf[:, V], "projection_ukf_slip": projection_ukf[:, SLIP],
    "analytic_ukf_speed": analytic_ukf[:, V], "analytic_ukf_slip": analytic_ukf[:, SLIP],
    "nn_ukf_speed": nn_ukf[:, V], "nn_ukf_slip": nn_ukf[:, SLIP],
    "analytic_prior": analytic_prior, "nn_prior": nn_prior,
    "curve_speed": nn_curve_speed, "normalized_difference": nn_difference,
}
np.savetxt(BASE_DIR / "figure4_gyro_results_with_ablation.csv",
           np.column_stack(list(columns.values())),
           delimiter=",", header=",".join(columns), comments="")
with open(BASE_DIR / "ablation_rmse.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["演算法", "速度RMSE_打滑區間_m_s", "速度RMSE_恢復區間_m_s",
                     "打滑率RMSE_打滑區間", "打滑率RMSE_恢復區間"])
    for name, values in ablation_metrics.items():
        writer.writerow([name, *(f"{value:.6f}" for value in values)])
with open(BASE_DIR / "ablation_rmse_latex.txt", "w", encoding="utf-8") as file:
    for name, values in ablation_metrics.items():
        file.write(f"{name} & " + " & ".join(f"{value:.3f}" for value in values) + r" \\" + "\n")

print(f"\n已完成：圖 4、圖 5（PDF／PNG）、結果 NPZ／CSV 與消融 RMSE 表。\n輸出位置：{BASE_DIR}")
plt.show()
