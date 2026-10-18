import platform
from pathlib import Path

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# 1. 基本設定
DT, STEPS = 0.1, 600
KAPPA = 0.002
SIGMA_ODO, SIGMA_GYRO = 0.4, 0.01
ALPHA, BLEND, LAMBDA_MAX = 0.02, 0.06, 0.4
N_TRAIN, N_TEST = 800, 200
TRAIN_SEED, TEST_SEED, MODEL_SEED = 1, 99, 42

# 輸出到程式所在資料夾（Colab 為目前資料夾），與 paper_code.py 讀取的位置相同。
# 檔名避免覆蓋論文原本使用的模型。
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_FILE = BASE_DIR / "nn_gyro_prior_retrained.joblib"
FIGURE_FILE = BASE_DIR / "NN_training_gyro_prior_retrained"

font_file = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font_file.exists():
    fm.fontManager.addfont(str(font_file))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_file)).get_name()
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams.update({"axes.unicode_minus": False, "pdf.fonttype": 42})


# 2. 產生一條模擬軌跡，回傳（三個輸入特徵, 下一步的真實打滑率）
def make_sequence(rng, augment=False):
    speed, acceleration, slip = np.zeros(STEPS), np.zeros(STEPS), np.zeros(STEPS)
    speed[0] = rng.uniform(15, 25)
    for k in range(1, STEPS):
        acceleration[k] = np.clip(0.96 * acceleration[k - 1] + rng.normal(0, 0.025), -0.35, 0.35)
        speed[k] = np.clip(speed[k - 1] + acceleration[k - 1] * DT, 15, 25)

    # 隨機打滑事件：上升 → 維持 → 恢復
    if rng.random() > 0.15:
        start = rng.integers(60, 190)
        rise_end = min(start + rng.integers(1, 25), STEPS)
        hold_end = min(rise_end + rng.integers(60, 180), STEPS)
        recovery_end = min(hold_end + rng.integers(1, 100), STEPS)
        peak = rng.uniform(0.06, 0.30)
        slip[start:rise_end] = np.linspace(0, peak, rise_end - start, endpoint=False)
        slip[rise_end:hold_end] = peak
        slip[hold_end:recovery_end] = np.linspace(peak, 0, recovery_end - hold_end, endpoint=False)

    # 里程計與陀螺儀（含一個固定的陀螺儀偏差）
    odometer = speed * (1 + slip) + rng.normal(0, SIGMA_ODO, STEPS)
    gyro = speed * KAPPA + rng.normal(0, 0.0004) + rng.normal(0, SIGMA_GYRO, STEPS)

    # 曲率速度參考：陀螺儀換算的速度經指數平滑
    curve_speed = np.zeros(STEPS)
    previous = np.clip(speed[0] + rng.normal(0, 0.5), 10, 30)
    for k in range(STEPS):
        previous = curve_speed[k] = (1 - ALPHA) * previous + ALPHA * np.clip(gyro[k] / KAPPA, 10, 30)

    # 模擬濾波器目前的打滑率估計
    estimated_slip = np.zeros(STEPS)
    gain = rng.uniform(0.10, 1.10)
    for k in range(1, STEPS):
        estimated_slip[k] = np.clip(0.90 * estimated_slip[k - 1] + 0.10 * gain * slip[k]
                                    + rng.normal(0, 0.020), 0, LAMBDA_MAX)
    if augment:  # 訓練資料擴增：隨機替換 25% 的打滑率估計
        bad = rng.random(STEPS) < 0.25
        estimated_slip[bad] = rng.uniform(0, LAMBDA_MAX, bad.sum())

    # 三個輸入：曲率速度、目前打滑率、里程計與曲率速度之差
    innovation = np.clip(odometer / curve_speed - 1, -0.30, 0.60)
    features = np.column_stack([curve_speed, estimated_slip, innovation])
    return features[:-1], slip[1:]


# 3. 建立資料集並訓練 MLP
rng = np.random.default_rng(TRAIN_SEED)
x_train, y_train = map(np.concatenate, zip(*[make_sequence(rng, True) for _ in range(N_TRAIN)]))
rng = np.random.default_rng(TEST_SEED)
x_test, y_test = map(np.concatenate, zip(*[make_sequence(rng) for _ in range(N_TEST)]))

scaler = StandardScaler()
model = MLPRegressor(
    hidden_layer_sizes=(64, 64), activation="tanh", solver="adam",
    learning_rate_init=0.001, batch_size=2048, max_iter=150, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=15, random_state=MODEL_SEED,
)
model.fit(scaler.fit_transform(x_train), y_train)


# 4. 獨立測試資料評估
prediction = np.clip(model.predict(scaler.transform(x_test)), 0, LAMBDA_MAX)
active = y_test > 0.01
mae = mean_absolute_error(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
active_rmse = np.sqrt(mean_squared_error(y_test[active], prediction[active]))
r2 = r2_score(y_test, prediction)

# 自由遞迴測試：固定 r_o，打滑率從 0.30 開始反覆以 MLP 先驗混合 150 步
innovations = [0.0, 0.1, 0.2]
histories = []
for r_o in innovations:
    slip = 0.30
    history = [slip]
    for _ in range(150):
        features = [[20.0, np.clip(slip, 0, LAMBDA_MAX), r_o]]
        prior = np.clip(model.predict(scaler.transform(features))[0], 0, LAMBDA_MAX)
        slip = (1 - BLEND) * slip + BLEND * prior
        history.append(slip)
    histories.append(history)
final_values = [history[-1] for history in histories]

print(f"訓練資料：{x_train.shape}；測試資料：{x_test.shape}")
print(f"Epoch：{model.n_iter_}")
print(f"MAE：{mae:.5f}")
print(f"RMSE：{rmse:.5f}")
print(f"打滑區 RMSE：{active_rmse:.5f}")
print(f"R²：{r2:.4f}")
for r_o, value in zip(innovations, final_values):
    print(f"r_o={r_o:.1f} -> lambda={value:.3f}")

# 基本合理性檢查
if active_rmse > 0.06:
    raise RuntimeError("NN 的獨立測試誤差過高")
if abs(final_values[0]) > 0.02:
    raise RuntimeError("無打滑時無法回到零附近")
if abs(final_values[1] - 0.10) > 0.04 or abs(final_values[2] - 0.20) > 0.04:
    raise RuntimeError("自由遞迴穩態結果不合理")


# 5. 儲存模型與版本資訊（paper_code.py 會讀取 model、scaler 與四個參數）
joblib.dump({
    "model_type": "gyro_curve_slip_prior_v2",
    "run_label": "retrained_current_code_not_paper_epoch40",
    "model": model, "scaler": scaler,
    "alpha": ALPHA, "blend": BLEND, "kappa": KAPPA, "lambda_max": LAMBDA_MAX,
    "train_seed": TRAIN_SEED, "test_seed": TEST_SEED, "model_seed": MODEL_SEED,
    "epoch": model.n_iter_, "test_mae": mae, "test_rmse": rmse,
    "active_rmse": active_rmse, "test_r2": r2,
    "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
}, MODEL_FILE)


# 6. 畫圖：(a) 訓練損失、(b) 自由遞迴測試
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(model.loss_curve_, lw=2, label="訓練損失")
ax[0].set(yscale="log", title="(a) MLP 網路訓練收斂曲線", xlabel="Epoch", ylabel="訓練損失")
time = np.arange(151) * DT
for history, label in zip(histories, [r"$r_o=0.0$（打滑結束）", r"$r_o=0.1$", r"$r_o=0.2$"]):
    ax[1].plot(time, history, lw=2, label=label)
ax[1].set(title=r"(b) 固定 $r_o$ 下的打滑率先驗自由遞迴測試",
          xlabel="時間（s）", ylabel=r"打滑率 $\lambda$")
for graph in ax:
    graph.grid(linestyle=":", alpha=0.7)
    graph.legend()
plt.tight_layout()
fig.savefig(f"{FIGURE_FILE}.pdf", bbox_inches="tight")
fig.savefig(f"{FIGURE_FILE}.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"\n已輸出：{MODEL_FILE.name}、{FIGURE_FILE.name}.pdf、{FIGURE_FILE.name}.png"
      f"\n輸出位置：{BASE_DIR}")
