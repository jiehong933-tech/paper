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


# 基本設定
DT, STEPS = 0.1, 600
KAPPA = 0.002
SIGMA_ODO, SIGMA_GYRO = 0.4, 0.01
ALPHA, BLEND, LAMBDA_MAX = 0.02, 0.06, 0.4

N_TRAIN, N_TEST = 800, 200
TRAIN_SEED, TEST_SEED, MODEL_SEED = 1, 99, 42

# 避免覆蓋論文原本使用的模型
MODEL_FILE = "nn_gyro_prior_retrained.joblib"
FIGURE_NAME = "NN_training_gyro_prior_retrained"


# 設定中文字體
def set_chinese_font():
    font_path = Path(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )

    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(
            fname=str(font_path)
        ).get_name()
        plt.rcParams["font.family"] = font_name
    else:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft JhengHei",
            "DejaVu Sans",
        ]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


# 產生一條模擬軌跡
def make_sequence(rng, augment=False):
    speed = np.zeros(STEPS)
    acceleration = np.zeros(STEPS)
    slip = np.zeros(STEPS)

    speed[0] = rng.uniform(15, 25)

    # 產生速度與加速度
    for k in range(1, STEPS):
        acceleration[k] = np.clip(
            0.96 * acceleration[k - 1]
            + rng.normal(0, 0.025),
            -0.35,
            0.35,
        )

        speed[k] = np.clip(
            speed[k - 1]
            + acceleration[k - 1] * DT,
            15,
            25,
        )

    # 隨機產生打滑事件
    if rng.random() > 0.15:
        start = rng.integers(60, 190)
        rise = rng.integers(1, 25)
        hold = rng.integers(60, 180)
        recovery = rng.integers(1, 100)
        peak = rng.uniform(0.06, 0.30)

        rise_end = min(start + rise, STEPS)
        hold_end = min(rise_end + hold, STEPS)
        recovery_end = min(hold_end + recovery, STEPS)

        slip[start:rise_end] = np.linspace(
            0,
            peak,
            rise_end - start,
            endpoint=False,
        )

        slip[rise_end:hold_end] = peak

        slip[hold_end:recovery_end] = np.linspace(
            peak,
            0,
            recovery_end - hold_end,
            endpoint=False,
        )

    # 模擬里程計與陀螺儀
    odometer = (
        speed * (1 + slip)
        + rng.normal(0, SIGMA_ODO, STEPS)
    )

    gyro_bias = rng.normal(0, 0.0004)

    gyro = (
        speed * KAPPA
        + gyro_bias
        + rng.normal(0, SIGMA_GYRO, STEPS)
    )

    # 曲率速度參考
    curve_speed = np.zeros(STEPS)

    curve_speed[0] = np.clip(
        speed[0] + rng.normal(0, 0.5),
        10,
        30,
    )

    for k in range(STEPS):
        if k == 0:
            old_speed = curve_speed[0]
        else:
            old_speed = curve_speed[k - 1]

        raw_speed = np.clip(
            gyro[k] / KAPPA,
            10,
            30,
        )

        curve_speed[k] = (
            (1 - ALPHA) * old_speed
            + ALPHA * raw_speed
        )

    # 模擬目前的打滑率估計
    estimated_slip = np.zeros(STEPS)
    gain = rng.uniform(0.10, 1.10)

    for k in range(1, STEPS):
        estimated_slip[k] = np.clip(
            0.90 * estimated_slip[k - 1]
            + 0.10 * gain * slip[k]
            + rng.normal(0, 0.020),
            0,
            LAMBDA_MAX,
        )

    # 訓練資料擴增
    if augment:
        bad = rng.random(STEPS) < 0.25
        estimated_slip[bad] = rng.uniform(
            0,
            LAMBDA_MAX,
            bad.sum(),
        )

    # 里程計與曲率速度之差
    innovation = np.clip(
        odometer / curve_speed - 1,
        -0.30,
        0.60,
    )

    # 三個輸入：曲率速度、目前打滑率、里程計差異
    features = np.column_stack([
        curve_speed[:-1],
        estimated_slip[:-1],
        innovation[:-1],
    ])

    # 預測下一個取樣點的真實打滑率
    target = slip[1:]

    return features, target


# 建立完整資料集
def make_dataset(number, seed, augment=False):
    rng = np.random.default_rng(seed)

    sequences = [
        make_sequence(rng, augment)
        for _ in range(number)
    ]

    x = np.vstack([
        sequence[0]
        for sequence in sequences
    ])

    y = np.hstack([
        sequence[1]
        for sequence in sequences
    ])

    return x, y


# 執行一次神經網路預測
def predict_once(
    model,
    scaler,
    curve_speed,
    slip_estimate,
    innovation,
):
    x = np.array([[
        np.clip(curve_speed, 10, 30),
        np.clip(slip_estimate, 0, LAMBDA_MAX),
        np.clip(innovation, -0.30, 0.60),
    ]])

    prediction = model.predict(
        scaler.transform(x)
    )[0]

    return np.clip(
        prediction,
        0,
        LAMBDA_MAX,
    )


# 自由遞迴測試
def rollout(model, scaler, innovation):
    slip = 0.30
    history = [slip]

    for _ in range(150):
        prior = predict_once(
            model,
            scaler,
            20.0,
            slip,
            innovation,
        )

        slip = (
            (1 - BLEND) * slip
            + BLEND * prior
        )

        history.append(slip)

    return np.array(history)


def main():
    set_chinese_font()

    # 建立資料
    x_train, y_train = make_dataset(
        N_TRAIN,
        TRAIN_SEED,
        augment=True,
    )

    x_test, y_test = make_dataset(
        N_TEST,
        TEST_SEED,
    )

    # 標準化輸入
    scaler = StandardScaler()

    x_train_n = scaler.fit_transform(x_train)
    x_test_n = scaler.transform(x_test)

    # 建立 MLP
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=2048,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=MODEL_SEED,
    )

    # 訓練
    model.fit(x_train_n, y_train)

    # 測試
    prediction = np.clip(
        model.predict(x_test_n),
        0,
        LAMBDA_MAX,
    )

    active = y_test > 0.01

    mae = mean_absolute_error(
        y_test,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction,
        )
    )

    active_rmse = np.sqrt(
        mean_squared_error(
            y_test[active],
            prediction[active],
        )
    )

    r2 = r2_score(
        y_test,
        prediction,
    )

    # 自由遞迴測試
    innovations = [0.0, 0.1, 0.2]

    histories = [
        rollout(model, scaler, value)
        for value in innovations
    ]

    final_values = [
        history[-1]
        for history in histories
    ]

    # 印出結果
    print(
        f"訓練資料：{x_train.shape}；"
        f"測試資料：{x_test.shape}"
    )
    print(f"Epoch：{model.n_iter_}")
    print(f"MAE：{mae:.5f}")
    print(f"RMSE：{rmse:.5f}")
    print(f"打滑區 RMSE：{active_rmse:.5f}")
    print(f"R²：{r2:.4f}")

    for value, answer in zip(
        innovations,
        final_values,
    ):
        print(
            f"r_o={value:.1f} "
            f"-> lambda={answer:.3f}"
        )

    # 基本合理性檢查
    if active_rmse > 0.06:
        raise RuntimeError(
            "NN 的獨立測試誤差過高"
        )

    if abs(final_values[0]) > 0.02:
        raise RuntimeError(
            "無打滑時無法回到零附近"
        )

    if (
        abs(final_values[1] - 0.10) > 0.04
        or abs(final_values[2] - 0.20) > 0.04
    ):
        raise RuntimeError(
            "自由遞迴穩態結果不合理"
        )

    # 儲存模型與版本資訊
    joblib.dump(
        {
            "model_type": "gyro_curve_slip_prior_v2",
            "run_label":
                "retrained_current_code_not_paper_epoch40",
            "model": model,
            "scaler": scaler,
            "alpha": ALPHA,
            "blend": BLEND,
            "kappa": KAPPA,
            "lambda_max": LAMBDA_MAX,
            "train_seed": TRAIN_SEED,
            "test_seed": TEST_SEED,
            "model_seed": MODEL_SEED,
            "epoch": model.n_iter_,
            "test_mae": mae,
            "test_rmse": rmse,
            "active_rmse": active_rmse,
            "test_r2": r2,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
        },
        MODEL_FILE,
    )

    # 畫圖
    time = np.arange(151) * DT
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    ax[0].plot(
        model.loss_curve_,
        lw=2,
        label="訓練損失",
    )

    ax[0].set_yscale("log")
    ax[0].set_title(
        "(a) MLP 網路訓練收斂曲線"
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("訓練損失")

    labels = [
        r"$r_o=0.0$（打滑結束）",
        r"$r_o=0.1$",
        r"$r_o=0.2$",
    ]

    for history, label in zip(
        histories,
        labels,
    ):
        ax[1].plot(
            time,
            history,
            lw=2,
            label=label,
        )

    ax[1].set_title(
        r"(b) 固定 $r_o$ 下的打滑率先驗自由遞迴測試"
    )
    ax[1].set_xlabel("時間（s）")
    ax[1].set_ylabel(
        r"打滑率 $\lambda$"
    )

    for graph in ax:
        graph.grid(
            linestyle=":",
            alpha=0.7,
        )
        graph.legend()

    plt.tight_layout()

    fig.savefig(
        f"{FIGURE_NAME}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        f"{FIGURE_NAME}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"\n已輸出：{MODEL_FILE}、"
        f"{FIGURE_NAME}.pdf、"
        f"{FIGURE_NAME}.png"
    )


if __name__ == "__main__":
    main()
