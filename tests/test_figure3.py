"""figure3_simulation.py：真實資料產生、基準 UKF，以及與 paper_code.py 的銜接。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from paper_harness import load_training, module_assign, PAPER_FILE, FIG3_FILE


def test_truth_constant_speed_and_slip_window(fig3):
    ns, _ = fig3
    truth, V, SLIP = ns["truth"], ns["V"], ns["SLIP"]
    assert truth.shape == (600, 6)
    assert np.all(truth[:, V] == 20.0)                 # 加速度 0、偏差 0 → 等速
    expected_slip = np.where((np.arange(600) >= 200) & (np.arange(600) < 400), 0.2, 0.0)
    assert_array_equal(truth[:, SLIP], expected_slip)
    assert np.all(truth[:, [ns["BA"], ns["BG"]]] == 0.0)


def test_truth_is_state_after_propagation(fig3):
    """truth[k] 是傳遞一步後的狀態（s 從 2.0 開始），與濾波器「先預測再更新」的時間對齊。"""
    ns, _ = fig3
    truth = ns["truth"]
    assert_allclose(truth[:, ns["S"]], 2.0 * np.arange(1, 601), rtol=1e-12)
    assert_allclose(truth[:, ns["YAW"]], 20.0 * 0.002 * 0.1 * np.arange(1, 601), rtol=1e-9)


def test_measurement_model_and_noise(fig3):
    ns, _ = fig3
    truth, z, u, V, SLIP = ns["truth"], ns["measurements"], ns["inputs"], ns["V"], ns["SLIP"]
    odo_noise = z[:, 0] - truth[:, V] * (1 + truth[:, SLIP])
    gyro_noise = z[:, 1] - truth[:, V] * ns["KAPPA"]
    # 600 個樣本：標準差估計約有 ±3% 的抽樣誤差，容許 ±15%
    assert abs(odo_noise.std() / 0.4 - 1) < 0.15 and abs(odo_noise.mean()) < 0.1
    assert abs(gyro_noise.std() / 0.01 - 1) < 0.15 and abs(gyro_noise.mean()) < 0.003


def test_gyro_input_is_same_sample_as_gyro_measurement(fig3):
    """記錄行為：陀螺儀同一筆量測同時當作輸入 u 與量測 z；加速度計輸入恆為 0。"""
    ns, _ = fig3
    assert_array_equal(ns["inputs"][:, 1], ns["measurements"][:, 1])
    assert np.all(ns["inputs"][:, 0] == 0.0)


def test_reproducible_with_seed(fig3, tmp_path):
    from paper_harness import run_figure3
    again, _ = run_figure3(tmp_path)
    assert_array_equal(again["measurements"], fig3[0]["measurements"])
    assert_array_equal(again["ukf_result"], fig3[0]["ukf_result"])


def test_printed_rmse_matches(fig3):
    ns, stdout = fig3
    rmse, truth, V = ns["rmse"], ns["truth"], ns["V"]
    sp, rp = ns["slip_range"], ns["recovery_range"]
    values = [rmse(ns["measurements"][sp, 0], truth[sp, V]),
              rmse(ns["ukf_result"][sp, V], truth[sp, V]),
              rmse(ns["ukf_result"][rp, V], truth[rp, V])]
    for line, value in zip([l for l in stdout.splitlines() if "RMSE" in l], values):
        assert line.endswith(f"{value:.3f}")


def test_output_file_contents(fig3):
    ns, _ = fig3
    workdir = ns["BASE_DIR"]
    with np.load(workdir / "figure3_shared_data.npz") as data:
        assert set(data.files) >= {"step", "truth", "measurement", "imu_input", "ukf_result"}
        assert_array_equal(data["measurement"], ns["measurements"])
        assert_array_equal(data["ukf_result"], ns["ukf_result"])
    assert (workdir / "simulation_results_tw.pdf").exists()
    assert (workdir / "simulation_results_tw.png").exists()


def test_constants_match_other_programs(fig3, defs):
    ns, _ = fig3
    train = load_training()
    assert (ns["DT"], ns["STEPS"]) == (defs["DT"], train["STEPS"])
    assert (ns["SLIP_START"], ns["SLIP_END"]) == (defs["SLIP_START"], defs["SLIP_END"])
    assert ns["KAPPA"] == train["KAPPA"]
    assert (ns["SIGMA_ODO"], ns["SIGMA_GYRO"]) == (train["SIGMA_ODO"], train["SIGMA_GYRO"])
    assert_array_equal(ns["kf"].R, defs["R"])
    names = ("S", "V", "YAW", "SLIP", "BA", "BG")
    assert [ns[n] for n in names] == [defs[n] for n in names]


def test_output_folder_matches_paper_code():
    """圖 3 程式寫出資料的資料夾（BASE_DIR）與 paper_code.py 讀取的相同。"""
    import ast
    assert ast.dump(module_assign(FIG3_FILE, "BASE_DIR")) == ast.dump(module_assign(PAPER_FILE, "BASE_DIR"))


@pytest.mark.parametrize("state", ["S", "V", "YAW", "SLIP", "BA", "BG"])
def test_fx_hx_match_paper_code(fig3, defs, state):
    """圖 3 的 fx/hx（無打滑率先驗）與 paper_code.py 的 fx(slip_prior=None)/hx 相同。"""
    ns, _ = fig3
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.normal(size=6)
        u = rng.normal(size=2)
        i = ns[state]
        assert ns["fx"](x, 0.1, u)[i] == defs["fx"](x, 0.1, u)[i]
        assert_array_equal(ns["hx"](x), defs["hx"](x))


def test_paper_code_rebuilds_figure3_baseline_exactly(real_run):
    """paper_code.py 的 baseline_check 與圖 3 的基準 UKF 完全相同（max_gap = 0）。"""
    assert real_run.namespace["max_gap"] == 0.0
    assert "警告" not in real_run.stdout
