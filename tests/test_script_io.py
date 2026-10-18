"""`paper_code.py` 整體流程：資料／模型讀取、基準 UKF 重建檢查與輸出檔格式。"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import synthetic
from paper_harness import model_payload, run_script, write_data

CSV_HEADER = (
    "step,true_speed,true_slip,odometer_speed,ekf_speed,ekf_slip,"
    "baseline_ukf_speed,baseline_ukf_slip,projection_ukf_speed,"
    "projection_ukf_slip,analytic_ukf_speed,analytic_ukf_slip,"
    "nn_ukf_speed,nn_ukf_slip,analytic_prior,nn_prior,"
    "curve_speed,normalized_difference"
)


def _payload(**kw):
    return model_payload(synthetic.golden_model(), synthetic.IdentityScaler(), **kw)


def test_model_parameters_are_read_from_payload(golden_run):
    ns = golden_run.namespace
    assert (ns["ALPHA"], ns["BLEND"], ns["KAPPA"], ns["LAMBDA_MAX"]) == \
        (synthetic.ALPHA, synthetic.BLEND, synthetic.KAPPA, synthetic.LAMBDA_MAX)


def test_wrong_model_type_is_rejected(tmp_path, scenario):
    write_data(tmp_path, scenario)
    with pytest.raises(ValueError, match="模型格式不符"):
        run_script(tmp_path, _payload(model_type="something_else"))


def test_missing_data_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="figure3_shared_data.npz.*圖 3 程式"):
        run_script(tmp_path, _payload())


def test_missing_model_file_message(tmp_path, scenario):
    write_data(tmp_path, scenario)
    with pytest.raises(FileNotFoundError, match="nn_gyro_prior_retrained.joblib.*train_nn_gyro_prior.py"):
        run_script(tmp_path, _payload(), model_placeholder=False)


def test_data_fallback_subdir_and_estimate_key(tmp_path, scenario, golden_run):
    """資料檔可放在 figure3_output/，基準欄位也可叫 ukf_estimate。"""
    write_data(tmp_path, scenario, baseline_key="ukf_estimate", subdir="figure3_output")
    run = run_script(tmp_path, _payload())
    assert_array_equal(run.namespace["baseline_ukf"], scenario["baseline_ukf"])
    assert_array_equal(run.namespace["nn_ukf"], golden_run.namespace["nn_ukf"])


def test_baseline_check_matches_reference_reimplementation(golden_run):
    """合成資料的 ukf_result 由 tests/synthetic.py 的獨立實作產生；paper_code.py 的
    baseline_check 應與之相同。（與真正 figure3_simulation.py 的比對見 test_figure3.py。）"""
    ns = golden_run.namespace
    assert ns["max_gap"] <= 1e-6
    assert "警告" not in golden_run.stdout


def test_baseline_mismatch_prints_warning(tmp_path, scenario):
    shifted = dict(scenario, baseline_ukf=scenario["baseline_ukf"] + 1e-3)
    write_data(tmp_path, shifted)
    run = run_script(tmp_path, _payload())
    assert run.namespace["max_gap"] == pytest.approx(1e-3, rel=1e-6)
    assert "警告" in run.stdout


def test_ekf_matches_hook_records(golden_run):
    ekf = golden_run.namespace["ekf"]
    assert_array_equal(np.array([p[1] for p in golden_run.ekf_post]), ekf)


def test_outputs_written(golden_run):
    names = {p.name for p in golden_run.workdir.iterdir()}
    for stem in ("Comparison_Fig_gyro_prior", "Ablation_Fig_gyro_prior"):
        assert {f"{stem}.pdf", f"{stem}.png"} <= names
    assert {"figure4_gyro_results_with_ablation.npz",
            "figure4_gyro_results_with_ablation.csv",
            "ablation_rmse.csv", "ablation_rmse_latex.txt"} <= names


def test_npz_contents(golden_run):
    ns = golden_run.namespace
    with np.load(golden_run.workdir / "figure4_gyro_results_with_ablation.npz") as out:
        assert set(out.files) == {"ekf", "baseline_ukf", "projection_ukf", "analytic_ukf",
                                  "nn_ukf", "analytic_prior", "nn_prior", "curve_speed",
                                  "normalized_difference"}
        assert_array_equal(out["nn_ukf"], ns["nn_ukf"])
        assert_array_equal(out["curve_speed"], ns["nn_curve_speed"])


def test_csv_header_and_columns(golden_run, defs):
    ns, V, SLIP = golden_run.namespace, defs["V"], defs["SLIP"]
    path = golden_run.workdir / "figure4_gyro_results_with_ablation.csv"
    assert path.read_text().splitlines()[0] == CSV_HEADER
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    assert table.shape == (600, 18)
    assert_allclose(table[:, 0], ns["step"])
    assert_allclose(table[:, 3], ns["measurements"][:, 0], rtol=1e-15)
    assert_allclose(table[:, 12], ns["nn_ukf"][:, V], rtol=1e-15)
    assert_allclose(table[:, 13], ns["nn_ukf"][:, SLIP], rtol=1e-15)
