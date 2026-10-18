import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic  # noqa: E402
from paper_harness import (load_definitions, model_payload, run_figure3, run_real,  # noqa: E402
                           run_script, training_payload, write_data)


@pytest.fixture(scope="session")
def defs():
    """`paper_code.py` 的常數與函式（外部參數使用合成值）。"""
    return load_definitions(alpha=synthetic.ALPHA, blend=synthetic.BLEND,
                            kappa=synthetic.KAPPA, lambda_max=synthetic.LAMBDA_MAX,
                            n_steps=synthetic.N_STEPS)


@pytest.fixture(scope="session")
def scenario():
    return synthetic.make_scenario()


def _run(tmp_path_factory, scenario, model, name, **params):
    workdir = tmp_path_factory.mktemp(name)
    write_data(workdir, scenario)
    payload = model_payload(model, synthetic.IdentityScaler(),
                            alpha=params.get("alpha", synthetic.ALPHA),
                            blend=params.get("blend", synthetic.BLEND),
                            kappa=params.get("kappa", synthetic.KAPPA),
                            lambda_max=params.get("lambda_max", synthetic.LAMBDA_MAX))
    run = run_script(workdir, payload)
    run.model = model
    return run


@pytest.fixture(scope="session")
def golden_run(tmp_path_factory, scenario):
    """以固定假模型跑整份 `paper_code.py`（合成資料）。"""
    return _run(tmp_path_factory, scenario, synthetic.golden_model(), "golden")


@pytest.fixture(scope="session")
def run_factory(tmp_path_factory, scenario):
    def factory(model, name="run", **params):
        return _run(tmp_path_factory, scenario, model, name, **params)
    return factory


@pytest.fixture(scope="session")
def fig3(tmp_path_factory):
    """執行 figure3_simulation.py 產生的真實資料：(命名空間, 印出的文字)。"""
    return run_figure3(tmp_path_factory.mktemp("figure3"))


@pytest.fixture(scope="session")
def real_run(tmp_path_factory):
    """真實圖 3 資料 + 恆輸出 0 的假 MLP（參數取自訓練程式）跑 paper_code.py。

    只用來檢查與 MLP 權重無關的方法（EKF、基準 UKF、投影、解析先驗、偏差對照組）。
    """
    return run_real(tmp_path_factory.mktemp("real_no_mlp"),
                    payload=training_payload(synthetic.zero_model()))
