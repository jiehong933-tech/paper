import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic  # noqa: E402
from paper_harness import load_definitions, model_payload, run_script, write_data  # noqa: E402


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
