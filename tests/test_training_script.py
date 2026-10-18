"""train_nn_gyro_prior.py 的資料產生與推論輔助函式（不執行 main、不訓練 MLP）。"""

import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import synthetic
from paper_harness import MODEL_TYPE, REPO_ROOT, paper_source

TRAIN_FILE = REPO_ROOT / "train_nn_gyro_prior.py"


@pytest.fixture(scope="module")
def train():
    if not TRAIN_FILE.exists():
        pytest.skip("repo 中沒有 train_nn_gyro_prior.py")
    spec = importlib.util.spec_from_file_location("train_nn_gyro_prior", TRAIN_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # main() 受 __name__ 保護，不會訓練
    return module


def test_constants_consistent_with_paper_code(train):
    assert (train.DT, train.STEPS) == (0.1, 600)
    assert (train.SIGMA_ODO, train.SIGMA_GYRO) == (0.4, 0.01)
    # 與 paper code 的 R = diag(0.4², 0.01²) 一致
    assert "R = np.diag([0.4 ** 2, 0.01 ** 2])" in paper_source()
    assert f'"{MODEL_TYPE}"' in TRAIN_FILE.read_text(encoding="utf-8")
    # 合成測試使用與訓練程式相同的參數
    assert (train.KAPPA, train.ALPHA, train.BLEND, train.LAMBDA_MAX) == \
        (synthetic.KAPPA, synthetic.ALPHA, synthetic.BLEND, synthetic.LAMBDA_MAX)


@pytest.mark.parametrize("augment", [False, True])
def test_make_sequence_shapes_and_ranges(train, augment):
    rng = np.random.default_rng(7)
    for _ in range(20):
        features, target = train.make_sequence(rng, augment)
        assert features.shape == (train.STEPS - 1, 3)
        assert target.shape == (train.STEPS - 1,)
        assert features[:, 0].min() >= 10 and features[:, 0].max() <= 30
        assert features[:, 1].min() >= 0 and features[:, 1].max() <= train.LAMBDA_MAX
        assert features[:, 2].min() >= -0.30 and features[:, 2].max() <= 0.60
        assert target.min() >= 0 and target.max() <= 0.30
        assert np.all(np.isfinite(features))


def test_make_sequence_reproducible(train):
    a = train.make_sequence(np.random.default_rng(3), True)
    b = train.make_sequence(np.random.default_rng(3), True)
    assert_array_equal(a[0], b[0])
    assert_array_equal(a[1], b[1])


def test_make_dataset_stacks_sequences(train):
    x, y = train.make_dataset(3, seed=5)
    assert x.shape == (3 * (train.STEPS - 1), 3) and y.shape == (3 * (train.STEPS - 1),)
    x2, y2 = train.make_dataset(3, seed=5)
    assert_array_equal(x, x2)
    assert not np.array_equal(x, train.make_dataset(3, seed=6)[0])


def test_predict_once_clips_inputs_and_output(train):
    model = synthetic.RecordingModel(weights=(0.0, 0.0, 10.0))
    out = train.predict_once(model, synthetic.IdentityScaler(), 50.0, -1.0, 0.9)
    assert_array_equal(model.calls[0], [[30.0, 0.0, 0.60]])
    assert out == train.LAMBDA_MAX          # 10*0.6 = 6 → clip 0.4
    model = synthetic.RecordingModel(bias=-0.2)
    assert train.predict_once(model, synthetic.IdentityScaler(), 20.0, 0.1, 0.0) == 0.0


@pytest.mark.parametrize("c", [0.0, 0.1, 0.25])
def test_rollout_closed_form_for_constant_prior(train, c):
    """先驗恆為 c：λ_n = c + (0.30 − c)(1 − BLEND)^n。"""
    history = train.rollout(synthetic.RecordingModel(bias=c), synthetic.IdentityScaler(), 0.1)
    n = np.arange(151)
    assert history.shape == (151,)
    assert_allclose(history, c + (0.30 - c) * (1 - train.BLEND) ** n, rtol=1e-12, atol=1e-15)


def test_rollout_matches_paper_fx_blend(train, defs):
    """rollout 的混合式與 paper code 的 fx（slip_prior 分支）相同。"""
    x = np.array([0.0, 20.0, 0.0, 0.3, 0.0, 0.0])
    prior = 0.12
    blended = defs["fx"](x, defs["DT"], np.zeros(2), slip_prior=prior)[defs["SLIP"]]
    history = train.rollout(synthetic.RecordingModel(bias=prior),
                            synthetic.IdentityScaler(), 0.0)
    assert_allclose(history[1], blended, rtol=1e-15)
