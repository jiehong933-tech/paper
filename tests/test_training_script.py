"""train_nn_gyro_prior.py：資料產生、資料集組裝、自由遞迴測試，以及與 paper code 的銜接。

訓練程式在模組層級直接訓練，因此以 AST 只執行需要的部分（不訓練 MLP）。
"""

import ast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import synthetic
from paper_harness import (MODEL_TYPE, PAPER_FILE, TRAIN_FILE, load_training,
                           module_assign, paper_source, training_datasets,
                           training_rollout)


@pytest.fixture(scope="module")
def train():
    if not TRAIN_FILE.exists():
        pytest.skip("repo 中沒有 train_nn_gyro_prior.py")
    return load_training()


def test_constants_consistent_with_paper_code(train):
    assert (train["DT"], train["STEPS"]) == (0.1, 600)
    assert (train["SIGMA_ODO"], train["SIGMA_GYRO"]) == (0.4, 0.01)
    # 與 paper code 的 R = diag(0.4², 0.01²) 一致
    assert "R = np.diag([0.4 ** 2, 0.01 ** 2])" in paper_source()
    # 合成測試使用與訓練程式相同的參數
    assert (train["KAPPA"], train["ALPHA"], train["BLEND"], train["LAMBDA_MAX"]) == \
        (synthetic.KAPPA, synthetic.ALPHA, synthetic.BLEND, synthetic.LAMBDA_MAX)


# ---------- 兩支程式的銜接：檔名、存放位置、模型欄位 ----------

def test_model_file_name_and_folder_match():
    """訓練程式輸出與 paper code 讀取的是同一個檔名，且都放在 BASE_DIR（程式所在資料夾）。"""
    for name in ("BASE_DIR", "MODEL_FILE"):
        assert ast.dump(module_assign(TRAIN_FILE, name)) == ast.dump(module_assign(PAPER_FILE, name))
    model_file = module_assign(PAPER_FILE, "MODEL_FILE")
    assert model_file.right.value == "nn_gyro_prior_retrained.joblib"


def _saved_keys():
    """訓練程式 joblib.dump({...}, MODEL_FILE) 的字典鍵。"""
    for node in ast.walk(ast.parse(TRAIN_FILE.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "dump"
                and isinstance(node.args[0], ast.Dict)):
            assert isinstance(node.args[1], ast.Name) and node.args[1].id == "MODEL_FILE"
            return {key.value: value for key, value in zip(node.args[0].keys, node.args[0].values)}
    raise LookupError("找不到 joblib.dump")


def test_saved_model_has_every_field_paper_code_reads():
    read = {node.slice.value for node in ast.walk(ast.parse(paper_source()))
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "saved"}
    saved = _saved_keys()
    assert read == {"model", "scaler", "alpha", "blend", "kappa", "lambda_max"}
    assert read <= set(saved)
    assert saved["model_type"].value == MODEL_TYPE
    # 存進模型檔的參數就是訓練程式的常數
    for key, constant in [("alpha", "ALPHA"), ("blend", "BLEND"), ("kappa", "KAPPA"),
                          ("lambda_max", "LAMBDA_MAX")]:
        assert saved[key].id == constant


# ---------- make_sequence ----------

@pytest.mark.parametrize("augment", [False, True])
def test_make_sequence_shapes_and_ranges(train, augment):
    rng = np.random.default_rng(7)
    for _ in range(20):
        features, target = train["make_sequence"](rng, augment)
        assert features.shape == (train["STEPS"] - 1, 3)
        assert target.shape == (train["STEPS"] - 1,)
        assert features[:, 0].min() >= 10 and features[:, 0].max() <= 30
        assert features[:, 1].min() >= 0 and features[:, 1].max() <= train["LAMBDA_MAX"]
        assert features[:, 2].min() >= -0.30 and features[:, 2].max() <= 0.60
        assert target.min() >= 0 and target.max() <= 0.30
        assert np.all(np.isfinite(features))


def test_make_sequence_reproducible(train):
    a = train["make_sequence"](np.random.default_rng(3), True)
    b = train["make_sequence"](np.random.default_rng(3), True)
    assert_array_equal(a[0], b[0])
    assert_array_equal(a[1], b[1])


def test_datasets_stack_sequences_in_order():
    """資料集 = 以同一個 rng 連續產生的序列依序串接（以小 N 驗證原始碼）。"""
    ns = load_training(N_TRAIN=3, N_TEST=2)
    x_train, y_train, x_test, y_test = training_datasets(ns)
    steps = ns["STEPS"] - 1
    assert x_train.shape == (3 * steps, 3) and y_train.shape == (3 * steps,)
    assert x_test.shape == (2 * steps, 3) and y_test.shape == (2 * steps,)
    for seed, augment, count, x, y in [(ns["TRAIN_SEED"], True, 3, x_train, y_train),
                                       (ns["TEST_SEED"], False, 2, x_test, y_test)]:
        rng = np.random.default_rng(seed)
        parts = [ns["make_sequence"](rng, augment) for _ in range(count)]
        assert_array_equal(x, np.vstack([p[0] for p in parts]))
        assert_array_equal(y, np.hstack([p[1] for p in parts]))


# ---------- 自由遞迴測試 ----------

@pytest.mark.parametrize("c", [0.0, 0.1, 0.25])
def test_rollout_closed_form_for_constant_prior(train, c):
    """先驗恆為 c：λ_n = c + (0.30 − c)(1 − BLEND)^n。"""
    histories, final_values = training_rollout(
        train, synthetic.RecordingModel(bias=c), synthetic.IdentityScaler())
    n = np.arange(151)
    assert len(histories) == 3
    for history in histories:
        assert_allclose(history, c + (0.30 - c) * (1 - train["BLEND"]) ** n,
                        rtol=1e-12, atol=1e-15)
    assert final_values == [h[-1] for h in histories]


def test_rollout_features_and_output_clip(train):
    model = synthetic.RecordingModel(weights=(0.0, 0.0, 10.0))
    histories, _ = training_rollout(train, model, synthetic.IdentityScaler())
    first = [np.asarray(call)[0] for call in model.calls[::150]]   # 每個 r_o 的第一步
    assert_array_equal(first, [[20.0, 0.30, r_o] for r_o in (0.0, 0.1, 0.2)])
    # r_o=0.2 → 輸出 2.0，clip 為 LAMBDA_MAX=0.4
    assert_allclose(histories[2][1], (1 - train["BLEND"]) * 0.30 + train["BLEND"] * 0.4)


def test_rollout_matches_paper_fx_blend(train, defs):
    """自由遞迴的混合式與 paper code 的 fx（slip_prior 分支）相同。"""
    x = np.array([0.0, 20.0, 0.0, 0.3, 0.0, 0.0])
    prior = 0.12
    blended = defs["fx"](x, defs["DT"], np.zeros(2), slip_prior=prior)[defs["SLIP"]]
    histories, _ = training_rollout(train, synthetic.RecordingModel(bias=prior),
                                    synthetic.IdentityScaler())
    assert_allclose(histories[0][1], blended, rtol=1e-15)
