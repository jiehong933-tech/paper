"""train_nn_gyro_prior.py：資料產生、資料集組裝、自由遞迴測試，以及與 paper_code.py 的銜接。

訓練程式在模組層級直接訓練，因此以 AST 只執行需要的部分（不訓練 MLP）。
"""

import ast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import synthetic
from paper_harness import (MODEL_TYPE, PAPER_FILE, TRAIN_FILE, clip_bounds, load_training,
                           module_assign, paper_source, training_datasets,
                           training_rollout)


@pytest.fixture(scope="module")
def train():
    if not TRAIN_FILE.exists():
        pytest.skip("repo 中沒有 train_nn_gyro_prior.py")
    return load_training()


def test_constants_consistent_with_paper_code(train, defs):
    assert (train["DT"], train["STEPS"]) == (defs["DT"], 600)
    assert (train["SIGMA_ODO"], train["SIGMA_GYRO"]) == (0.4, 0.01)
    # 訓練資料的雜訊與 paper_code.py 的量測雜訊 R 一致
    assert_array_equal(defs["R"], np.diag([train["SIGMA_ODO"] ** 2, train["SIGMA_GYRO"] ** 2]))
    # 合成測試使用與訓練程式相同的參數
    assert (train["KAPPA"], train["ALPHA"], train["BLEND"], train["LAMBDA_MAX"]) == \
        (synthetic.KAPPA, synthetic.ALPHA, synthetic.BLEND, synthetic.LAMBDA_MAX)


# ---------- 兩支程式的銜接：檔名、存放位置、模型欄位 ----------

def test_model_file_name_and_folder_match():
    """訓練程式輸出與 paper_code.py 讀取的是同一個檔名，且都放在 BASE_DIR（程式所在資料夾）。"""
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
    """自由遞迴的混合式與 paper_code.py 的 fx（slip_prior 分支）相同。"""
    x = np.array([0.0, 20.0, 0.0, 0.3, 0.0, 0.0])
    prior = 0.12
    blended = defs["fx"](x, defs["DT"], np.zeros(2), slip_prior=prior)[defs["SLIP"]]
    histories, _ = training_rollout(train, synthetic.RecordingModel(bias=prior),
                                    synthetic.IdentityScaler())
    assert_allclose(histories[0][1], blended, rtol=1e-15)


# ---------- 固定訓練設定：資料產生方式、MLP 超參數、合理性檢查門檻 ----------

def test_make_sequence_fingerprint(train):
    """make_sequence 的輸出與記錄的指紋相同（擴增比例、打滑事件、雜訊等任何改動都會被抓到）。"""
    import json
    import sys
    sys.path.insert(0, str(TRAIN_FILE.parent / "tests" / "tools"))
    from make_training_fingerprint import fingerprint

    saved = json.loads((TRAIN_FILE.parent / "tests" / "data" / "training_fingerprint.json")
                       .read_text(encoding="utf-8"))["fingerprint"]
    current = fingerprint(train["make_sequence"])
    for key, rows in saved.items():
        for saved_row, row in zip(rows, current[key]):
            assert row["slip_steps"] == saved_row["slip_steps"]
            for field in ("feature_sums", "target_sum", "row_0", "row_300"):
                assert_allclose(row[field], saved_row[field], rtol=1e-12, err_msg=f"{key} {field}")


def _training_tree():
    return ast.parse(TRAIN_FILE.read_text(encoding="utf-8"))


def test_mlp_hyperparameters_pinned():
    call = next(node for node in ast.walk(_training_tree())
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MLPRegressor")
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert kwargs == {
        "hidden_layer_sizes": "(64, 64)", "activation": "'tanh'", "solver": "'adam'",
        "learning_rate_init": "0.001", "batch_size": "2048", "max_iter": "150",
        "early_stopping": "True", "validation_fraction": "0.15", "n_iter_no_change": "15",
        "random_state": "MODEL_SEED",
    }


def test_sanity_thresholds_pinned():
    """訓練後的三個合理性檢查（失敗時 raise RuntimeError）門檻不變。"""
    checks = [ast.unparse(node.test) for node in _training_tree().body
              if isinstance(node, ast.If) and isinstance(node.body[0], ast.Raise)]
    assert checks == [
        "active_rmse > 0.06",
        "abs(final_values[0]) > 0.02",
        "abs(final_values[1] - 0.1) > 0.04 or abs(final_values[2] - 0.2) > 0.04",
    ]


@pytest.mark.parametrize("training_suffix, paper_suffix, bounds", [
    ("/ curve_speed - 1", "/ curve_speed - 1", (-0.30, 0.60)),   # 里程計差異特徵
    ("/ KAPPA", "/ KAPPA", (10, 30)),                             # 曲率速度
])
def test_feature_clips_match_between_training_and_inference(training_suffix, paper_suffix, bounds):
    """MLP 的輸入特徵在訓練與推論（paper_code.py）時用相同的限制範圍建立。"""
    assert clip_bounds(TRAIN_FILE, training_suffix) == {bounds}
    assert clip_bounds(PAPER_FILE, paper_suffix) == {bounds}
