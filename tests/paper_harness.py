"""在不修改原始程式的前提下，載入與執行 `figure3_simulation.py`、
`train_nn_gyro_prior.py` 與 `paper_code.py`。

三支程式都是在模組層級直接讀檔、跑模擬、畫圖與輸出的腳本，import 就會整支執行。
本模組提供兩種方式：

1. ``load_definitions``：用 AST 只挑出常數與函式定義（fx、hx、rmse、
   EKF 的 F 矩陣、區段切分 slice）執行，外部參數
   （KAPPA、BLEND 等原本來自 joblib 模型檔）由測試注入。
2. ``run_script``：在暫存目錄中執行整份腳本（含 EKF 與六種 UKF、繪圖與輸出），
   以合成資料與假模型取代缺少的 `figure3_shared_data.npz` 與
   `nn_gyro_prior_retrained.joblib`。執行時：
   - 以 ``unittest.mock.patch`` 把 ``filterpy.kalman.UnscentedKalmanFilter``
     換成記錄用子類別（行為完全相同，只多記錄 x、P、殘差、S 與 fx 呼叫），
   - 在 AST 層級於 EKF 迴圈內插入兩個記錄用 hook（不改變任何運算），
   - 若有提供假模型，則 patch ``joblib.load`` 直接回傳。
   原始檔案內容完全不被改動；編譯時使用原檔路徑，讓 coverage 能對應行號。
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FILE = REPO_ROOT / "paper_code.py"
TRAIN_FILE = REPO_ROOT / "train_nn_gyro_prior.py"
MODEL_TYPE = "gyro_curve_slip_prior_v2"

# 由 load_definitions 執行的模組層級賦值（以被賦值的名稱判斷）。
_DEFINITION_NAMES = {
    "DT", "SLIP_START", "SLIP_END", "S", "V", "YAW", "SLIP", "BA", "BG",
    "Q", "R", "X0", "P0", "F", "slip_part", "recovery_part",
}


def paper_source() -> str:
    return PAPER_FILE.read_text(encoding="utf-8")


def _target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return [target.value.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for element in target.elts:
            names += _target_names(element)
        return names
    return [None]


def load_definitions(alpha=0.02, blend=0.06, kappa=0.002, lambda_max=0.4,
                     n_steps=600):
    """執行 `paper_code.py` 中的常數與函式定義，回傳命名空間 dict。"""
    tree = ast.parse(paper_source(), filename=str(PAPER_FILE))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = [n for t in node.targets for n in _target_names(t)]
            if names and all(n in _DEFINITION_NAMES for n in names):
                selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "np": np, "ALPHA": alpha, "BLEND": blend, "KAPPA": kappa,
        "LAMBDA_MAX": lambda_max, "step": np.arange(n_steps),
    }
    exec(compile(module, str(PAPER_FILE), "exec"), namespace)
    return namespace


def _ekf_loop(tree):
    """回傳 EKF 迴圈（模組層級、迴圈內有 `H = np.zeros(...)` 的 for）。"""
    for node in tree.body:
        if isinstance(node, ast.For):
            for statement in node.body:
                if (isinstance(statement, ast.Assign)
                        and _target_names(statement.targets[0]) == ["H"]):
                    return node
    raise LookupError("找不到 EKF 迴圈")


def ekf_measurement_jacobian(namespace, x):
    """執行 EKF 迴圈中組 H 的那幾行（原始碼），回傳 H。"""
    loop = _ekf_loop(ast.parse(paper_source()))
    statements = [s for s in loop.body
                  if isinstance(s, ast.Assign)
                  and all(n == "H" for t in s.targets for n in _target_names(t))]
    local = dict(namespace, x=np.asarray(x, dtype=float))
    exec(compile(ast.Module(body=statements, type_ignores=[]),
                 str(PAPER_FILE), "exec"), local)
    return local["H"]


def ukf_mode_order():
    """從原始碼讀出 UKF 迴圈 `for mode in (...)` 的模式順序。"""
    for node in ast.parse(paper_source()).body:
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "mode"):
            return tuple(ast.literal_eval(node.iter))
    raise LookupError("找不到 UKF 模式迴圈")


def _instrumented_tree():
    tree = ast.parse(paper_source(), filename=str(PAPER_FILE))
    loop = _ekf_loop(tree)
    last_h = max(i for i, s in enumerate(loop.body)
                 if isinstance(s, ast.Assign)
                 and all(n == "H" for t in s.targets for n in _target_names(t)))
    prior_hook = ast.parse(
        "__ekf_prior_hook__(k, x.copy(), P.copy(), H.copy(), z)").body[0]
    post_hook = ast.parse("__ekf_post_hook__(k, x.copy(), P.copy())").body[0]
    ast.copy_location(prior_hook, loop.body[last_h])
    ast.copy_location(post_hook, loop.body[-1])
    loop.body.insert(last_h + 1, prior_hook)
    loop.body.append(post_hook)
    return ast.fix_missing_locations(tree)


def _recording_ukf_class(registry):
    from filterpy.kalman import UnscentedKalmanFilter

    class RecordingUKF(UnscentedKalmanFilter):
        """與 filterpy UKF 完全相同，只額外記錄每一步的中間量。"""

        def __init__(self, dim_x, dim_z, dt, hx, fx, points, **kwargs):
            self.fx_calls = []  # 每次 predict：各 sigma point 收到的 slip_prior

            def recording_fx(x, dt, **fx_args):
                self.fx_calls[-1].append(fx_args.get("slip_prior"))
                return fx(x, dt, **fx_args)

            super().__init__(dim_x, dim_z, dt, hx, recording_fx, points, **kwargs)
            self.log = {k: [] for k in (
                "x_prior", "P_prior", "x_post", "P_post", "y", "S",
                "sigmas_update", "z")}
            registry.append(self)

        def predict(self, dt=None, UT=None, fx=None, **fx_args):
            if not self.fx_calls:  # 第一次 predict 前：記錄程式設定的 P0 與 Q
                self.initial_P, self.initial_Q = self.P.copy(), np.array(self.Q, copy=True)
            self.fx_calls.append([])
            super().predict(dt=dt, UT=UT, fx=fx, **fx_args)
            self.log["x_prior"].append(self.x.copy())
            self.log["P_prior"].append(self.P.copy())

        def update(self, z, R=None, UT=None, hx=None, **hx_args):
            self.log["sigmas_update"].append(np.array(self.sigmas_f, copy=True))
            super().update(z, R=R, UT=UT, hx=hx, **hx_args)
            self.log["x_post"].append(self.x.copy())
            self.log["P_post"].append(self.P.copy())
            self.log["y"].append(np.array(self.y, copy=True))
            self.log["S"].append(np.array(self.S, copy=True))
            self.log["z"].append(np.array(z, copy=True))

    return RecordingUKF


@dataclass
class ScriptRun:
    namespace: dict
    ukf: dict            # mode -> RecordingUKF（含 log 與 fx_calls）
    ekf_prior: list = field(default_factory=list)   # (k, x, P, H, z)
    ekf_post: list = field(default_factory=list)    # (k, x, P)
    stdout: str = ""
    workdir: Path | None = None


def model_payload(model, scaler, alpha=0.02, blend=0.06, kappa=0.002,
                  lambda_max=0.4, model_type=MODEL_TYPE):
    """組出與 `paper_code.py` 讀取格式相同的模型 dict。"""
    return {"model_type": model_type, "model": model, "scaler": scaler,
            "alpha": alpha, "blend": blend, "kappa": kappa,
            "lambda_max": lambda_max}


def write_data(directory, scenario, baseline_key="ukf_result", subdir=None):
    directory = Path(directory)
    target = directory / subdir if subdir else directory
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "figure3_shared_data.npz",
             step=scenario["step"], truth=scenario["truth"],
             measurement=scenario["measurement"],
             imu_input=scenario["imu_input"],
             **{baseline_key: scenario["baseline_ukf"]})


def run_script(workdir, payload=None, model_placeholder=True):
    """在 workdir 執行整份 `paper_code.py`。

    workdir 內需已有 figure3_shared_data.npz（或 figure3_output/ 子目錄）。
    payload 為 None 時使用 workdir 內真正的 joblib 模型檔；否則 joblib.load 直接回傳
    payload，並（model_placeholder=True 時）建立空的模型檔，讓程式的存在檢查通過。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workdir = Path(workdir)
    model_file = workdir / "nn_gyro_prior_retrained.joblib"
    if payload is not None and model_placeholder and not model_file.exists():
        model_file.touch()
    registry, ekf_prior, ekf_post = [], [], []
    namespace = {
        "__name__": "paper_code_under_test",
        "__file__": str(workdir / "paper_code.py"),
        "__ekf_prior_hook__": lambda *a: ekf_prior.append(a),
        "__ekf_post_hook__": lambda *a: ekf_post.append(a),
    }
    code = compile(_instrumented_tree(), str(PAPER_FILE), "exec")
    patches = [mock.patch("filterpy.kalman.UnscentedKalmanFilter",
                          _recording_ukf_class(registry))]
    if payload is not None:
        patches.append(mock.patch("joblib.load", lambda *_a, **_k: payload))
    buffer = io.StringIO()
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(contextlib.redirect_stdout(buffer))
        # 測試環境多半沒有中文字型；略過 matplotlib 的缺字訊息
        logger = logging.getLogger("matplotlib")
        level = logger.level
        logger.setLevel(logging.ERROR)
        stack.callback(logger.setLevel, level)
        try:
            exec(code, namespace)
        finally:
            plt.close("all")
    modes = ukf_mode_order()
    if len(registry) != len(modes):
        raise AssertionError(f"預期 {len(modes)} 個 UKF，實際建立 {len(registry)} 個")
    return ScriptRun(namespace, dict(zip(modes, registry)), ekf_prior, ekf_post,
                     buffer.getvalue(), workdir)


def speed_slip_metrics(namespace):
    """以 `paper_code.py` 自己的 rmse 與區段，計算各方法的速度／打滑率 RMSE。

    回傳 {方法: {"speed_slip", "speed_recovery", "slip_slip", "slip_recovery"}}；
    Raw Odometry 只有速度兩項。
    """
    ns = namespace
    rmse, truth, V, SLIP = ns["rmse"], ns["truth"], ns["V"], ns["SLIP"]
    sp, rp = ns["slip_part"], ns["recovery_part"]
    methods = {"ekf": ns["ekf"], "baseline_ukf": ns["baseline_ukf"],
               **{f"ukf_{mode}": result for mode, result in ns["ukf_results"].items()}}
    metrics = {"raw_odometry": {
        "speed_slip": float(rmse(ns["measurements"][sp, 0], truth[sp, V])),
        "speed_recovery": float(rmse(ns["measurements"][rp, 0], truth[rp, V]))}}
    for name, result in methods.items():
        metrics[name] = {
            "speed_slip": float(rmse(result[sp, V], truth[sp, V])),
            "speed_recovery": float(rmse(result[rp, V], truth[rp, V])),
            "slip_slip": float(rmse(result[sp, SLIP], truth[sp, SLIP])),
            "slip_recovery": float(rmse(result[rp, SLIP], truth[rp, SLIP])),
        }
    return metrics


def find_real_inputs():
    """尋找論文實際使用的資料與權重；回傳 (data_file, model_file)，缺檔為 None。

    搜尋順序：環境變數 PAPER_DATA_DIR、repo 根目錄、repo/figure3_output/。
    """
    import os
    roots = [Path(p) for p in [os.environ.get("PAPER_DATA_DIR")] if p] + [REPO_ROOT]
    data = model = None
    for root in roots:
        for candidate in (root / "figure3_shared_data.npz",
                          root / "figure3_output" / "figure3_shared_data.npz"):
            if data is None and candidate.exists():
                data = candidate
        if model is None and (root / "nn_gyro_prior_retrained.joblib").exists():
            model = root / "nn_gyro_prior_retrained.joblib"
    return data, model


def run_real(workdir):
    """以論文實際資料與權重執行（連結到暫存目錄，避免輸出覆蓋 repo 內的檔案）。"""
    data, model = find_real_inputs()
    if data is None or model is None:
        return None
    workdir = Path(workdir)
    (workdir / "figure3_shared_data.npz").symlink_to(data)
    (workdir / "nn_gyro_prior_retrained.joblib").symlink_to(model)
    return run_script(workdir, payload=None)


# ---------- 訓練程式 train_nn_gyro_prior.py ----------
# 訓練程式在模組層級直接訓練，因此同樣以 AST 只挑出需要的部分執行（不訓練）。

_TRAIN_CONSTANTS = {
    "DT", "STEPS", "KAPPA", "SIGMA_ODO", "SIGMA_GYRO", "ALPHA", "BLEND", "LAMBDA_MAX",
    "N_TRAIN", "N_TEST", "TRAIN_SEED", "TEST_SEED", "MODEL_SEED",
}


def _assigned_names(node):
    if isinstance(node, ast.Assign):
        return {n for t in node.targets for n in _target_names(t)}
    return set()


def _exec_parts(path, select, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [node for node in tree.body if select(node)]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_training(**overrides):
    """執行訓練程式的常數與 make_sequence；overrides 可覆寫常數（如 N_TRAIN）。"""
    namespace = {"np": np}
    _exec_parts(TRAIN_FILE, lambda n: isinstance(n, ast.FunctionDef)
                or (_assigned_names(n) and _assigned_names(n) <= _TRAIN_CONSTANTS), namespace)
    namespace.update(overrides)
    return namespace


def training_datasets(namespace):
    """執行訓練程式中建立 x_train、y_train、x_test、y_test 的原始碼。"""
    names = {"rng", "x_train", "y_train", "x_test", "y_test"}
    _exec_parts(TRAIN_FILE, lambda n: bool(_assigned_names(n)) and _assigned_names(n) <= names,
                namespace)
    return namespace["x_train"], namespace["y_train"], namespace["x_test"], namespace["y_test"]


def training_rollout(namespace, model, scaler):
    """以指定的 model／scaler 執行訓練程式中的自由遞迴測試，回傳 (histories, final_values)。"""
    namespace = dict(namespace, model=model, scaler=scaler)
    names = {"innovations", "histories", "final_values"}

    def select(node):
        if isinstance(node, ast.For):
            return isinstance(node.target, ast.Name) and node.target.id == "r_o"
        return bool(_assigned_names(node)) and _assigned_names(node) <= names

    _exec_parts(TRAIN_FILE, select, namespace)
    return namespace["histories"], namespace["final_values"]


def module_assign(path, name):
    """回傳模組層級 `name = ...` 的 AST 值節點。"""
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if _assigned_names(node) == {name}:
            return node.value
    raise LookupError(name)
