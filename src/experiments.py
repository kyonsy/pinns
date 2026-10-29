"""
実験の一覧. 実験の名前・設定ファイル・説明はここにだけ書く
一括実行 (script.run_all), 差と MSE (script.make_diff), グラフ (script.plot_all) はこの一覧を使う

名前の付け方
    soft_...  : ソフト制約 (script.soft). w<λ> は境界の損失の重み λ
    hard_...  : ハード制約 (script.hard). 後ろは境界条件を満たす部分 (g: G, gl: G·L)
    hc_...    : ハード制約に使う関数そのもの (script.export_hard_constraint)
    numerical : 差分法による数値解 (script.numerical)
    diff_<a>-<b> : a - b (script.make_diff)
グラフの凡例・ラベル (日本語) は plot/ の各 .plt に名前ごとに書いてある
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Experiment:
    name        : str   # 出力 out/<name>.dat の名前
    kind        : str   # "soft" (script.soft で学習) か "hard" (script.hard で学習)
    config      : str   # 設定ファイル config/<config>.yaml
    description : str   # 何の実験か

# PINN の実験
EXPERIMENTS: dict[str, Experiment] = {e.name: e for e in [
    Experiment("soft_w1",           "soft", "soft_w1",           "ソフト制約. 損失の重み [方程式, 外周, 中心4点] = [1, 1, 1]"),
    Experiment("soft_w1e2",         "soft", "soft_w1e2",         "ソフト制約. 境界の重みを 10^2 にする"),
    Experiment("soft_w1e4",         "soft", "soft_w1e4",         "ソフト制約. 境界の重みを 10^4 にする"),
    Experiment("soft_w1e6",         "soft", "soft_w1e6",         "ソフト制約. 境界の重みを 10^6 にする"),
    Experiment("soft_w1_deep64",    "soft", "soft_w1_deep64",    "soft_w1 の隠れ層を 64 層にする. 格子が 101 点なので数値解とは比べられない"),
    Experiment("soft_hardedge_w20", "soft", "soft_hardedge_w20", "外周の φ=0 だけハード制約にする. 損失の重み [方程式, 中心4点] = [1, 20]"),
    Experiment("hard_g",            "hard", "hard",              "ハード制約 φ = G + D·C·N (fcnn.PoissonFCNN1)"),
    Experiment("hard_gl",           "hard", "hard",              "ハード制約 φ = G·L + D·C·N (fcnn.PoissonFCNN2)"),
]}

# 一括実行 (script.run_all) で学習する実験
BATCH = ["soft_w1", "soft_w1e2", "soft_w1e4", "soft_w1e6", "hard_g", "hard_gl"]

# ソフト制約で境界の重みを変えた比較 (y=0.5 の断面図 section_soft)
SOFT_WEIGHTS = ["soft_w1", "soft_w1e2", "soft_w1e4", "soft_w1e6"]

# 等高線図・曲面を描く結果
SURFACES = BATCH + ["numerical", "hc_g", "hc_gl"]

# 差をとる組 (a, b). out/diff_<a>-<b>.dat に a - b を保存する
DIFF_PAIRS = [
    ("hard_g",  "hc_g"),
    ("hard_gl", "hc_gl"),
    ("hard_g",  "numerical"),
    ("hard_gl", "numerical"),
    ("hc_g",    "numerical"),
    ("hc_gl",   "numerical"),
]

# y=0.5 の断面で重ねて描く組 (section.plt)
SECTIONS = [
    ["hc_g",  "hard_g",  "numerical"],
    ["hc_gl", "hard_gl", "numerical"],
]

# 数値解との MSE を棒グラフにする手法. 並び順がそのまま棒の並び順になる
MSE_TYPES = SOFT_WEIGHTS + ["hard_g", "hard_gl"]

def diff_name(a:str, b:str) -> str:
    """a - b を保存するファイルの名前"""
    return f"diff_{a}-{b}"

def of_kind(kind:str) -> list[str]:
    """kind ("soft" / "hard") の実験の名前"""
    return [e.name for e in EXPERIMENTS.values() if e.kind == kind]
