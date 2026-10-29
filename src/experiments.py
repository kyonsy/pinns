"""
実験の一覧. 実験の設定ファイル・出力ファイル・説明はここにだけ書く
ファイル名は拡張子まで含めて書く. 設定ファイルは config/, 出力ファイルは out/ の中にある
各スクリプトは設定ファイル名でこの一覧を引き, 差と MSE (script.make_diff) は比べる組をここから読む

名前の付け方
    soft_...  : ソフト制約 (script.soft). w<λ> は境界の損失の重み λ
    hard_...  : ハード制約 (script.hard). 後ろは境界条件を満たす部分 (g: G, gl: G·L)
    hc_...    : ハード制約に使う関数そのもの (script.export_hard_constraint)
    numerical : 差分法による数値解 (script.numerical)
    diff_<a>-<b> : a - b (script.make_diff)
グラフの凡例・ラベル (日本語) は plot/ の各 .plt にファイル名ごとに書いてある
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Experiment:
    config      : str   # 設定ファイル (config/ の中)
    output      : str   # 出力ファイル (out/ の中)
    kind        : str   # "soft" (script.soft で学習) か "hard" (script.hard で学習)
    description : str   # 何の実験か

# PINN の実験. 設定ファイル名 -> 実験
EXPERIMENTS: dict[str, Experiment] = {e.config: e for e in [
    Experiment("soft_w1.yaml",        "soft_w1.dat",        "soft", "ソフト制約. 損失の重み [方程式, 外周, 中心4点] = [1, 1, 1]"),
    Experiment("soft_w1e2.yaml",      "soft_w1e2.dat",      "soft", "ソフト制約. 境界の重みを 10^2 にする"),
    Experiment("soft_w1e4.yaml",      "soft_w1e4.dat",      "soft", "ソフト制約. 境界の重みを 10^4 にする"),
    Experiment("soft_w1e6.yaml",      "soft_w1e6.dat",      "soft", "ソフト制約. 境界の重みを 10^6 にする"),
    Experiment("soft_w1_deep64.yaml", "soft_w1_deep64.dat", "soft", "soft_w1 の隠れ層を 64 層にする. 格子が 101 点なので数値解とは比べられない"),
    Experiment("hard_g.yaml",         "hard_g.dat",         "hard", "ハード制約 φ = G + D·C·N (fcnn.PoissonFCNN1)"),
    Experiment("hard_gl.yaml",        "hard_gl.dat",        "hard", "ハード制約 φ = G·L + D·C·N (fcnn.PoissonFCNN2)"),
]}

# 差をとる組 (a, b, 出力). out/ の a - b を out/ の 出力 に保存する
DIFFS = [
    ("hard_g.dat",  "hc_g.dat",      "diff_hard_g-hc_g.dat"),
    ("hard_gl.dat", "hc_gl.dat",     "diff_hard_gl-hc_gl.dat"),
    ("hard_g.dat",  "numerical.dat", "diff_hard_g-numerical.dat"),
    ("hard_gl.dat", "numerical.dat", "diff_hard_gl-numerical.dat"),
    ("hc_g.dat",    "numerical.dat", "diff_hc_g-numerical.dat"),
    ("hc_gl.dat",   "numerical.dat", "diff_hc_gl-numerical.dat"),
]

# 数値解との MSE を棒グラフにする出力ファイル. 並び順がそのまま棒の並び順になる
MSE_FILES = ["soft_w1.dat", "soft_w1e2.dat", "soft_w1e4.dat", "soft_w1e6.dat", "hard_g.dat", "hard_gl.dat"]

def of_kind(kind:str) -> list[str]:
    """kind ("soft" / "hard") の実験の設定ファイル名"""
    return [e.config for e in EXPERIMENTS.values() if e.kind == kind]
