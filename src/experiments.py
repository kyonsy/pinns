"""
実験の一覧 (メモ) と, script.make_diff で比べる組

out/ のファイル        作り方                                                   内容
soft_w1.dat            python -m script.soft soft_w1.yaml soft_w1.dat           ソフト制約. 損失の重み [方程式, 外周, 中心4点] = [1, 1, 1]
soft_w1e2.dat          python -m script.soft soft_w1e2.yaml soft_w1e2.dat       ソフト制約. 境界の重みを 10^2 にする
soft_w1e4.dat          python -m script.soft soft_w1e4.yaml soft_w1e4.dat       ソフト制約. 境界の重みを 10^4 にする
soft_w1e6.dat          python -m script.soft soft_w1e6.yaml soft_w1e6.dat       ソフト制約. 境界の重みを 10^6 にする
soft_w1_deep64.dat     python -m script.soft soft_w1_deep64.yaml soft_w1_deep64.dat
                                                                                soft_w1 の隠れ層を 64 層にする. 格子が 101 点なので数値解とは比べられない
hard_g.dat             python -m script.hard g  hard.yaml hard_g.dat            ハード制約 φ = G + D·C·N
hard_gl.dat            python -m script.hard gl hard.yaml hard_gl.dat           ハード制約 φ = G·L + D·C·N
numerical.dat          python -m script.numerical                               差分法による数値解
hc_g.dat, hc_gl.dat    python -m script.export_hard_constraint                  ハード制約に使う関数 G, G·L
diff_*.dat, mse.dat    python -m script.make_diff                               下の DIFFS, MSE_FILES

グラフの凡例・ラベル (日本語) は plot/ の各 .plt にファイル名ごとに書いてある
"""

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
