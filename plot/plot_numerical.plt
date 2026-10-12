# ポアソン方程式の数値解 (out/numerical.dat) を滑らかな3次元曲面で描く
#   out/numerical.dat は script/poisson_numerical.py で作る
#   1列目 x, 2列目 y, 3列目 φ
#
# 実行:  gnuplot plot/plot_numerical.plt  (どのフォルダから実行してもよい)
# PNG に保存するときは下の set terminal と set output のコメントを外す

# このスクリプトがあるフォルダ (ARG0 = スクリプトのパス) を基準にファイルを探す
# → プロジェクト直下からでも plot/ からでも同じファイルを読める
script_dir = ""
do for [k=strlen(ARG0):1:-1] {
    if (script_dir eq "" && (ARG0[k:k] eq "/" || ARG0[k:k] eq "\\")) { script_dir = ARG0[1:k] }
}
file = script_dir."../out/numerical.dat"

# ---------- 格子の形に並べ直す ----------
# 曲面 (pm3d) で描くには, x が変わるところに空行が必要だが,
# numerical.dat は np.savetxt で書いたので空行がない.
# 正方格子 (n×n 点) として, n 行ごとに空行を入れたデータブロック $grid を作る
set table $raw
plot file using 1:2:3 with table
unset table
n = int(sqrt(|$raw|) + 0.5)

set print $grid
do for [k=1:|$raw|] {
    print $raw[k]
    if (k % n == 0) { print "" }
}
unset print

# ---------- 見た目 ----------
# set terminal pngcairo size 900,700
# set output script_dir."../graph/numerical3d.png"

set termoption noenhanced   # タイトルの _ を下付き文字にしない
set title "Poisson equation (out/numerical.dat)"
set xrange [0:1]
set yrange [0:1]
set zrange [0:1]
set cbrange [0:1]
set xlabel "x"
set ylabel "y"
set zlabel "phi" rotate parallel
set ticslevel 0
set view 60,30

set palette rgbformulae 33,13,10      # 紫 → 青 → 緑 → 赤 (虹色)
set pm3d depthorder                    # 奥から順に塗る (面の重なりを正しく)
set pm3d interpolate 2,2               # 格子の間を補間して色を滑らかにする
set pm3d lighting primary 0.5 specular 0.2   # 陰影をつけて立体感を出す
set style fill solid noborder          # 格子の線を描かない

splot $grid using 1:2:3 with pm3d notitle

pause -1
# set output
