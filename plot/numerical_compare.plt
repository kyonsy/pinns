# ポアソン方程式の数値解を Python と C で比べる
#   Python : out/numerical.dat   (script/poisson_numerical.py で作る)
#   C      : out/numerical_c.dat (csrc で make run して作る)
# どちらも 1列目 x, 2列目 y, 3列目 φ
#
# プロジェクト直下で実行する:  gnuplot plot/numerical_compare.plt
# PNG に保存するときは下の set terminal と set output のコメントを外す

py = "./out/numerical.dat"
c  = "./out/numerical_c.dat"

n = 101      # 分割数 (config/poisson_eq.yaml の grid_density と同じにする)
i = 51       # 断面を取る x の格子番号. x = i/n (中心の φ=1 の点を通る)
# ※ データは x が外側, y が内側の順に並んでいるので, x を固定すると
#   断面の点が連続した行になり, with lines で線がつながる
#   (y を固定すると点が飛び飛びになり, 線が途切れて何も描かれない)

# set terminal pngcairo size 1500,500
# set output "./graph/numerical_compare.png"

set multiplot layout 1,3
set termoption noenhanced   # タイトルの _ を下付き文字にしない

# ---------- 左: Python の結果 ----------
set title "Python (out/numerical.dat)"
set xrange [0:1]
set yrange [0:1]
set zrange [0:1]
set cbrange [0:1]
set xlabel "x"
set ylabel "y"
set view 60,30
unset colorbox
splot py using 1:2:3 with points pt 7 ps 0.2 lc palette notitle

# ---------- 中: C の結果 ----------
set title "C (out/numerical_c.dat)"
set colorbox
splot c using 1:2:3 with points pt 7 ps 0.2 lc palette notitle

# ---------- 右: x = i/n での断面を重ねる ----------
# 2つの線が重なっていれば, Python と C の結果が一致している
set title sprintf("cross section at x = %.4f", real(i)/n)
set xlabel "y"
set ylabel "phi"
set yrange [0:1]
unset colorbox
# x が i/n に一番近い行だけを使い, それ以外は 1/0 (欠損値) にして描かない
plot py using 2:(abs($1 - real(i)/n) < 0.5/n ? $3 : 1/0) with lines lw 4 lc rgb "#1f77b4" title "Python", \
     c  using 2:(abs($1 - real(i)/n) < 0.5/n ? $3 : 1/0) with lines lw 2 dt 2 lc rgb "#d62728" title "C"

unset multiplot

pause -1
# set output
