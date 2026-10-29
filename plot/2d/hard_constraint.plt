# ハード制約の説明図
#   左: G(x,y) = D/D(p_k) の等高線図 (./out/g.dat)
#   右: 中心付近 y=b での C と G の断面 (./out/hc_section.dat)
# 先に python script/export_hard_constraint.py でデータを作っておく
# 使い方: gnuplot -c ./plot/2d/hard_constraint.plt   → ./graph/2d/hard_constraint.png

set encoding utf8
n = 101    # 分割数 = POINTS_PER_SIDE - 1
a = ceil(n/2.0)/real(n)     # 中心の4点の座標 (hard_constraint.param と同じ). real で整数割り算を避ける
b = floor(n/2.0)/real(n)

set terminal pngcairo size 1400,560 font "Meiryo,14" enhanced
set output "./graph/2d/hard_constraint.png"
set multiplot layout 1,2

# ---- 左: G の等高線図 ----
set title "G(x,y) = D/D(p_k)"
set view map
set size ratio -1
set xlabel "x"
set ylabel "y"
set xrange [0:1]
set yrange [0:1]
set cbrange [0:1]
set xtics 0.5
set ytics 0.5
set cbtics 0.5
unset key
set palette defined (0 "#f7fbff", 0.3 "#c6dbef", 0.6 "#6baed6", 0.8 "#2171b5", 1 "#08306b")   # 白→青→紺
set palette maxcolors 10                          # 10段階に塗り分けて等高線図らしくする
set pm3d interpolate 4,4                         # 格子の粗さを目立たなくする
splot "./out/g.dat" using 1:2:3 with pm3d

# ---- 右: 中心付近の断面 ----
reset
set title "中心付近での断面"
set xlabel "x （y = b での断面, 中心付近）"
set xtics ("b" b, "a" a)
set ytics 0.5
set yrange [-0.05:1.15]
set key bottom left noautotitle
set border 3                       # 左と下の枠だけ
set tics nomirror
set arrow from b, graph 0 to b, graph 1 nohead dt 3 lc "gray"
set arrow from a, graph 0 to a, graph 1 nohead dt 3 lc "gray"
plot "./out/hc_section.dat" using 1:2 with lines lw 3 lc "#d62728" title "C(x, b)", \
     ""                     using 1:3 with lines lw 3 lc "#1f5fbf" dt 2 title "G(x, b)"

unset multiplot
