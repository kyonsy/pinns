# 学習区間ごとの PINN の出力と解析解 (ζ=0)
# 実線: PINN, 破線: 解析解 (同じ学習区間は同じ色)
# 入力: out/out<j>.dat (列: t pinn exact abs_err)  ← script/dump_sys_trange.py の出力
#       学習区間は [0, jπ/4)
# 実行: プロジェクトルートで  gnuplot plot/solution_trange.plt

N = 1   # dump_sys_trange.py のループ回数と揃える

set title "PINN solution vs exact solution (ζ=0)"
set xlabel "t"
set ylabel "y"

set yrange [-1.1:1.1]
set xtics pi/4
set format x '%.2Pπ'

# 見分けやすい線の色 (gnuplot 既定の5色目は薄い黄色で見えにくいため)
colors = "#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd"
color(j) = word(colors,(j-1)%words(colors)+1)

set grid xtics ytics
set key outside right
set terminal pngcairo enhanced size 900,500
set output "graph/solution_trange.png"

plot for [j=1:N] sprintf("out/out%d.dat",j) using 1:2 \
         with lines lw 2 lc rgb color(j) title sprintf("PINN [0,%dπ/4)",j), \
     for [j=1:N] sprintf("out/out%d.dat",j) using 1:3 \
         with lines lw 1 dt 2 lc rgb color(j) notitle,      NaN with lines lw 1 dt 2 lc rgb "black" title "exact"   # 凡例用のダミー

set output
