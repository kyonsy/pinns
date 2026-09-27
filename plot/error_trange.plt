# 学習区間ごとの PINN と解析解の誤差 |y_pinn - y_exact| (ζ=0)
# 入力: out/out<j>.dat (列: t pinn exact abs_err)  ← script/dump_sys_trange.py の出力
#       学習区間は [0, jπ/4)
# 実行: プロジェクトルートで  gnuplot plot/error_trange.plt

N = 1   # dump_sys_trange.py のループ回数と揃える

set title "PINN error vs training range (ζ=0)"
set xlabel "t"
set ylabel "|y_{PINN} - y_{exact}|"

set xtics pi/4
set format x '%.2Pπ'

set logscale y
set format y '10^{%L}'

# 見分けやすい線の色 (gnuplot 既定の5色目は薄い黄色で見えにくいため)
set linetype 1 lc rgb "#1f77b4" lw 2
set linetype 2 lc rgb "#ff7f0e" lw 2
set linetype 3 lc rgb "#2ca02c" lw 2
set linetype 4 lc rgb "#d62728" lw 2
set linetype 5 lc rgb "#9467bd" lw 2
set linetype cycle 5

set grid xtics ytics
set key outside right
set terminal pngcairo enhanced size 900,500
set output "graph/error_trange.png"

plot for [j=1:N] sprintf("out/out%d.dat",j) using 1:4 with lines title sprintf("[0,%dπ/4)",j)

set output
