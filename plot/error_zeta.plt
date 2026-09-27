# 減衰比ζごとの PINN と解析解の誤差 |y_pinn - y_exact|
# 入力: out/zeta<ζ>.dat (列: t pinn exact abs_err)  ← script/dump_sys_zeta.py の出力
# 実行: プロジェクトルートで  gnuplot plot/error_zeta.plt

zetas = "0.0 0.5 1.0 1.5 2.0"   # dump_sys_zeta.py の zetas と揃える

set title "PINN error vs damping ratio"
set xlabel "t"
set ylabel "|y_{PINN} - y_{exact}|"

set xrange [0:2*pi]
set xtics pi/2
set format x '%.1Pπ'
set mxtics 2

set logscale y
set format y '10^{%L}'

# 見分けやすい線の色 (gnuplot 既定の5色目は薄い黄色で見えにくいため)
set linetype 1 lc rgb "#1f77b4" lw 2
set linetype 2 lc rgb "#ff7f0e" lw 2
set linetype 3 lc rgb "#2ca02c" lw 2
set linetype 4 lc rgb "#d62728" lw 2
set linetype 5 lc rgb "#9467bd" lw 2
set linetype cycle 5

set grid xtics mxtics ytics
set key outside right
set terminal pngcairo enhanced size 900,500
set output "graph/error_zeta.png"

plot for [z in zetas] "out/zeta".z.".dat" using 1:4 with lines title "ζ=".z

set output
