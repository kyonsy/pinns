# 減衰比ζごとの PINN の出力と解析解
# 実線: PINN, 破線: 解析解 (同じζは同じ色)
# 入力: out/zeta<ζ>.dat (列: t pinn exact abs_err)  ← script/dump_sys_zeta.py の出力
# 実行: プロジェクトルートで  gnuplot plot/solution_zeta.plt

zetas = "0.0 0.5 1.0 1.5 2.0"   # dump_sys_zeta.py の zetas と揃える

set title "PINN solution vs exact solution"
set xlabel "t"
set ylabel "y"

set xrange [0:2*pi]
set yrange [-1.1:1.1]
set xtics pi/2
set format x '%.1Pπ'
set mxtics 2

# 見分けやすい線の色 (gnuplot 既定の5色目は薄い黄色で見えにくいため)
colors = "#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd"

set grid xtics mxtics ytics
set key outside right
set terminal pngcairo enhanced size 900,500
set output "graph/solution_zeta.png"

plot for [i=1:words(zetas)] "out/zeta".word(zetas,i).".dat" using 1:2 \
         with lines lw 2 lc rgb word(colors,i) title "PINN ζ=".word(zetas,i), \
     for [i=1:words(zetas)] "out/zeta".word(zetas,i).".dat" using 1:3 \
         with lines lw 1 dt 2 lc rgb word(colors,i) notitle,      NaN with lines lw 1 dt 2 lc rgb "black" title "exact"   # 凡例用のダミー

set output
