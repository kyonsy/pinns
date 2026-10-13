# 各手法と数値解との MSE を対数軸の棒グラフにする
# 先に python script/make_diff.py で ./out/mse.dat を作っておく
# 使い方: gnuplot -c ./plot/2d/mse.plt   → ./graph/2d/mse.png

set encoding utf8

# out/ のファイル名 → 棒の下に出すラベル
label(t) = t eq "soft1" ? "ソフト\nλ = 1"      : \
           t eq "soft2" ? "ソフト\nλ = 10^2"   : \
           t eq "soft3" ? "ソフト\nλ = 10^4"   : \
           t eq "soft4" ? "ソフト\nλ = 10^6"   : \
           t eq "hard1" ? "ハード1\n(G)"       : \
           t eq "hard2" ? "ハード2\n(G·L)"     : t

# 強調する棒 (hard2) だけ青, ほかは灰色
color(t) = t eq "hard2" ? 0x1f5fbf : 0xa0a0a0

set terminal pngcairo size 1200,800 font "Meiryo,16" enhanced
set output "./graph/2d/mse.png"

set ylabel "数値解との MSE"
set logscale y
set yrange [5e-4:1]
set format y "10^{%L}"
set xrange [-0.6:5.6]
set xtics nomirror scale 0 offset 0,-0.5
set ytics nomirror
set border 3
set bmargin 6
unset key

set style fill solid 1.0 noborder
set boxwidth 0.6

plot "./out/mse.dat" using 0:2:(color(strcol(1))):xtic(label(strcol(1))) with boxes lc rgb variable, \
     ""              using 0:($2*1.35):(gprintf("%.1t×10^{%T}", $2)) with labels
