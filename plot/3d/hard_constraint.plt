# ハード制約の説明図 (3D 版). 色や視点は plot/3d/surface.plt とそろえている
#   左: G(x,y) = D/D(p_k)            (./out/g.dat)
#   右: 中心付近を拡大した C(x,y)     (./out/hc_C_center.dat)
# 先に python script/export_hard_constraint.py でデータを作っておく
# 使い方: gnuplot -c ./plot/3d/hard_constraint.plt   → ./graph/3d/hard_constraint.png

set encoding utf8
n = 101    # 分割数 = POINTS_PER_SIDE - 1
a = ceil(n/2.0)/real(n)     # 中心の4点の座標 (hard_constraint.param と同じ). real で整数割り算を避ける
b = floor(n/2.0)/real(n)

set terminal pngcairo size 1600,700 font "Meiryo,14" enhanced
set output "./graph/3d/hard_constraint.png"
set multiplot layout 1,2

# ---- 共通の設定 ----
set xlabel "x"
set ylabel "y"
unset key
set zrange [0:1]
set cbrange [0:1]
set xyplane at 0
set view 60,30
set palette defined (0 "navy" ,1 "blue",3 "yellow",4 "red")
set pm3d depthorder
set pm3d lighting primary 0.5 specular 0.2
set pm3d border lw 0.3 lc rgb "#40000000"
unset colorbox
set pm3d scansautomatic             # depthorder だと p_k の印が曲面に隠れるので切る (上からのぞくので重なりは起きない)

# ---- 左: G 全体 ----
set title "G(x,y) = D/D(p_k)　（外周で0, 中心の4点で1）"
set xrange [0:1]
set yrange [0:1]
set xtics 0.5
set ytics 0.5
set ztics 0.5
splot "./out/g.dat" using 1:2:3 with pm3d

# ---- 右: 中心付近の C ----
set title "C(x,y)　中心付近を拡大　（4点 p_k で0）"
h = 1/real(n)
set xrange [b-1.5*h : a+1.5*h]      # 平らな部分を減らしてくぼみを大きく見せる
set yrange [b-1.5*h : a+1.5*h]
set view 20,30                      # 上からのぞき込んで底の4点のくぼみを見えるようにする
set xtics ("b" b, "a" a)
set ytics ("b" b, "a" a)
set colorbox
# 4点 p_k に印を付ける (点のあいだの盛り上がりは 0.03 程度しかなく, 曲面だけでは位置が分かりにくい)
# 曲面に隠れないよう少し浮かせる
set print $P
do for [px in "a b"] {
      do for [py in "a b"] {
            print sprintf("%.8f %.8f 0.05", value(px), value(py))
      }
}
unset print
splot "./out/hc_C_center.dat" using 1:2:3 with pm3d, \
      $P using 1:2:3 with points pt 7 ps 1.5 lc "white"

unset multiplot
