# φ を pm3d で曲面として描き, ./graph/3d/<type>.png に保存する
# 使い方: gnuplot -c ./plot/3d/surface.plt <type>   (入力は ./out/<type>.dat)

type="soft_w1"

# コマンドライン引数がある場合は、そちらを type に代入
if (ARGC >= 1) {
      type = ARG1
}

# ファイル名
in  = "./out/".type.".dat"
out = "./graph/3d/".type.".png"

set xlabel "x"
set ylabel "y"
set zlabel "f" rotate parallel
set cblabel "f"
unset key

set xrange [0:1]
set yrange [0:1]
set zrange [0:1]
set cbrange [0:1]
set xyplane at 0              # z 軸の底を xy 平面に合わせる
set view 60,30

set terminal pngcairo size 1000,800 enhanced
set output out

set palette defined (0 "navy" ,1 "blue",3 "yellow",4 "red")
set pm3d depthorder           # 奥の面から順に描いて隠面を正しく処理する
set pm3d lighting primary 0.5 specular 0.2   # 陰影をつけて凹凸を見やすくする
set pm3d border lw 0.3 lc rgb "#40000000"     # 格子線をうっすら重ねる (#AARRGGBB)

splot in using 1:2:3 with pm3d
