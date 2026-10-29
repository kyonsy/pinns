set view map
set size ratio -1
set xlabel "x"
set ylabel "y"
set cblabel "f"

type="diff_hard_g-numerical"

# コマンドライン引数がある場合は、そちらを type に代入
if (ARGC >= 1) {
      type = ARG1
}

# ファイル名
in  = "./out/".type.".dat"
out = "./graph/2d/".type.".png"

set contour base
set cntrparam levels 10
unset key

# diff は負の値もあり範囲も小さいので, カラー範囲は自動にする
set autoscale cb
set terminal pngcairo size 800,600
set output out

set palette defined (0 "navy" ,1 "blue",3 "yellow",4 "red")
set pm3d interpolate 2,2

splot in using 1:2:3 with pm3d nocontours, \
      in using 1:2:3 with lines nosurface lc "black"