set view map
set size ratio -1
set xlabel "x"
set ylabel "y"
set cblabel "f"

file="diff_hard_g-numerical.dat"

# コマンドライン引数がある場合は、そちらを file に代入 (./out/ の中のファイル名)
if (ARGC >= 1) {
      file = ARG1
}

# ファイル名
# 拡張子を除いた名前 (出力する png の名前や凡例に使う)
stem(f) = strstrt(f, ".") ? f[1:strstrt(f, ".")-1] : f
in  = "./out/".file
out = "./graph/2d/".stem(file).".png"

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