set view map
set size ratio -1
set xlabel "x"
set ylabel "y"
set cblabel "f"

type="g"
#ファイル名
in  = "./out/out_".type.".dat"
out = "./graph/graph_".type.".png"

set contour base
set cntrparam levels 10
unset key

set cbrange [0:1]
set terminal pngcairo size 800,600
set output out

set palette defined (0 "navy" ,1 "blue",3 "yellow",4 "red")
set pm3d interpolate 2,2

splot in using 1:2:3 with pm3d nocontours, \
      in using 1:2:3 with lines nosurface lc "black"