set view map
set size ratio -1
set xlabel "x"
set ylabel "y"
set cblabel "f"

file ="./out/out_numerical.dat"

set contour base
set cntrparam levels 10
unset key

set palette defined (0 "navy" ,1 "blue",3 "yellow",4 "red")
set pm3d interpolate 2,2

splot file using 1:2:3 with pm3d nocontours, \
      file using 1:2:3 with lines nosurface lc "black"

pause -1