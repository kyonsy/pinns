set xrange [0:1]
set yrange [0:1]
set zrange [0:1]

set isosamples 40
set hidden3d
set view 60,30
# set terminal pngcairo size 800,600
set output "./graph/graph3d.png"
# splot exp(-x**2-y**2) 
splot "./out/out.dat" using 1:2:3 

pause -1
# set output
