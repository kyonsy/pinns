set title "y-t"

set xlabel "y"
set ylabel "t"

set xrange [0:10*pi]
set yrange [-1:1]

set xtics pi/2
set format x '%.1Pπ'
set mxtics 2

set grid xtics mxtics ytics
set terminal pngcairo
set output "graph.png"

plot for [i=1:10] sprintf("out/out%d.dat",i*4) using 1:2 with lines title sprintf("data%d",i*4)

set output


