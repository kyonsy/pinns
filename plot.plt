set title "y-t"

set xlabel "y"
set ylabel "t"

set xrange [0:1]
set yrange [0:1]

set grid
set terminal pngcairo
set output "graph.png"

plot "out.txt" using 1:2 with lines title "y(t)"

set output


