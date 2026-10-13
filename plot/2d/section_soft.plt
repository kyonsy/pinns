# ソフト制約の PINN (soft1〜4) と数値解を y=0.5 の断面で比べる
# y=0.5 の取り方は section.plt と同じ (両隣の 2 行を x ごとに平均する)
# 使い方: gnuplot -c ./plot/2d/section_soft.plt   → ./graph/2d/section_soft.png

set encoding utf8
n = 101    # GRID_DENSITY
i0 = (n-1)/2    # y=0.5 の下側の行 (整数割り算)

types = "numerical soft1 soft2 soft3 soft4"

# 凡例と線の種類 (types と同じ順番)
array TITLE[5] = ["数値解", "λ = 1", "λ = 10^2", "λ = 10^4", "λ = 10^6"]
array LC[5] = ["black", "#a0a0a0", "#808080", "#505050", "#1f5fbf"]
array DT[5] = [1, 3, 4, 2, 2]     # 1:実線 2:破線 3:点線 4:一点鎖線
array LW[5] = [3, 3, 2.5, 2.5, 3]

# 各ファイルから y=0.5 の両隣の 2 行を取り出して $sec にまとめる (section.plt と同じ)
in(t) = "./out/".t.".dat"
set print $sec
do for [t in types] {
      set table $raw
      plot in(t) using 1:3 every ::i0::i0+1 with table
      unset table
      do for [k=1:|$raw|] {
            if (words($raw[k]) > 0) { print $raw[k] }
      }
      print ""
      print ""
}
unset print

set terminal pngcairo size 1200,900 font "Meiryo,16" enhanced
set output "./graph/2d/section_soft.png"

set title "y = 0.5 での断面"
set xlabel "x"
set ylabel "φ" norotate
set xrange [0:1]
set yrange [0:1.05]
set xtics 0.2
set ytics 0.2
set format x "%.1f"
set format y "%.1f"
set grid xtics ytics lt 1 lw 1 lc rgb "#e0e0e0"
set border 3
set tics nomirror
set key below maxrows 2 width 2

plot for [k=1:words(types)] $sec index k-1 using 1:2 smooth unique \
     with lines lw LW[k] dt DT[k] lc rgb LC[k] title TITLE[k]
