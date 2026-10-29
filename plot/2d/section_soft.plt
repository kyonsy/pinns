# ソフト制約の PINN (境界の重みを変えたもの) と数値解を y=0.5 の断面で比べる
# y=0.5 の取り方は section.plt と同じ (両隣の 2 行を x ごとに平均する)
# 使い方: gnuplot -c ./plot/2d/section_soft.plt [<file1> <file2> ...]   → ./graph/2d/section_soft.png (最大 5 個)

set encoding utf8
if (!exists("n")) n = 101    # 分割数 = POINTS_PER_SIDE - 1. gnuplot -e "n=..." -c ... で変えられる
i0 = (n-1)/2    # y=0.5 の下側の行 (整数割り算)

types = "numerical.dat soft_w1.dat soft_w1e2.dat soft_w1e4.dat soft_w1e6.dat"

# コマンドライン引数がある場合は、そちらを types に代入 (./out/ の中のファイル名)
if (ARGC >= 1) {
      types = ""
      do for [k=1:ARGC] { types = types.value("ARG".k)." " }
}

# out/ のファイル名 → 凡例
title(t) = t eq "numerical.dat" ? "数値解"   : \
           t eq "soft_w1.dat"   ? "λ = 1"    : \
           t eq "soft_w1e2.dat" ? "λ = 10^2" : \
           t eq "soft_w1e4.dat" ? "λ = 10^4" : \
           t eq "soft_w1e6.dat" ? "λ = 10^6" : t
array LC[5] = ["black", "#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]   # 線の色. 白背景で見やすい濃い色 (黒, 赤, 橙, 緑, 青)

# 各ファイルから y=0.5 の両隣の 2 行を取り出して $sec にまとめる (section.plt と同じ)
in(t) = "./out/".t
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
set key top right box opaque

plot for [k=1:words(types)] $sec index k-1 using 1:2 smooth unique \
     with lines lw 2 lc rgb LC[k] title title(word(types,k))
