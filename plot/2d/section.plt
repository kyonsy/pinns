# y=0.5 の断面で複数の解を重ねて描き, ./graph/2d/section_<file1 の拡張子を除いた名前>.png に保存する
# 使い方: gnuplot -c ./plot/2d/section.plt <file1> <file2> ...   (入力は ./out/<file>, 最大 9 個)
#
# 格子は y=i/n (n = POINTS_PER_SIDE - 1 は奇数) なので y=0.5 ちょうどの点はない
# 両隣の 2 行 (i=(n-1)/2, (n+1)/2) を smooth unique で x ごとに平均し, y=0.5 での値とする (線形補間と同じ)

if (!exists("n")) n = 101    # 分割数 = POINTS_PER_SIDE - 1. gnuplot -e "n=..." -c ... で変えられる

types = "hc_g.dat hard_g.dat numerical.dat"

# コマンドライン引数がある場合は、そちらを types に代入
if (ARGC >= 1) {
      types = ""
      do for [k=1:ARGC] { types = types.value("ARG".k)." " }
}

# ファイル名
# 拡張子を除いた名前 (出力する png の名前や凡例に使う)
stem(f) = strstrt(f, ".") ? f[1:strstrt(f, ".")-1] : f
in(t) = "./out/".t
out   = "./graph/2d/section_".stem(word(types,1)).".png"

i0 = (n-1)/2    # y=0.5 の下側の行 (整数割り算)

set xlabel "x"
set ylabel "f"
set xrange [0:1]
set yrange [0:1.05]
set grid
set key top right box opaque
set title "y = 0.5"

# 各ファイルから y=0.5 の両隣の 2 行を取り出して $sec にまとめる
# ファイルは x ごとのブロック (空行区切り) なので, そのままだと 1 ブロック 1 点になり線がつながらない
# 空行を除いて 1 本の線にし, 解ごとに空行 2 つで区切る (plot の index で選べるようにする)
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

set terminal pngcairo size 800,600
set output out

plot for [k=1:words(types)] $sec index k-1 using 1:2 smooth unique with lines lw 2 title stem(word(types,k)) noenhanced   # noenhanced: 名前の _ を下付き文字にしない
