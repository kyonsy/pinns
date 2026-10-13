# pinn/ から実行する
# gnuplot の実行. 2d は ./graph/2d/, 3d は ./graph/3d/ に保存する
$types = @("soft1","soft2","soft3","soft4","numerical","hard1","hard2","g","gl")
$diff_types = @("diff_hard1_g","diff_hard2_gl","diff_hard1_numerical","diff_hard2_numerical",
                "diff_g_numerical","diff_gl_numerical")

# gnuplot は保存先のフォルダを作らないので先に作っておく
New-Item -ItemType Directory -Force "./graph/2d","./graph/3d" | Out-Null

foreach($type in $types){
    gnuplot -c "./plot/2d/contour.plt" $type
    gnuplot -c "./plot/3d/surface.plt" $type
}

foreach($type in $diff_types){
    gnuplot -c "./plot/2d/diff.plt" $type
    gnuplot -c "./plot/3d/diff.plt" $type
}

# y=0.5 の断面
gnuplot -c "./plot/2d/section.plt" g hard1 numerical
gnuplot -c "./plot/2d/section.plt" gl hard2 numerical

# ハード制約の説明図 (G の等高線図と中心付近の C, G の断面)
gnuplot -c "./plot/2d/hard_constraint.plt"
gnuplot -c "./plot/3d/hard_constraint.plt"

# ハード制約の断面 (G, hard1, G·L, hard2, 数値解) と, 数値解との MSE の棒グラフ (先に script/make_diff.py で out/mse.dat を作っておく)
gnuplot -c "./plot/2d/section_hard.plt"
gnuplot -c "./plot/2d/mse.plt"
