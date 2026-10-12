# gnuplot の実行
$types = @("soft1","soft2","soft3","soft4","numerical","hard","g")

foreach($type in $types){
    gnuplot "./plot/contour.plt" $type
}