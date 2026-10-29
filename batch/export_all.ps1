# pinn/ から実行する

# python の実行
$soft_types = @("soft1","soft2","soft3","soft4")

foreach($type in $soft_types){
    python -u "./script/soft.py" $type
}
python -u "./script/numerical.py"
python -u "./script/hard.py" hard1
python -u "./script/hard.py" hard2
python -u "./script/export_hard_constraint.py"
python -u "./script/make_diff.py"



