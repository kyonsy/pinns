# pinn/ から実行する

# python の実行
$soft_types = @("soft1","soft2","soft3","soft4")

foreach($type in $soft_types){
    python -u "./script/poisson_soft.py" $type
}
python -u "./script/poisson_numerical.py"
python -u "./script/poisson_hard.py"
python -u "./script/poisson_check_G.py"



