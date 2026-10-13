import sys
import numpy as np
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.config as cfg
import src.util as util

cfg.load(ROOT / "config/templete.yaml")

n = cfg.GRID_DENSITY

x, y = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
X =np.column_stack([x.ravel(), y.ravel()])

g         = np.loadtxt("./out/g.dat")[:,2:]
gl        = np.loadtxt("./out/gl.dat")[:,2:]
hard1     = np.loadtxt("./out/hard1.dat")[:,2:]
hard2     = np.loadtxt("./out/hard2.dat")[:,2:]
numerical = np.loadtxt("./out/numerical.dat")[:,2:]

diff_hard1_g         = hard1 - g
diff_hard2_gl        = hard2 - gl
diff_hard1_numerical = hard1 - numerical
diff_hard2_numerical = hard2 - numerical
diff_g_numerical     = g     - numerical
diff_gl_numerical    = gl    - numerical


mse_hard1_g          = (diff_hard1_g**2).mean()
mse_hard2_gl         = (diff_hard2_gl**2).mean()
mse_hard1_numerical  = (diff_hard1_numerical**2).mean()
mse_hard2_numerical  = (diff_hard2_numerical**2).mean()
mse_g_numerical      = (diff_g_numerical**2).mean()
mse_gl_numerical     = (diff_gl_numerical**2).mean()

util.save_grid(ROOT / "out/diff_hard1_g.dat",X,diff_hard1_g)
util.save_grid(ROOT / "out/diff_hard2_gl.dat",X,diff_hard2_gl)
util.save_grid(ROOT / "out/diff_hard1_numerical.dat",X,diff_hard1_numerical)
util.save_grid(ROOT / "out/diff_hard2_numerical.dat",X,diff_hard2_numerical)
util.save_grid(ROOT / "out/diff_g_numerical.dat",X,diff_g_numerical)
util.save_grid(ROOT / "out/diff_gl_numerical.dat",X,diff_gl_numerical)

print(f"mse_hard1_g          :{mse_hard1_g}")
print(f"mse_hard2_gl         :{mse_hard2_gl}")

print(f"mse_hard1_numerical  :{mse_hard1_numerical}")
print(f"mse_hard2_numerical  :{mse_hard2_numerical}")

print(f"mse_g_numerical      :{mse_g_numerical}")
print(f"mse_gl_numerical     :{mse_gl_numerical}")


# 各手法と数値解との MSE を並べて保存する (plot/2d/mse.plt で棒グラフにする)
# 1列目は out/ のファイル名, 2列目が MSE. 行の順番がそのまま棒の並び順になる
mse_types = ["soft1","soft2","soft3","soft4","hard1","hard2"]
with open(ROOT / "out/mse.dat","w") as fp:
    for t in mse_types:
        mse = ((np.loadtxt(f"./out/{t}.dat")[:,2:] - numerical)**2).mean()
        fp.write(f"{t} {mse:.6e}\n")
