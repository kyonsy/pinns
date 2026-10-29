import sys
import numpy as np
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.config as config
import src.sampling as sampling
import src.util as util

cfg = config.load(ROOT / "config/templete.yaml")

n = cfg.divisions()    # 分割数. 1辺の点数は n+1

# 数値解・PINN と同じ出力点
X = sampling.output_grid(n)

def load_phi(name:str) -> np.ndarray:
    """
    out/{name}.dat の φ の列を読む
    点の数や座標が X と違う (別の POINTS_PER_SIDE で作った古いファイルなど) と, 同じ行どうしの比較が
    別の点どうしの比較になってしまうので, その場合はエラーにする
    """
    data = np.loadtxt(ROOT / f"out/{name}.dat")
    if data.shape[0] != X.shape[0] or not np.allclose(data[:,:2], X, rtol=0, atol=1e-5):
        raise ValueError(
            f"out/{name}.dat の点が出力格子と一致しません: {data.shape[0]} 点 (期待 {X.shape[0]} 点, POINTS_PER_SIDE={n+1}). "
            f"同じ POINTS_PER_SIDE で作り直してください"
        )
    return data[:,2:]

g         = load_phi("g")
gl        = load_phi("gl")
hard1     = load_phi("hard1")
hard2     = load_phi("hard2")
numerical = load_phi("numerical")

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
        mse = ((load_phi(t) - numerical)**2).mean()
        fp.write(f"{t} {mse:.6e}\n")
