"""
out/ の各結果どうしの差を out/diff_<a>-<b>.dat に保存し, MSE を表示する
あわせて各手法と数値解との MSE を out/mse.dat に保存する (plot/2d/mse.plt で棒グラフにする)
差をとる組と MSE を出す手法は src/experiments.py の DIFF_PAIRS, MSE_TYPES
使い方: python -m script.make_diff
"""
import numpy as np

import src.config as config
import src.experiments as experiments
import src.sampling as sampling
import src.util as util

def load_phi(name:str, X:np.ndarray) -> np.ndarray:
    """
    out/{name}.dat の φ の列を読む
    点の数や座標が X と違う (別の POINTS_PER_SIDE で作った古いファイルなど) と, 同じ行どうしの比較が
    別の点どうしの比較になってしまうので, その場合はエラーにする
    """
    data = np.loadtxt(config.ROOT / f"out/{name}.dat")
    if data.shape[0] != X.shape[0] or not np.allclose(data[:,:2], X, rtol=0, atol=1e-5):
        raise ValueError(
            f"out/{name}.dat の点が出力格子と一致しません: {data.shape[0]} 点 (期待 {X.shape[0]} 点). "
            f"同じ POINTS_PER_SIDE で作り直してください"
        )
    return data[:,2:]

def run() -> None:
    cfg = config.load(config.ROOT / "config/templete.yaml")
    out = config.ROOT / "out"

    # 数値解・PINN と同じ出力点
    X = sampling.output_grid(cfg.divisions())

    width = max(len(experiments.diff_name(a, b)) for a, b in experiments.DIFF_PAIRS)
    for a, b in experiments.DIFF_PAIRS:
        name = experiments.diff_name(a, b)
        diff = load_phi(a, X) - load_phi(b, X)
        util.save_grid(out / f"{name}.dat",X,diff)
        print(f"MSE {name:<{width}} : {(diff**2).mean()}")

    # 1列目は out/ のファイル名, 2列目が MSE
    numerical = load_phi("numerical", X)
    with open(out / "mse.dat","w") as fp:
        for t in experiments.MSE_TYPES:
            mse = ((load_phi(t, X) - numerical)**2).mean()
            fp.write(f"{t} {mse:.6e}\n")

if __name__ == "__main__":
    run()
