"""
ポアソンの方程式 φ_xx + φ_yy = 0 を差分法で解く

領域は単位正方形 [0,1]x[0,1]. 刻み幅 h=1/n で (n+1)x(n+1) 個の格子点をとり,
phi[i,j] に点 (x,y) = (i/n, j/n) での φ の値を入れる

境界条件は PoissonFCNN と同じ
    外周で φ=0
    中心の周りの4点 (a,a),(a,b),(b,a),(b,b) で φ=1   (a=ceil(n/2)/n, b=floor(n/2)/n)

5点差分で離散化すると, 値が決まっていない各点で
    φ[i,j] = (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1]) / 4
つまり「各点の値は上下左右4点の平均」という連立一次方程式になる. これを次の2通りで解く
    solve_poisson_jacobi : ヤコビ法. 全点を4点の平均で置き換える操作をひたすら繰り返す
    solve_poisson_scipy  : 連立一次方程式として scipy の疎行列ソルバーで一度に解く (速い)
どちらも戻り値は [x, y, φ] を並べた (n+1)^2 x 3 の配列. 並び順は sampling.sample_grid と同じ (x が外側, y が内側)
"""
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray


"""
境界条件を設定した初期状態を返す
    phi   : 外周は0, 中心の4点は1, それ以外は0 (初期値)
    fixed : 境界条件で値が決まっている点は True
"""
def boundary_condition(n:int) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    phi:   NDArray[np.float64] = np.zeros((n+1, n+1))
    fixed: NDArray[np.bool_]   = np.zeros((n+1, n+1), dtype=bool)

    # 外周で φ=0
    fixed[0, :]  = True   # x=0
    fixed[n, :]  = True   # x=1
    fixed[:, 0]  = True   # y=0
    fixed[:, n]  = True   # y=1

    # 中心の周りの4点で φ=1
    centers: list[int] = [math.ceil(n/2), math.floor(n/2)]
    i: int
    j: int
    for i in centers:
        for j in centers:
            phi[i, j]   = 1.0
            fixed[i, j] = True

    return phi, fixed


"""
phi[i,j] を [x, y, φ] の表に並べ直す
"""
def to_table(phi:NDArray[np.float64]) -> NDArray[np.float64]:
    n: int = phi.shape[0] - 1
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = np.meshgrid(np.arange(n+1) / n, np.arange(n+1) / n, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), phi.ravel()])


"""
ヤコビ法
各点を上下左右4点の平均で置き換える操作を, 値がほとんど変わらなくなるまで繰り返す
    tol : 1回の更新での変化の最大値がこれを下回ったら終了
          ヤコビ法は収束が遅く, 変化が小さくても正しい解からまだ離れていることがあるので小さめにとる
"""
def solve_poisson_jacobi(n:int, tol:float=1e-10, max_iter:int=100000) -> NDArray[np.float64]:
    phi:   NDArray[np.float64]
    fixed: NDArray[np.bool_]
    phi, fixed = boundary_condition(n)

    num_iter: int = 0          # 実際に繰り返した回数
    change: float = math.inf
    for _ in range(max_iter):
        # 内側の各点 (i,j) について, 上下左右の点の値を取り出す
        right: NDArray[np.float64] = phi[2:,   1:-1]   # φ[i+1, j]
        left:  NDArray[np.float64] = phi[:-2,  1:-1]   # φ[i-1, j]
        up:    NDArray[np.float64] = phi[1:-1, 2:  ]   # φ[i, j+1]
        down:  NDArray[np.float64] = phi[1:-1, :-2 ]   # φ[i, j-1]

        new_phi: NDArray[np.float64] = phi.copy()
        new_phi[1:-1, 1:-1] = (right + left + up + down) / 4

        # 境界条件で値が決まっている点は元に戻す
        new_phi[fixed] = phi[fixed]

        change = float(np.abs(new_phi - phi).max())
        phi = new_phi
        num_iter += 1
        if change < tol:
            break

    print(f"Jacobi: n={n} iter={num_iter} change={change:.3e}")
    return to_table(phi)


"""
連立一次方程式として一度に解く
「各点の値 = 4点の平均」を (4点の和) - 4φ[i,j] = 0 の形に並べて A φ = b とし, scipy の spsolve で解く
"""
def solve_poisson_scipy(n:int) -> NDArray[np.float64]:
    phi:   NDArray[np.float64]
    fixed: NDArray[np.bool_]
    phi, fixed = boundary_condition(n)

    # 5点差分の行列 L (L φ が各点での「4点の和 - 4φ[i,j]」になる)
    # 1次元の2階差分 D をクロネッカー積で2次元に広げて作る
    D: sp.dia_matrix = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n+1, n+1))
    I: sp.dia_matrix = sp.identity(n+1)
    L: sp.csr_matrix = (sp.kron(D, I) + sp.kron(I, D)).tocsr()

    # 値が決まっていない点だけを未知数にし, 決まっている点の寄与は右辺に移す
    known:   NDArray[np.bool_]   = fixed.ravel()
    unknown: NDArray[np.bool_]   = ~known
    A:       sp.csr_matrix       = L[unknown][:, unknown]
    b:       NDArray[np.float64] = -L[unknown][:, known] @ phi.ravel()[known]

    phi.ravel()[unknown] = spla.spsolve(A.tocsc(), b)
    return to_table(phi)

