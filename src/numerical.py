"""
ポアソンの方程式 φ_xx + φ_yy = 0 を差分法で解く

領域は [0,1]x[0,1]. 間隔 h=1/n で (n+1)x(n+1) 個の格子点をとり, phi[i,j] に点 (i/n, j/n) での φ を入れる
境界条件は PINN と同じ (外周で φ=0, 中心の4点で φ=1)

5点差分で離散化すると, 値が決まっていない各点で
    φ[i,j] = (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1]) / 4
となる. これを解く方法を2つ用意している
    solve_jacobi : ヤコビ法 (上の式で全点を更新する操作を繰り返す)
    solve_scipy  : 連立一次方程式として scipy で一度に解く (速い)
どちらも [x, y, φ] を並べた (n+1)^2 x 3 の配列を返す. 点の並びは sampling.output_grid と同じ
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import sampling


def boundary_condition(n:int):
    """
    境界条件を入れた phi と, 値が決まっている点を True にした fixed を返す
    phi は外周が0, 中心の4点が1, それ以外は初期値0
    """
    phi   = np.zeros((n+1, n+1))
    fixed = np.zeros((n+1, n+1), dtype=bool)

    # 外周で φ=0
    fixed[0, :] = True
    fixed[n, :] = True
    fixed[:, 0] = True
    fixed[:, n] = True

    # 中心の4点で φ=1
    for i in sampling.center_indices(n):
        for j in sampling.center_indices(n):
            phi[i, j]   = 1.0
            fixed[i, j] = True

    return phi, fixed


def to_table(phi):
    """phi[i,j] を [x, y, φ] の表に並べ直す"""
    n = phi.shape[0] - 1
    return np.column_stack([sampling.output_grid(n), phi.ravel()])


def solve_jacobi(n:int, tol:float=1e-10, max_iter:int=100000):
    """
    ヤコビ法. 内側の各点を上下左右4点の平均で置き換える操作を, 1回の変化の最大値が tol を下回るまで繰り返す
    ヤコビ法は収束が遅く, 変化が小さくてもまだ解から離れていることがあるので tol は小さめにとる
    """
    phi, fixed = boundary_condition(n)

    for num_iter in range(1, max_iter+1):
        new_phi = phi.copy()
        new_phi[1:-1, 1:-1] = (phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, :-2]) / 4
        new_phi[fixed] = phi[fixed]     # 境界条件で決まっている点は戻す

        change = np.abs(new_phi - phi).max()
        phi = new_phi
        if change < tol:
            break

    print(f"Jacobi: n={n} iter={num_iter} change={change:.3e}")
    return to_table(phi)


def solve_scipy(n:int):
    """
    連立一次方程式 A φ = b として scipy の spsolve で一度に解く
    5点差分の行列 L は, 1次元の2階差分 D をクロネッカー積で2次元に広げて作る
    値が決まっている点の寄与は右辺 b に移し, 決まっていない点だけを未知数にする
    """
    phi, fixed = boundary_condition(n)

    D = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n+1, n+1))
    I = sp.identity(n+1)
    L = (sp.kron(D, I) + sp.kron(I, D)).tocsr()

    known   = fixed.ravel()
    unknown = ~known
    A = L[unknown][:, unknown]
    b = -L[unknown][:, known] @ phi.ravel()[known]

    phi.ravel()[unknown] = spla.spsolve(A.tocsc(), b)
    return to_table(phi)
