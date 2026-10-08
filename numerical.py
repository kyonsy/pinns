# -*- coding: utf-8 -*-
"""
numerical.py
============================================================
2次元 Poisson 方程式を「有限差分法 (FDM)」で数値的に解くプログラム
============================================================

■ 解く問題
------------------------------------------------------------
    -Δu(x, y) = f(x, y)      (x, y) ∈ Ω = (0, 1) × (0, 1)
          u(x, y) = g(x, y)      (x, y) ∈ ∂Ω   (Dirichlet 境界条件)

  ここで Δ はラプラシアン:
      Δu = ∂²u/∂x² + ∂²u/∂y²

  ※ 符号について:
     「-Δu = f」と負号をつけて書くと、離散化した行列が
     「正定値対称行列」になるので数値計算上扱いやすい。
     (Δu = f で書きたい場合は f の符号を反転すればよいだけ)

■ 検証用の厳密解 (製造解 / Manufactured Solution)
------------------------------------------------------------
    u_exact(x, y) = sin(πx) sin(πy)

  これをラプラシアンに代入すると
      ∂²u/∂x² = -π² sin(πx) sin(πy)
      ∂²u/∂y² = -π² sin(πx) sin(πy)
  なので
      -Δu = 2π² sin(πx) sin(πy)  =: f(x, y)

  また境界 (x=0,1 または y=0,1) では sin が 0 になるので g = 0。
  → 数値解と厳密解を比べて、誤差を評価できる!

■ 解法
------------------------------------------------------------
  1. 直接法 : 疎行列 (scipy.sparse) を組み立てて spsolve で一発で解く
  2. 反復法 : SOR 法 (Successive Over-Relaxation, 逐次過緩和法)
     → 行列を作らずに格子点を順番に更新していく古典的な方法

  どちらでも解けるようにしておき、結果を比較できるようにしている。

■ 必要なライブラリ
------------------------------------------------------------
    numpy, scipy, matplotlib
"""

# ------------------------------------------------------------
# ライブラリのインポート
# ------------------------------------------------------------
import time                       # 計算時間の計測用

import numpy as np                # 数値計算の基本ライブラリ
import scipy.sparse as sp         # 疎行列 (ほとんどの要素が 0 の行列) を扱う
import scipy.sparse.linalg as spla  # 疎行列用の線形ソルバー
import matplotlib.pyplot as plt   # 結果の可視化


# ============================================================
# 問題設定: ソース項 f, 境界値 g, 厳密解 u_exact
# ============================================================
def source_term(x, y):
    """
    右辺 (ソース項) f(x, y) を返す関数。

    -Δu = f の f にあたる。
    厳密解 u = sin(πx)sin(πy) から逆算した値を使っている。

    Parameters
    ----------
    x, y : ndarray
        座標 (同じ形状の配列)

    Returns
    -------
    ndarray
        各点での f の値
    """
    return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def boundary_value(x, y):
    """
    境界上の値 g(x, y) を返す関数 (Dirichlet 境界条件)。

    今回の厳密解では境界上で u = 0 なので、0 を返すだけ。
    別の問題を解きたいときは、ここを書き換えればよい。
    例: return x**2 + y**2 など
    """
    return np.zeros_like(x)


def exact_solution(x, y):
    """
    厳密解 u_exact(x, y) = sin(πx) sin(πy)。
    数値解の誤差評価に使う。
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# ============================================================
# 格子 (メッシュ) の生成
# ============================================================
def make_grid(n):
    """
    正方形領域 [0,1]×[0,1] 上に、等間隔の格子を作る。

    Parameters
    ----------
    n : int
        各方向の「分割数」。格子点の数は各方向 n+1 個になる。
        (境界の点も含む)

        例: n = 4 のとき x = 0, 0.25, 0.5, 0.75, 1.0

    Returns
    -------
    x, y : ndarray, shape (n+1, n+1)
        各格子点の座標
    h : float
        格子幅 (= 1/n)
    """
    h = 1.0 / n                          # 格子幅
    coords = np.linspace(0.0, 1.0, n + 1)  # 0 から 1 まで n+1 点

    # meshgrid で 2次元の座標配列を作る。
    # indexing="ij" にすると、x[i, j] が i 番目の x 座標、
    # y[i, j] が j 番目の y 座標になる (行列の添字と直感的に対応)。
    x, y = np.meshgrid(coords, coords, indexing="ij")
    return x, y, h


# ============================================================
# 解法 1: 疎行列を組み立てて直接法で解く
# ============================================================
def build_laplacian_matrix(m, h):
    """
    内部格子点 (境界を除いた点) に対する離散ラプラシアン行列 A を作る。

    ■ 5点差分 (five-point stencil)
      2階微分を中心差分で近似すると
        ∂²u/∂x² ≈ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / h²
        ∂²u/∂y² ≈ (u[i,j+1] - 2u[i,j] + u[i,j-1]) / h²
      よって
        -Δu ≈ (4u[i,j] - u[i+1,j] - u[i-1,j] - u[i,j+1] - u[i,j-1]) / h²

      図で書くと (係数 × 1/h²):
                    -1
                -1   4  -1
                    -1

      この近似の誤差 (打ち切り誤差) は O(h²) 、つまり2次精度。

    ■ クロネッカー積による組み立て
      1次元の 2階差分行列 T (m×m) を
          T = (1/h²) * tridiag(-1, 2, -1)
      とすると、2次元のラプラシアン行列は
          A = I ⊗ T + T ⊗ I
      と書ける (⊗ はクロネッカー積)。
      ループで1行ずつ書くより、短くて速くてバグりにくい!

    Parameters
    ----------
    m : int
        各方向の内部格子点の数 (= n - 1)
    h : float
        格子幅

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (m*m, m*m)
        離散ラプラシアン行列 (正定値対称)
    """
    # 1次元の 2階差分行列 tridiag(-1, 2, -1) / h² を作る
    main_diag = 2.0 * np.ones(m)    # 対角成分
    off_diag = -1.0 * np.ones(m - 1)  # 対角の1つ上・1つ下
    T = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1]) / h**2

    # m×m の単位行列
    I = sp.identity(m)

    # クロネッカー積で 2次元化。
    # 未知数の並べ方は「k = i*m + j」 (i: x方向, j: y方向)。
    # これは numpy の reshape (C順序) と一致する。
    A = sp.kron(T, I) + sp.kron(I, T)

    # CSR 形式 (行方向に圧縮した形式) は行列ベクトル積やソルバーに向いている
    return A.tocsr()


def solve_direct(n):
    """
    直接法 (疎行列 LU 分解) で Poisson 方程式を解く。

    手順:
      1. 格子を作る
      2. 内部点の右辺ベクトル b を作る (境界値の寄与も足し込む)
      3. 行列 A を作る
      4. A u = b を解く
      5. 境界値と内部の解をまとめて 2次元配列に戻す

    Parameters
    ----------
    n : int
        各方向の分割数

    Returns
    -------
    x, y : ndarray
        格子点座標
    u : ndarray, shape (n+1, n+1)
        数値解 (境界も含む)
    """
    x, y, h = make_grid(n)
    m = n - 1  # 内部格子点の数 (各方向)

    # --- 解の配列を用意し、まず境界値を代入しておく ---
    u = np.zeros((n + 1, n + 1))
    u[0, :] = boundary_value(x[0, :], y[0, :])    # x = 0 の辺
    u[-1, :] = boundary_value(x[-1, :], y[-1, :])  # x = 1 の辺
    u[:, 0] = boundary_value(x[:, 0], y[:, 0])    # y = 0 の辺
    u[:, -1] = boundary_value(x[:, -1], y[:, -1])  # y = 1 の辺

    # --- 右辺ベクトル b (内部点のみ) ---
    # [1:-1, 1:-1] で境界を除いた内部点だけを取り出す
    b = source_term(x[1:-1, 1:-1], y[1:-1, 1:-1]).copy()

    # --- 境界値の寄与を右辺に移す ---
    # 境界に隣接する内部点では、5点差分のうち1つ (角なら2つ) が
    # 境界上の「既知の値」になる。既知の項は右辺へ移項する:
    #   (4u[i,j] - u[境界] - ...) / h² = f
    #   → (4u[i,j] - ...) / h² = f + u[境界] / h²
    b[0, :] += u[0, 1:-1] / h**2     # 左側の境界 (x = 0) に隣接
    b[-1, :] += u[-1, 1:-1] / h**2   # 右側の境界 (x = 1) に隣接
    b[:, 0] += u[1:-1, 0] / h**2     # 下側の境界 (y = 0) に隣接
    b[:, -1] += u[1:-1, -1] / h**2   # 上側の境界 (y = 1) に隣接

    # --- 行列を作って解く ---
    A = build_laplacian_matrix(m, h)
    # b.ravel() で 2次元配列 (m, m) を 1次元ベクトル (m*m,) に並べ替える
    u_inner = spla.spsolve(A, b.ravel())

    # --- 1次元ベクトルを 2次元に戻して、内部点に書き込む ---
    u[1:-1, 1:-1] = u_inner.reshape(m, m)

    return x, y, u


# ============================================================
# 解法 2: SOR 法 (反復法)
# ============================================================
def solve_sor(n, omega=None, tol=1e-10, max_iter=100000):
    """
    SOR 法 (逐次過緩和法) で Poisson 方程式を解く。

    ■ 考え方
      5点差分の式
        (4u[i,j] - u[i+1,j] - u[i-1,j] - u[i,j+1] - u[i,j-1]) / h² = f[i,j]
      を u[i,j] について解くと
        u[i,j] = ( u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + h² f[i,j] ) / 4
      になる。これを「全点が変化しなくなるまで」繰り返すのが反復法。

      - Jacobi 法      : 前回の反復の値だけを使って更新
      - Gauss-Seidel 法: 更新済みの最新の値をすぐ使う (Jacobi より速い)
      - SOR 法         : Gauss-Seidel の更新量を ω 倍して「行き過ぎ」させる
                          u_new = (1-ω) u_old + ω * u_GS
                          1 < ω < 2 で加速。ω = 1 なら Gauss-Seidel と同じ。

    ■ 赤黒 (Red-Black) 順序付け
      Python の二重 for ループは非常に遅いので、格子点を
      チェス盤のように「赤 (i+j が偶数)」「黒 (i+j が奇数)」に塗り分ける。
      赤の点の隣は必ず黒なので、赤の点は全部同時に (= numpy でまとめて)
      更新でき、次に黒の点をまとめて更新できる。
      → Gauss-Seidel 的な性質を保ったままベクトル化できる!

    ■ 最適な緩和係数 ω
      正方形領域・等間隔格子の Poisson 方程式では、理論的な最適値が
        ω_opt = 2 / (1 + sin(π h))
      と分かっている。omega を指定しなければこれを使う。

    Parameters
    ----------
    n : int
        各方向の分割数
    omega : float or None
        緩和係数 (None なら最適値を自動設定)
    tol : float
        収束判定の閾値 (残差の最大値がこれ以下になったら終了)
    max_iter : int
        最大反復回数 (収束しないときの安全装置)

    Returns
    -------
    x, y : ndarray
        格子点座標
    u : ndarray, shape (n+1, n+1)
        数値解
    n_iter : int
        実際にかかった反復回数
    """
    x, y, h = make_grid(n)
    f = source_term(x, y)

    # 最適な緩和係数を設定
    if omega is None:
        omega = 2.0 / (1.0 + np.sin(np.pi * h))

    # --- 初期値: 内部は 0、境界には境界値を入れる ---
    u = np.zeros((n + 1, n + 1))
    u[0, :] = boundary_value(x[0, :], y[0, :])
    u[-1, :] = boundary_value(x[-1, :], y[-1, :])
    u[:, 0] = boundary_value(x[:, 0], y[:, 0])
    u[:, -1] = boundary_value(x[:, -1], y[:, -1])

    # --- 赤黒のマスク (内部点のみ) を作る ---
    # i, j は内部点の添字 (1 ～ n-1)
    i_idx, j_idx = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
    red = (i_idx + j_idx) % 2 == 0    # i+j 偶数 → 赤
    black = ~red                      # それ以外 → 黒

    h2 = h * h  # h² を毎回計算しないように先に求めておく

    for n_iter in range(1, max_iter + 1):
        # 赤 → 黒 の順に更新する
        for mask in (red, black):
            # 内部点 [1:-1, 1:-1] から見た上下左右の隣接点の和
            neighbors = (
                u[2:, 1:-1]      # u[i+1, j]
                + u[:-2, 1:-1]   # u[i-1, j]
                + u[1:-1, 2:]    # u[i, j+1]
                + u[1:-1, :-2]   # u[i, j-1]
            )
            # Gauss-Seidel の更新値
            u_gs = (neighbors + h2 * f[1:-1, 1:-1]) / 4.0

            # 内部点のビュー (u の一部を直接書き換えるための参照)
            interior = u[1:-1, 1:-1]
            # SOR: 古い値と GS の値を ω で重み付け (マスクの点だけ更新)
            interior[mask] = (1.0 - omega) * interior[mask] + omega * u_gs[mask]

        # --- 収束判定: 残差 r = f - (-Δu) の最大値を見る ---
        # 毎回計算するとやや重いので、10回に1回だけチェックする
        if n_iter % 10 == 0:
            lap = (
                4.0 * u[1:-1, 1:-1]
                - u[2:, 1:-1] - u[:-2, 1:-1]
                - u[1:-1, 2:] - u[1:-1, :-2]
            ) / h2
            residual = np.max(np.abs(f[1:-1, 1:-1] - lap))
            if residual < tol:
                break
    else:
        # for ループが break されずに終わった = 収束しなかった
        print(f"  [警告] SOR 法が {max_iter} 回で収束しませんでした")

    return x, y, u, n_iter


# ============================================================
# 誤差評価
# ============================================================
def compute_errors(x, y, u):
    """
    数値解 u と厳密解の誤差を計算する。

    - 最大値ノルム (L∞) : 全点の中で一番大きい誤差
    - L2 ノルム         : 誤差の二乗平均の平方根 (面積で重み付け)
                          ≈ sqrt( Σ e² h² )

    Returns
    -------
    err_max, err_l2 : float
    """
    err = u - exact_solution(x, y)
    h = x[1, 0] - x[0, 0]  # 格子幅 (x 方向の隣の点との差)
    err_max = np.max(np.abs(err))
    err_l2 = np.sqrt(np.sum(err**2) * h**2)
    return err_max, err_l2


def convergence_study(n_list):
    """
    格子を細かくしていったときに誤差がどう減るかを調べる (収束性の確認)。

    5点差分は2次精度なので、h を半分にすると誤差は約 1/4 になるはず。
    収束次数 p は
        p = log(e_粗 / e_細) / log(h_粗 / h_細)
    で見積もれる。p ≈ 2 になれば実装は正しいと言ってよい。
    """
    print("\n===== 収束性の確認 (直接法) =====")
    print(f"{'n':>6} {'h':>10} {'L∞ 誤差':>14} {'L2 誤差':>14} {'次数':>6}")

    prev_err = None
    prev_h = None
    for n in n_list:
        x, y, u = solve_direct(n)
        err_max, err_l2 = compute_errors(x, y, u)
        h = 1.0 / n

        # 1つ前の格子と比べて収束次数を計算
        if prev_err is not None:
            order = np.log(prev_err / err_max) / np.log(prev_h / h)
            order_str = f"{order:6.2f}"
        else:
            order_str = "   ---"  # 最初の格子は比較対象がない

        print(f"{n:6d} {h:10.5f} {err_max:14.4e} {err_l2:14.4e} {order_str}")
        prev_err, prev_h = err_max, h


# ============================================================
# 可視化
# ============================================================
def plot_results(x, y, u):
    """
    数値解・厳密解・誤差の3つを並べて描画する。
    """
    u_ex = exact_solution(x, y)
    err = np.abs(u - u_ex)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (1) 数値解
    c0 = axes[0].contourf(x, y, u, levels=50, cmap="viridis")
    fig.colorbar(c0, ax=axes[0])
    axes[0].set_title("Numerical solution")

    # (2) 厳密解
    c1 = axes[1].contourf(x, y, u_ex, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=axes[1])
    axes[1].set_title("Exact solution")

    # (3) 絶対誤差
    c2 = axes[2].contourf(x, y, err, levels=50, cmap="inferno")
    fig.colorbar(c2, ax=axes[2])
    axes[2].set_title("Absolute error")

    # 共通の軸設定
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")  # 正方形領域なので縦横比を揃える

    plt.tight_layout()
    plt.savefig("poisson_result.png", dpi=150)  # 画像として保存
    plt.show()


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    n = 64  # 各方向の分割数 (大きくすると精度↑・計算時間↑)

    # ---------- 直接法 ----------
    print(f"===== 直接法 (n = {n}) =====")
    t0 = time.perf_counter()
    x, y, u_direct = solve_direct(n)
    t1 = time.perf_counter()
    err_max, err_l2 = compute_errors(x, y, u_direct)
    print(f"  計算時間 : {t1 - t0:.4f} 秒")
    print(f"  L∞ 誤差  : {err_max:.4e}")
    print(f"  L2 誤差  : {err_l2:.4e}")

    # ---------- SOR 法 ----------
    print(f"\n===== SOR 法 (n = {n}) =====")
    t0 = time.perf_counter()
    _, _, u_sor, n_iter = solve_sor(n)
    t1 = time.perf_counter()
    err_max, err_l2 = compute_errors(x, y, u_sor)
    print(f"  計算時間 : {t1 - t0:.4f} 秒")
    print(f"  反復回数 : {n_iter}")
    print(f"  L∞ 誤差  : {err_max:.4e}")
    print(f"  L2 誤差  : {err_l2:.4e}")

    # 2つの解法の結果がほぼ一致しているか確認
    diff = np.max(np.abs(u_direct - u_sor))
    print(f"\n  直接法と SOR 法の差 (最大値): {diff:.4e}")

    # ---------- 収束性の確認 ----------
    convergence_study([8, 16, 32, 64, 128])

    # ---------- 可視化 ----------
    plot_results(x, y, u_direct)
