/*
 * poisson_jacobi.c
 * ============================================================
 * ポアソンの方程式 φ_xx + φ_yy = 0 を「差分法 + ヤコビ法」で解く C プログラム
 * (src/numerical.py の solve_poisson_jacobi の C 版)
 * ============================================================
 *
 * ■ 問題設定 (src/numerical.py, PoissonFCNN と同じ)
 * ------------------------------------------------------------
 *   領域は単位正方形 [0,1]x[0,1]。刻み幅 h = 1/n で (n+1)x(n+1) 個の格子点をとり、
 *   phi[i,j] に点 (x, y) = (i/n, j/n) での φ の値を入れる。
 *
 *   境界条件:
 *     - 外周で φ = 0
 *     - 中心の周りの4点 (a,a), (a,b), (b,a), (b,b) で φ = 1
 *         a = ceil(n/2)/n,  b = floor(n/2)/n
 *       (n が偶数なら a = b となり、中心の1点だけになる)
 *
 * ■ 離散化: 5点差分
 * ------------------------------------------------------------
 *   φ_xx ≈ (φ[i+1,j] - 2φ[i,j] + φ[i-1,j]) / h²
 *   φ_yy ≈ (φ[i,j+1] - 2φ[i,j] + φ[i,j-1]) / h²
 *   これを φ_xx + φ_yy = 0 に代入して φ[i,j] について解くと
 *       φ[i,j] = (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1]) / 4
 *   つまり「各点の値は上下左右4点の平均」という連立一次方程式になる。
 *   (右辺が 0 なので h² が消え、格子幅が式に出てこないのがポイント)
 *
 * ■ ヤコビ法 (Jacobi method)
 * ------------------------------------------------------------
 *   右辺の4点に「前の反復の値 (old)」だけを使って、全点を一斉に置き換える:
 *       new[i,j] = (old[i+1,j] + old[i-1,j] + old[i,j+1] + old[i,j-1]) / 4
 *   これを、1回の更新での変化の最大値が tol を下回るまで繰り返す。
 *   (収束判定も src/numerical.py と同じにしてある)
 *
 *   - 長所: 各点の更新が互いに独立なので、並列化しやすい
 *   - 短所: 収束が遅い。反復回数はおおよそ O(n²) で増える
 *
 * ■ 使っているライブラリ
 * ------------------------------------------------------------
 *   - C 標準ライブラリ : stdio.h / stdlib.h / string.h / math.h
 *   - OpenMP          : ループの並列化と時間計測 (omp_get_wtime)
 *     ※ OpenMP なしでもコンパイルできる (#ifdef _OPENMP で切り替え)。
 *       Makefile の既定は OpenMP なし (理由は Makefile のコメント参照)。
 *
 * ■ ビルドと実行 (csrc/ で)
 * ------------------------------------------------------------
 *   make                 # ビルド (逐次版)。実行ファイルは build/ に出力される
 *   make OMP=1 -B        # ビルド (OpenMP 並列版)
 *   make run             # n = 101 で解き、out/numerical_c.dat に出力
 *
 *   直接実行する場合 (プロジェクト直下で):
 *   ./build/poisson_jacobi [n] [出力ファイル] [tol] [max_iter]
 *     n        : 分割数 (既定 101 = config/poisson_eq.yaml の grid_density)
 *     出力     : 既定 out/numerical_c.dat (実行した場所からの相対パス)
 *     tol      : 既定 1e-10
 *     max_iter : 既定 100000
 *
 * ■ 出力形式 (out/numerical.dat と同じ)
 * ------------------------------------------------------------
 *   1列目 x, 2列目 y, 3列目 φ を空白区切り・小数点以下6桁で書き出す。
 *   並び順は x が外側、y が内側 (sampling.sample_grid と同じ)。
 */

#include <stdio.h>   /* printf, fprintf, fopen など入出力 */
#include <stdlib.h>  /* malloc, calloc, free, atoi, atof, EXIT_* */
#include <string.h>  /* memcpy */
#include <math.h>    /* fabs */

#ifdef _OPENMP
#include <omp.h>     /* omp_get_wtime, omp_get_max_threads */
#else
/* OpenMP が無いとき用の代替: clock() で時間を測る */
#include <time.h>
static double omp_get_wtime(void) { return (double)clock() / CLOCKS_PER_SEC; }
static int omp_get_max_threads(void) { return 1; }
#endif

/* 2次元の添字 (i, j) を 1次元配列の添字に変換するマクロ。
 *   - 1行の長さは (n+1)
 *   - i が x 方向、j が y 方向 (numpy の phi[i, j] と同じ並び)
 *   - C では大きさが実行時に決まる2次元配列は扱いにくいので、
 *     1次元配列 + 添字計算で2次元配列を表現するのが定番 */
#define IDX(i, j, n) ((size_t)(i) * (size_t)((n) + 1) + (size_t)(j))


/* ============================================================
 * 境界条件を設定した初期状態を作る
 * ============================================================
 *   phi   : 外周は 0、中心の4点は 1、それ以外は 0 (初期値)
 *   fixed : 境界条件で値が決まっている点は 1、それ以外は 0
 *
 * src/numerical.py の boundary_condition と同じ。
 */
static void boundary_condition(int n, double *phi, unsigned char *fixed)
{
    /* 外周で φ = 0
     * (calloc で 0 初期化済みなので、phi はそのままでよい。fixed だけ立てる) */
    for (int k = 0; k <= n; k++) {
        fixed[IDX(0, k, n)] = 1;   /* x = 0 */
        fixed[IDX(n, k, n)] = 1;   /* x = 1 */
        fixed[IDX(k, 0, n)] = 1;   /* y = 0 */
        fixed[IDX(k, n, n)] = 1;   /* y = 1 */
    }

    /* 中心の周りの4点で φ = 1
     *   ceil(n/2)  = (n + 1) / 2   (整数の割り算は切り捨てなので +1 して切り上げにする)
     *   floor(n/2) = n / 2 */
    const int centers[2] = {(n + 1) / 2, n / 2};
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            phi[IDX(centers[a], centers[b], n)] = 1.0;
            fixed[IDX(centers[a], centers[b], n)] = 1;
        }
    }
}


/* ============================================================
 * ヤコビ法
 * ============================================================
 * 各点を上下左右4点の平均で置き換える操作を、
 * 値がほとんど変わらなくなるまで繰り返す。
 *
 * 引数:
 *   n        : 分割数
 *   phi      : [入出力] 初期状態 (境界条件を設定済み)。終了時には解が入る
 *   fixed    : [入力]   境界条件で値が決まっている点のフラグ
 *   tol      : 1回の更新での変化の最大値がこれを下回ったら終了。
 *              ヤコビ法は収束が遅く、変化が小さくても正しい解から
 *              まだ離れていることがあるので小さめにとる
 *   max_iter : 最大反復回数
 *   change   : [出力] 最後の更新での変化の最大値
 *
 * 戻り値: 実際に繰り返した回数
 */
static int solve_poisson_jacobi(int n, double *phi, const unsigned char *fixed,
                                double tol, int max_iter, double *change)
{
    const size_t size = (size_t)(n + 1) * (size_t)(n + 1);

    /* ヤコビ法では「古い値」と「新しい値」の2つの配列が必要。
     * 新しい方を確保し、境界の値もまとめてコピーしておく
     * (境界の点は更新しないので、両方の配列に同じ値が入っている必要がある) */
    double *work = (double *)malloc(size * sizeof(double));
    if (work == NULL) {
        fprintf(stderr, "メモリ確保に失敗しました\n");
        exit(EXIT_FAILURE);
    }
    memcpy(work, phi, size * sizeof(double));  /* 標準ライブラリで一括コピー */

    /* old / new はポインタで持ち、反復ごとに「入れ替える」だけにする。
     * 毎回配列全体をコピーするより速い (ダブルバッファリング) */
    double *old_phi = phi;
    double *new_phi = work;

    int num_iter = 0;
    *change = HUGE_VAL;  /* math.h の「無限大」。Python の math.inf にあたる */

    while (num_iter < max_iter) {
        double max_change = 0.0;

        /* ----------------------------------------------------------
         * 内側の各点 (i, j = 1 ～ n-1) を一斉に更新する
         *
         * #pragma omp parallel for :
         *     外側の i ループを複数スレッドで分担する (OMP=1 でビルドしたとき)
         * reduction(max:max_change) :
         *     各スレッドの max_change の最大値を最後にまとめてくれる
         * ---------------------------------------------------------- */
        #pragma omp parallel for reduction(max:max_change)
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                const size_t k = IDX(i, j, n);

                /* 境界条件で値が決まっている点 (中心の4点) は更新しない。
                 * new_phi には最初にコピーした値がずっと残っている */
                if (fixed[k]) {
                    continue;
                }

                /* 上下左右4点の平均 (すべて古い値を使う = ヤコビ法) */
                const double right = old_phi[IDX(i + 1, j, n)];  /* φ[i+1, j] */
                const double left  = old_phi[IDX(i - 1, j, n)];  /* φ[i-1, j] */
                const double up    = old_phi[IDX(i, j + 1, n)];  /* φ[i, j+1] */
                const double down  = old_phi[IDX(i, j - 1, n)];  /* φ[i, j-1] */
                const double value = (right + left + up + down) / 4.0;

                new_phi[k] = value;

                /* 変化の最大値を記録 (収束判定に使う) */
                const double diff = fabs(value - old_phi[k]);
                if (diff > max_change) {
                    max_change = diff;
                }
            }
        }

        /* 古い配列と新しい配列を入れ替える (ポインタの交換だけ) */
        double *tmp = old_phi;
        old_phi = new_phi;
        new_phi = tmp;

        num_iter++;
        *change = max_change;
        if (max_change < tol) {
            break;
        }
    }

    /* 最新の解は old_phi 側に入っている。
     * それが呼び出し元の phi と別の配列なら、phi にコピーして返す */
    if (old_phi != phi) {
        memcpy(phi, old_phi, size * sizeof(double));
    }
    free(work);

    return num_iter;
}


/* ============================================================
 * 結果をファイルに書き出す
 * ============================================================
 * 形式は script/poisson_numerical.py の
 *     np.savetxt("out/numerical.dat", XY, "%.6f")
 * と同じ: 1行に「x y φ」を空白区切り・小数点以下6桁で書く。
 * 並び順は x が外側、y が内側 (src/numerical.py の to_table と同じ)。
 */
static int write_table(const char *filename, int n, const double *phi)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror(filename);  /* 失敗理由 (フォルダが無い など) を表示 */
        return 0;
    }

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            /* 座標は i/n で計算する (numpy の np.arange(n+1) / n と同じ) */
            fprintf(fp, "%.6f %.6f %.6f\n",
                    (double)i / n, (double)j / n, phi[IDX(i, j, n)]);
        }
    }

    fclose(fp);
    return 1;
}


/* ============================================================
 * メイン関数
 * ============================================================ */
int main(int argc, char *argv[])
{
    /* ---------- コマンドライン引数 (省略時は既定値) ---------- */
    int n              = (argc > 1) ? atoi(argv[1]) : 101;   /* grid_density と同じ */
    const char *output = (argc > 2) ? argv[2] : "out/numerical_c.dat";
    double tol         = (argc > 3) ? atof(argv[3]) : 1e-10; /* src/numerical.py と同じ */
    int max_iter       = (argc > 4) ? atoi(argv[4]) : 100000;

    if (n < 2 || tol <= 0.0 || max_iter < 1) {
        fprintf(stderr, "使い方: %s [n (2以上)] [出力ファイル] [tol] [max_iter]\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* ---------- 配列の確保 ----------
     * calloc は確保と同時に 0 で初期化してくれるので、
     * 「外周 0、内部の初期値 0」がそのまま入る */
    const size_t size = (size_t)(n + 1) * (size_t)(n + 1);
    double *phi          = (double *)calloc(size, sizeof(double));
    unsigned char *fixed = (unsigned char *)calloc(size, sizeof(unsigned char));
    if (phi == NULL || fixed == NULL) {
        fprintf(stderr, "メモリ確保に失敗しました\n");
        return EXIT_FAILURE;
    }

    boundary_condition(n, phi, fixed);

    /* ---------- ヤコビ法で解く ---------- */
    double change;
    const double t0 = omp_get_wtime();
    const int num_iter = solve_poisson_jacobi(n, phi, fixed, tol, max_iter, &change);
    const double t1 = omp_get_wtime();

    /* src/numerical.py と同じ形式で表示 + 時間とスレッド数 */
    printf("Jacobi: n=%d iter=%d change=%.3e time=%.3fs threads=%d\n",
           n, num_iter, change, t1 - t0, omp_get_max_threads());
    if (change >= tol) {
        fprintf(stderr, "[警告] %d 回で収束しませんでした (change=%.3e >= tol=%.1e)\n",
                max_iter, change, tol);
    }

    /* ---------- 書き出し ---------- */
    int ok = write_table(output, n, phi);
    if (ok) {
        printf("complete: %s\n", output);
    }

    free(phi);
    free(fixed);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
