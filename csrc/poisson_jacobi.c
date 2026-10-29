/*
 * ポアソンの方程式 φ_xx + φ_yy = 0 をヤコビ法で解く (src/numerical.py の solve_jacobi の C 版. 速度比較用)
 *
 * 問題設定・収束判定・出力形式は src/numerical.py と同じ
 *   領域 [0,1]x[0,1], 間隔 h=1/n. 外周で φ=0, 中心の4点で φ=1
 *   出力は 1列目 x, 2列目 y, 3列目 φ. x が同じ点ごとに空行で区切る (out/numerical.dat と同じ)
 *
 * 使い方 (プロジェクト直下で): ./build/jacobi [n] [出力ファイル] [tol] [max_iter]
 *   既定値は n=101, out/numerical_c.dat, tol=1e-10, max_iter=100000
 * ビルドは csrc/Makefile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
static double omp_get_wtime(void) { return (double)clock() / CLOCKS_PER_SEC; }
static int omp_get_max_threads(void) { return 1; }
#endif

/* phi[i][j] を 1次元配列で持つときの添字. i が x 方向, j が y 方向 */
#define IDX(i, j, n) ((size_t)(i) * (size_t)((n) + 1) + (size_t)(j))

/* 外周と中心の4点を fixed=1 にし, 中心の4点は phi=1 にする (phi, fixed は 0 で初期化済み) */
static void boundary_condition(int n, double *phi, unsigned char *fixed)
{
    for (int k = 0; k <= n; k++) {
        fixed[IDX(0, k, n)] = 1;
        fixed[IDX(n, k, n)] = 1;
        fixed[IDX(k, 0, n)] = 1;
        fixed[IDX(k, n, n)] = 1;
    }

    const int centers[2] = {(n + 1) / 2, n / 2};   /* ceil(n/2), floor(n/2) */
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            phi[IDX(centers[a], centers[b], n)] = 1.0;
            fixed[IDX(centers[a], centers[b], n)] = 1;
        }
    }
}

/*
 * ヤコビ法. 内側の各点を上下左右4点の (前の反復の) 平均で置き換える操作を,
 * 1回の変化の最大値が tol を下回るまで繰り返す. 解は phi に入る. 戻り値は反復回数
 */
static int solve_jacobi(int n, double *phi, const unsigned char *fixed,
                        double tol, int max_iter, double *change)
{
    const size_t size = (size_t)(n + 1) * (size_t)(n + 1);
    double *work = malloc(size * sizeof(double));
    if (work == NULL) {
        fprintf(stderr, "メモリ確保に失敗しました\n");
        exit(EXIT_FAILURE);
    }
    memcpy(work, phi, size * sizeof(double));   /* 境界の値を両方の配列に入れておく */

    /* 古い値と新しい値の配列を, 反復ごとにポインタの入れ替えで交互に使う */
    double *old_phi = phi;
    double *new_phi = work;

    int num_iter = 0;
    *change = HUGE_VAL;
    while (num_iter < max_iter) {
        double max_change = 0.0;

        #pragma omp parallel for reduction(max:max_change)
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                const size_t k = IDX(i, j, n);
                if (fixed[k]) {
                    continue;
                }
                const double value = (old_phi[IDX(i + 1, j, n)] + old_phi[IDX(i - 1, j, n)]
                                    + old_phi[IDX(i, j + 1, n)] + old_phi[IDX(i, j - 1, n)]) / 4.0;
                new_phi[k] = value;

                const double diff = fabs(value - old_phi[k]);
                if (diff > max_change) {
                    max_change = diff;
                }
            }
        }

        double *tmp = old_phi;
        old_phi = new_phi;
        new_phi = tmp;

        num_iter++;
        *change = max_change;
        if (max_change < tol) {
            break;
        }
    }

    if (old_phi != phi) {   /* 最新の解が work 側にあるときは phi に移す */
        memcpy(phi, old_phi, size * sizeof(double));
    }
    free(work);
    return num_iter;
}

/* x y φ を小数点以下6桁で書く. x が同じ点ごとに空行で区切る */
static int write_table(const char *filename, int n, const double *phi)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror(filename);
        return 0;
    }
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            fprintf(fp, "%.6f %.6f %.6f\n", (double)i / n, (double)j / n, phi[IDX(i, j, n)]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 1;
}

int main(int argc, char *argv[])
{
    int n              = (argc > 1) ? atoi(argv[1]) : 101;
    const char *output = (argc > 2) ? argv[2] : "out/numerical_c.dat";
    double tol         = (argc > 3) ? atof(argv[3]) : 1e-10;
    int max_iter       = (argc > 4) ? atoi(argv[4]) : 100000;
    if (n < 2 || tol <= 0.0 || max_iter < 1) {
        fprintf(stderr, "使い方: %s [n (2以上)] [出力ファイル] [tol] [max_iter]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t size = (size_t)(n + 1) * (size_t)(n + 1);
    double *phi          = calloc(size, sizeof(double));
    unsigned char *fixed = calloc(size, sizeof(unsigned char));
    if (phi == NULL || fixed == NULL) {
        fprintf(stderr, "メモリ確保に失敗しました\n");
        return EXIT_FAILURE;
    }
    boundary_condition(n, phi, fixed);

    double change;
    const double t0 = omp_get_wtime();
    const int num_iter = solve_jacobi(n, phi, fixed, tol, max_iter, &change);
    const double t1 = omp_get_wtime();
    printf("Jacobi: n=%d iter=%d change=%.3e time=%.3fs threads=%d\n",
           n, num_iter, change, t1 - t0, omp_get_max_threads());
    if (change >= tol) {
        fprintf(stderr, "[警告] %d 回で収束しませんでした (change=%.3e >= tol=%.1e)\n", max_iter, change, tol);
    }

    const int ok = write_table(output, n, phi);
    if (ok) {
        printf("complete: %s\n", output);
    }
    free(phi);
    free(fixed);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
