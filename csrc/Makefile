# ------------------------------------------------------------
# poisson_jacobi のビルド用 Makefile (csrc/ で make する)
#
#   make          : ビルド (逐次版)
#   make OMP=1 -B : OpenMP を有効にしてビルド (並列版)
#   make run      : n = 101 で解き、../out/numerical_c.dat に出力
#   make clean    : 実行ファイルを削除
#
# ※ 既定は逐次版。手元の環境 (MinGW gcc 15, 16スレッド) で測ると、
#   n = 256 でも逐次版の方が速かった (逐次 17.6 秒 / OpenMP 16スレッド 24.6 秒)。
#   ヤコビ法は1反復の計算量が小さいのに、反復ごとにスレッド間の
#   同期 (バリア) が必要になるため、格子がかなり大きくないと並列化が得にならない。
# ------------------------------------------------------------

CC     = gcc
# -O2           : 最適化
# -Wall -Wextra : 警告をたくさん出す (バグの早期発見)
# -std=c11      : C11 規格でコンパイル
CFLAGS = -O2 -Wall -Wextra -std=c11
# -lm : 数学ライブラリ (fabs など) をリンク
LDLIBS = -lm

# OpenMP を有効にする (OMP=1 を指定したときだけ)
ifdef OMP
CFLAGS += -fopenmp
else
# 逐次版では #pragma omp が「知らない pragma」として警告されるので黙らせる
CFLAGS += -Wno-unknown-pragmas
endif

# Windows では実行ファイルに .exe が付く
ifeq ($(OS),Windows_NT)
EXE = .exe
endif

TARGET = poisson_jacobi$(EXE)

$(TARGET): poisson_jacobi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

run: $(TARGET)
	./$(TARGET) 101 ../out/numerical_c.dat

clean:
	rm -f $(TARGET)

.PHONY: run clean
