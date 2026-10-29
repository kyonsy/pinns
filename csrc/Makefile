# ヤコビ法の C 版をビルドする (csrc/ で make する). 実行ファイルは ../build/ に出力
#   make          : ビルド (逐次版)
#   make OMP=1 -B : OpenMP を有効にしてビルド (並列版)
#   make run      : n = 101 で解き, ../out/numerical_c.dat に出力
#   make clean    : 実行ファイルを削除
#
# 既定を逐次版にしているのは, 手元 (MinGW gcc 15, 16スレッド) では n = 256 でも逐次版の方が速かったため
# (逐次 17.6 秒 / OpenMP 24.6 秒). ヤコビ法は1反復の計算が軽く, スレッドの同期の手間の方が大きい

CC     = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDLIBS = -lm

ifdef OMP
CFLAGS += -fopenmp
else
CFLAGS += -Wno-unknown-pragmas
endif

ifeq ($(OS),Windows_NT)
EXE = .exe
endif

BUILD_DIR = ../build
TARGET    = $(BUILD_DIR)/jacobi$(EXE)

$(TARGET): poisson_jacobi.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	$(TARGET) 101 ../out/numerical_c.dat

clean:
	rm -f $(TARGET)

.PHONY: run clean
