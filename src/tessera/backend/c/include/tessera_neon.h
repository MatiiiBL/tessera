/*
 * NEON fp32 GEMM and convolution on one tile of the output.
 *
 * The microkernel keeps an 8 x 12 block of C in 24 of the 32 vector
 * registers. Per step of the reduction it loads 8 values of packed A (two
 * registers) and 12 of packed B (three), and issues 24 FMAs, each
 * multiplying a row of B by one lane of A. Blocking and packing are
 * tsr_blocked_gemm's (tessera_pack.h): a 96 x 256 block of A stays in L2 and
 * the 256 x 12 panel of B in L1 while the microkernel walks over them.
 *
 *   tsr_neon_sgemm     C (+)= A B, row-major operands with leading dimensions
 *   tsr_neon_sgemm_bt  the same with B read transposed (B[p, j] at B[j*ldb + p])
 *   tsr_neon_conv2d    implicit GEMM: output pixels x channels of an NHWC
 *                      convolution with HWIO weights
 */
#ifndef TESSERA_NEON_H
#define TESSERA_NEON_H

#include <arm_neon.h>

#include "tessera_pack.h"
#include "tessera_runtime.h"

#define TSR_NEON_MR 8
#define TSR_NEON_NR 12
#ifndef TSR_NEON_MC
#define TSR_NEON_MC 96
#endif
#ifndef TSR_NEON_NC
#define TSR_NEON_NC 2040 /* a multiple of NR */
#endif
#ifndef TSR_NEON_KC
#define TSR_NEON_KC 256
#endif

#define TSR_NEON_ROW(r, lanes, lane)                    \
  c##r##0 = vfmaq_laneq_f32(c##r##0, b0, lanes, lane); \
  c##r##1 = vfmaq_laneq_f32(c##r##1, b1, lanes, lane); \
  c##r##2 = vfmaq_laneq_f32(c##r##2, b2, lanes, lane);

#define TSR_NEON_LOAD_ROW(r)                                          \
  c##r##0 = load ? vld1q_f32(c + (r) * ldc) : vdupq_n_f32(0.0f);     \
  c##r##1 = load ? vld1q_f32(c + (r) * ldc + 4) : vdupq_n_f32(0.0f); \
  c##r##2 = load ? vld1q_f32(c + (r) * ldc + 8) : vdupq_n_f32(0.0f);

#define TSR_NEON_STORE_ROW(r)            \
  vst1q_f32(c + (r) * ldc, c##r##0);     \
  vst1q_f32(c + (r) * ldc + 4, c##r##1); \
  vst1q_f32(c + (r) * ldc + 8, c##r##2);

/* c[0:8, 0:12] (+)= a_panel . b_panel over kc steps. */
static inline void tsr_neon_micro_8x12(int64_t kc, const float* restrict a,
                                       const float* restrict b, float* restrict c, int64_t ldc,
                                       bool load) {
  float32x4_t c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;
  float32x4_t c40, c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72;
  TSR_NEON_LOAD_ROW(0) TSR_NEON_LOAD_ROW(1) TSR_NEON_LOAD_ROW(2) TSR_NEON_LOAD_ROW(3)
  TSR_NEON_LOAD_ROW(4) TSR_NEON_LOAD_ROW(5) TSR_NEON_LOAD_ROW(6) TSR_NEON_LOAD_ROW(7)
  for (int64_t q = 0; q < kc; ++q) {
    float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
    float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
    TSR_NEON_ROW(0, a0, 0) TSR_NEON_ROW(1, a0, 1) TSR_NEON_ROW(2, a0, 2) TSR_NEON_ROW(3, a0, 3)
    TSR_NEON_ROW(4, a1, 0) TSR_NEON_ROW(5, a1, 1) TSR_NEON_ROW(6, a1, 2) TSR_NEON_ROW(7, a1, 3)
    a += TSR_NEON_MR;
    b += TSR_NEON_NR;
  }
  TSR_NEON_STORE_ROW(0) TSR_NEON_STORE_ROW(1) TSR_NEON_STORE_ROW(2) TSR_NEON_STORE_ROW(3)
  TSR_NEON_STORE_ROW(4) TSR_NEON_STORE_ROW(5) TSR_NEON_STORE_ROW(6) TSR_NEON_STORE_ROW(7)
}

/* Every 8 x 12 block of C[0:mc, 0:nc]; partial blocks go through a scratch tile. */
static void tsr_neon_macro(int64_t mc, int64_t nc, int64_t kc, const float* ap, const float* bp,
                           float* C, int64_t ldc, bool load) {
  for (int64_t j = 0; j < nc; j += TSR_NEON_NR) {
    const float* b = bp + (j / TSR_NEON_NR) * kc * TSR_NEON_NR;
    int64_t cols = nc - j < TSR_NEON_NR ? nc - j : TSR_NEON_NR;
    for (int64_t i = 0; i < mc; i += TSR_NEON_MR) {
      const float* a = ap + (i / TSR_NEON_MR) * kc * TSR_NEON_MR;
      float* c = C + i * ldc + j;
      int64_t rows = mc - i < TSR_NEON_MR ? mc - i : TSR_NEON_MR;
      if (rows == TSR_NEON_MR && cols == TSR_NEON_NR) {
        tsr_neon_micro_8x12(kc, a, b, c, ldc, load);
        continue;
      }
      float tile[TSR_NEON_MR * TSR_NEON_NR] __attribute__((aligned(64)));
      if (load) {
        for (int64_t r = 0; r < rows; ++r)
          for (int64_t t = 0; t < cols; ++t) tile[r * TSR_NEON_NR + t] = c[r * ldc + t];
      }
      tsr_neon_micro_8x12(kc, a, b, tile, TSR_NEON_NR, load);
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t t = 0; t < cols; ++t) c[r * ldc + t] = tile[r * TSR_NEON_NR + t];
    }
  }
}

static const tsr_blocking tsr_neon_blocking = {
    TSR_NEON_MR, TSR_NEON_NR, TSR_NEON_MC, TSR_NEON_NC, TSR_NEON_KC, tsr_neon_macro,
};

/* C = A B (accumulate == 0) or C += A B, for one m x n tile. */
static void tsr_neon_sgemm(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                           const float* B, int64_t ldb, float* C, int64_t ldc,
                           int64_t accumulate) {
  tsr_lhs lhs = {A, lda, NULL, 0};
  tsr_blocked_gemm(&tsr_neon_blocking, m, n, k, &lhs, B, ldb, false, C, ldc, accumulate != 0);
}

/* The same with B read transposed: B[p, j] at B[j * ldb + p]. */
static void tsr_neon_sgemm_bt(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                              const float* B, int64_t ldb, float* C, int64_t ldc,
                              int64_t accumulate) {
  tsr_lhs lhs = {A, lda, NULL, 0};
  tsr_blocked_gemm(&tsr_neon_blocking, m, n, k, &lhs, B, ldb, true, C, ldc, accumulate != 0);
}

/*
 * Output pixels [p0, p0 + m) x channels [0, n) of a convolution of the NHWC
 * image x (h x wd x c) with HWIO weights w (kh x kw x c x n, row stride ldw),
 * into y (row stride ldy; y points at pixel p0, channel 0).
 */
static void tsr_neon_conv2d(int64_t m, int64_t n, int64_t p0, const float* x, const float* w,
                            int64_t ldw, float* y, int64_t ldy, int64_t h, int64_t wd,
                            int64_t c, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                            int64_t ph, int64_t pw, int64_t dh, int64_t dw, int64_t ow,
                            int64_t accumulate) {
  tsr_conv_shape shape = {h, wd, c, kh, kw, sh, sw, ph, pw, dh, dw, ow};
  tsr_lhs lhs = {x, 0, &shape, p0};
  tsr_blocked_gemm(&tsr_neon_blocking, m, n, kh * kw * c, &lhs, w, ldw, false, y, ldy,
                   accumulate != 0);
}

#endif /* TESSERA_NEON_H */
