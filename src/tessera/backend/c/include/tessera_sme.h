/*
 * SME fp32 GEMM for one tile: C[0:m, 0:n] (+)= A[0:m, 0:k] B[0:k, 0:n].
 *
 * The M4's streaming vectors hold 16 floats, and its ZA array holds four
 * 16 x 16 fp32 tiles. A 32 x 32 block of C lives in the four tiles; each
 * step of k loads 32 values of a column of A and 32 values of a row of B
 * and issues four FMOPA outer products, i.e. 1024 multiply-adds for four
 * loads (two with SME2's two-register loads).
 *
 * Operands are packed outside streaming mode with NEON (tessera_pack.h),
 * then one streaming call computes every 32 x 32 block of a packed MC x NC block of the tile
 * for KC values of k: entering and leaving streaming mode (smstart and
 * smstop, which also save and restore vector state) happens once per
 * macro-block of at least MC * NC * KC / 1024 outer products, not per
 * block of C. The packed A panel is A transposed
 * (k-major, 32 rows per panel), the B panel k-major with 32 columns; both
 * are zero-padded, so only the stores of edge blocks need predicates.
 *
 * Only code inside the streaming function uses SVE instructions; the M4
 * runs SVE only in streaming mode.
 *
 *   tsr_sme_sgemm, tsr_sme_sgemm_bt, tsr_sme_conv2d   as their NEON namesakes
 */
#ifndef TESSERA_SME_H
#define TESSERA_SME_H

#if !defined(__ARM_FEATURE_SME)
#error "tessera_sme.h needs a target with SME (clang -mcpu=apple-m4)"
#endif

#include <arm_neon.h>
#include <arm_sme.h>

#include "tessera_pack.h"
#include "tessera_runtime.h"

#define TSR_SME_BLOCK 32
/* Blocking of the tile (BLIS-style): a TSR_SME_KC x TSR_SME_NC panel of B and
 * a TSR_SME_MC x TSR_SME_KC panel of A are packed and kept in L2. */
#ifndef TSR_SME_KC
#define TSR_SME_KC 1024
#endif
#ifndef TSR_SME_MC
#define TSR_SME_MC 256
#endif
#ifndef TSR_SME_NC
#define TSR_SME_NC 512
#endif

/* Every 32 x 32 block of C[0:mc, 0:n] (+)= packed A . packed B over kc steps,
 * in one streaming-mode call. */
__arm_locally_streaming __arm_new("za") static void tsr_sme_blocks(
    int64_t mc, int64_t n, int64_t kc, const float* ap, const float* bp, float* C, int64_t ldc,
    bool load) {
  svbool_t all = svptrue_b32();
#if defined(__ARM_FEATURE_SME2)
  svcount_t all2 = svptrue_c32();
#endif
  for (int64_t j0 = 0; j0 < n; j0 += TSR_SME_BLOCK) {
    int64_t cols = n - j0 < TSR_SME_BLOCK ? n - j0 : TSR_SME_BLOCK;
    svbool_t left = svwhilelt_b32_s64(0, cols), right = svwhilelt_b32_s64(16, cols);
    for (int64_t i0 = 0; i0 < mc; i0 += TSR_SME_BLOCK) {
      int64_t rows = mc - i0 < TSR_SME_BLOCK ? mc - i0 : TSR_SME_BLOCK;
      float* c = C + i0 * ldc + j0;
      svzero_za();
      if (load) {
        for (int64_t r = 0; r < rows; ++r) {
          uint32_t slice = (uint32_t)(r & 15);
          if (r < 16) {
            svld1_hor_za32(0, slice, left, c + r * ldc);
            svld1_hor_za32(1, slice, right, c + r * ldc + 16);
          } else {
            svld1_hor_za32(2, slice, left, c + r * ldc);
            svld1_hor_za32(3, slice, right, c + r * ldc + 16);
          }
        }
      }
      const float* a = ap + (i0 / TSR_SME_BLOCK) * kc * TSR_SME_BLOCK;
      const float* b = bp + (j0 / TSR_SME_BLOCK) * kc * TSR_SME_BLOCK;
      for (int64_t q = 0; q < kc; ++q) {
#if defined(__ARM_FEATURE_SME2)
        svfloat32x2_t av = svld1_x2(all2, a), bv = svld1_x2(all2, b);
        svfloat32_t a0 = svget2(av, 0), a1 = svget2(av, 1);
        svfloat32_t b0 = svget2(bv, 0), b1 = svget2(bv, 1);
#else
        svfloat32_t a0 = svld1_f32(all, a), a1 = svld1_f32(all, a + 16);
        svfloat32_t b0 = svld1_f32(all, b), b1 = svld1_f32(all, b + 16);
#endif
        svmopa_za32_f32_m(0, all, all, a0, b0);
        svmopa_za32_f32_m(1, all, all, a0, b1);
        svmopa_za32_f32_m(2, all, all, a1, b0);
        svmopa_za32_f32_m(3, all, all, a1, b1);
        a += TSR_SME_BLOCK;
        b += TSR_SME_BLOCK;
      }
      for (int64_t r = 0; r < rows; ++r) {
        uint32_t slice = (uint32_t)(r & 15);
        if (r < 16) {
          svst1_hor_za32(0, slice, left, c + r * ldc);
          svst1_hor_za32(1, slice, right, c + r * ldc + 16);
        } else {
          svst1_hor_za32(2, slice, left, c + r * ldc);
          svst1_hor_za32(3, slice, right, c + r * ldc + 16);
        }
      }
    }
  }
}

static const tsr_blocking tsr_sme_blocking = {
    TSR_SME_BLOCK, TSR_SME_BLOCK, TSR_SME_MC, TSR_SME_NC, TSR_SME_KC, tsr_sme_blocks,
};

/* C = A B (accumulate == 0) or C += A B, for one m x n tile. */
static void tsr_sme_sgemm(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                          const float* B, int64_t ldb, float* C, int64_t ldc,
                          int64_t accumulate) {
  tsr_lhs lhs = {A, lda, NULL, 0};
  tsr_blocked_gemm(&tsr_sme_blocking, m, n, k, &lhs, B, ldb, false, C, ldc, accumulate != 0);
}

/* The same with B read transposed: B[p, j] at B[j * ldb + p]. */
static void tsr_sme_sgemm_bt(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                             const float* B, int64_t ldb, float* C, int64_t ldc,
                             int64_t accumulate) {
  tsr_lhs lhs = {A, lda, NULL, 0};
  tsr_blocked_gemm(&tsr_sme_blocking, m, n, k, &lhs, B, ldb, true, C, ldc, accumulate != 0);
}

/* Implicit-GEMM convolution, as tsr_neon_conv2d. */
static void tsr_sme_conv2d(int64_t m, int64_t n, int64_t p0, const float* x, const float* w,
                           int64_t ldw, float* y, int64_t ldy, int64_t h, int64_t wd, int64_t c,
                           int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph,
                           int64_t pw, int64_t dh, int64_t dw, int64_t ow, int64_t accumulate) {
  tsr_conv_shape shape = {h, wd, c, kh, kw, sh, sw, ph, pw, dh, dw, ow};
  tsr_lhs lhs = {x, 0, &shape, p0};
  tsr_blocked_gemm(&tsr_sme_blocking, m, n, kh * kw * c, &lhs, w, ldw, false, y, ldy,
                   accumulate != 0);
}

#endif /* TESSERA_SME_H */
