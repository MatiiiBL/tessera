/*
 * SME fp32 GEMM for one tile: C[0:m, 0:n] (+)= A[0:m, 0:k] B[0:k, 0:n].
 *
 * The M4's streaming vectors hold 16 floats, and its ZA array holds four
 * 16 x 16 fp32 tiles. A 32 x 32 block of C lives in the four tiles; each
 * step of k loads 32 values of a column of A and 32 values of a row of B
 * and issues four FMOPA outer products, i.e. 1024 multiply-adds for four
 * loads (two with SME2's two-register loads).
 *
 * Operands are packed outside streaming mode with NEON, then one streaming
 * call computes every 32 x 32 block of a packed MC x NC block of the tile
 * for KC values of k: entering and leaving streaming mode (smstart and
 * smstop, which also save and restore vector state) happens once per
 * macro-block of at least MC * NC * KC / 1024 outer products, not per
 * block of C. The packed A panel is A transposed
 * (k-major, 32 rows per panel), the B panel k-major with 32 columns; both
 * are zero-padded, so only the stores of edge blocks need predicates.
 *
 * Only code inside the streaming function uses SVE instructions; the M4
 * runs SVE only in streaming mode.
 */
#ifndef TESSERA_SME_H
#define TESSERA_SME_H

#if !defined(__ARM_FEATURE_SME)
#error "tessera_sme.h needs a target with SME (clang -mcpu=apple-m4)"
#endif

#include <arm_neon.h>
#include <arm_sme.h>

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

static float* tsr_sme_workspace(int which, size_t floats) {
  static _Thread_local float* buffers[2];
  static _Thread_local size_t sizes[2];
  if (sizes[which] < floats) {
    tsr_free(buffers[which]);
    buffers[which] = (float*)tsr_alloc(floats * sizeof(float));
    sizes[which] = floats;
  }
  return buffers[which];
}

/* A[0:m, 0:kc] transposed into 32-row panels: panel p holds A[32p + r, q] at [32q + r]. */
static void tsr_sme_pack_a(int64_t m, int64_t kc, const float* A, int64_t lda, float* out) {
  for (int64_t i0 = 0; i0 < m; i0 += TSR_SME_BLOCK, out += kc * TSR_SME_BLOCK) {
    int64_t rows = m - i0 < TSR_SME_BLOCK ? m - i0 : TSR_SME_BLOCK;
    for (int64_t r0 = 0; r0 < TSR_SME_BLOCK; r0 += 4) {
      const float* a = A + (i0 + r0) * lda;
      int64_t q = 0;
      if (r0 + 4 <= rows) {
        for (; q + 4 <= kc; q += 4) {
          float32x4_t x0 = vld1q_f32(a + q), x1 = vld1q_f32(a + lda + q);
          float32x4_t x2 = vld1q_f32(a + 2 * lda + q), x3 = vld1q_f32(a + 3 * lda + q);
          float32x4_t t0 = vtrn1q_f32(x0, x1), t1 = vtrn2q_f32(x0, x1);
          float32x4_t t2 = vtrn1q_f32(x2, x3), t3 = vtrn2q_f32(x2, x3);
          float64x2_t u0 = vreinterpretq_f64_f32(t0), u1 = vreinterpretq_f64_f32(t1);
          float64x2_t u2 = vreinterpretq_f64_f32(t2), u3 = vreinterpretq_f64_f32(t3);
          float* o = out + q * TSR_SME_BLOCK + r0;
          vst1q_f32(o, vreinterpretq_f32_f64(vtrn1q_f64(u0, u2)));
          vst1q_f32(o + TSR_SME_BLOCK, vreinterpretq_f32_f64(vtrn1q_f64(u1, u3)));
          vst1q_f32(o + 2 * TSR_SME_BLOCK, vreinterpretq_f32_f64(vtrn2q_f64(u0, u2)));
          vst1q_f32(o + 3 * TSR_SME_BLOCK, vreinterpretq_f32_f64(vtrn2q_f64(u1, u3)));
        }
      }
      for (; q < kc; ++q) {
        for (int64_t r = r0; r < r0 + 4; ++r) {
          out[q * TSR_SME_BLOCK + r] = r < rows ? A[(i0 + r) * lda + q] : 0.0f;
        }
      }
    }
  }
}

/* B[0:kc, 0:n] into 32-column panels: panel p holds B[q, 32p + c] at [32q + c].
 * With `transposed`, B[q, j] is read from B[j * ldb + q]. */
static void tsr_sme_pack_b(int64_t kc, int64_t n, const float* B, int64_t ldb, bool transposed,
                           float* out) {
  for (int64_t j0 = 0; j0 < n; j0 += TSR_SME_BLOCK, out += kc * TSR_SME_BLOCK) {
    int64_t cols = n - j0 < TSR_SME_BLOCK ? n - j0 : TSR_SME_BLOCK;
    if (transposed) {
      /* Columns of B are rows of the stored matrix: the A packing does it. */
      tsr_sme_pack_a(cols, kc, B + j0 * ldb, ldb, out);
      continue;
    }
    for (int64_t q = 0; q < kc; ++q) {
      const float* b = B + q * ldb + j0;
      float* o = out + q * TSR_SME_BLOCK;
      if (cols == TSR_SME_BLOCK) {
        for (int c = 0; c < TSR_SME_BLOCK; c += 4) vst1q_f32(o + c, vld1q_f32(b + c));
      } else {
        for (int64_t c = 0; c < TSR_SME_BLOCK; ++c) o[c] = c < cols ? b[c] : 0.0f;
      }
    }
  }
}

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

static void tsr_sme_sgemm_any(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                              const float* B, int64_t ldb, bool b_transposed, float* C,
                              int64_t ldc, bool accumulate) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0) {
    if (!accumulate)
      for (int64_t i = 0; i < m; ++i) memset(C + i * ldc, 0, (size_t)n * sizeof(float));
    return;
  }
  int64_t kc_max = k < TSR_SME_KC ? k : TSR_SME_KC;
  int64_t mc_max = m < TSR_SME_MC ? m : TSR_SME_MC;
  int64_t nc_max = n < TSR_SME_NC ? n : TSR_SME_NC;
  int64_t block = TSR_SME_BLOCK;
  float* ap = tsr_sme_workspace(0, (size_t)((mc_max + block - 1) / block * block * kc_max));
  float* bp = tsr_sme_workspace(1, (size_t)((nc_max + block - 1) / block * block * kc_max));
  for (int64_t j0 = 0; j0 < n; j0 += TSR_SME_NC) {
    int64_t nc = n - j0 < TSR_SME_NC ? n - j0 : TSR_SME_NC;
    for (int64_t p0 = 0; p0 < k; p0 += TSR_SME_KC) {
      int64_t kc = k - p0 < TSR_SME_KC ? k - p0 : TSR_SME_KC;
      const float* b = b_transposed ? B + j0 * ldb + p0 : B + p0 * ldb + j0;
      tsr_sme_pack_b(kc, nc, b, ldb, b_transposed, bp);
      for (int64_t i0 = 0; i0 < m; i0 += TSR_SME_MC) {
        int64_t mc = m - i0 < TSR_SME_MC ? m - i0 : TSR_SME_MC;
        tsr_sme_pack_a(mc, kc, A + i0 * lda + p0, lda, ap);
        tsr_sme_blocks(mc, nc, kc, ap, bp, C + i0 * ldc + j0, ldc, accumulate || p0 > 0);
      }
    }
  }
}

/* C = A B (accumulate == 0) or C += A B, for one m x n tile. */
static void tsr_sme_sgemm(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                          const float* B, int64_t ldb, float* C, int64_t ldc,
                          int64_t accumulate) {
  tsr_sme_sgemm_any(m, n, k, A, lda, B, ldb, false, C, ldc, accumulate != 0);
}

/* The same with B read transposed: B[p, j] at B[j * ldb + p]. */
static void tsr_sme_sgemm_bt(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                             const float* B, int64_t ldb, float* C, int64_t ldc,
                             int64_t accumulate) {
  tsr_sme_sgemm_any(m, n, k, A, lda, B, ldb, true, C, ldc, accumulate != 0);
}

#endif /* TESSERA_SME_H */
