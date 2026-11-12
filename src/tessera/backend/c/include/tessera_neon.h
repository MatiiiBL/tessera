/*
 * NEON fp32 GEMM for one tile: C[0:m, 0:n] (+)= A[0:m, 0:k] B[0:k, 0:n].
 *
 * The operands are row-major with leading dimensions (A[i, p] is
 * A[i * lda + p]); B can also be read transposed (B[p, j] is B[j * ldb + p]),
 * which is how attention multiplies by K^T. The tile is processed BLIS-style:
 *
 *   for each block of TSR_NEON_NC columns of C:
 *   for each block of TSR_NEON_KC rows of B:      pack B into NR-column panels
 *     for each block of TSR_NEON_MC rows of A:    pack A into MR-row panels
 *       for each NR-column panel:                 (stays in L1)
 *         for each MR-row panel:                  (the A block stays in L2)
 *           8 x 12 register-blocked microkernel
 *
 * Packed panels are k-major: for every k, the MR values of A (or NR values
 * of B) the microkernel needs are contiguous, so its loads are sequential and
 * edges are zero-padded. The microkernel keeps the 8 x 12 tile of C in 24 of
 * the 32 vector registers and per k issues 2 + 3 loads for 24 FMAs (each FMA
 * multiplies a row of B by one lane of A).
 */
#ifndef TESSERA_NEON_H
#define TESSERA_NEON_H

#include <arm_neon.h>

#include "tessera_runtime.h"

#define TSR_NEON_MR 8
#define TSR_NEON_NR 12
#ifndef TSR_NEON_KC
#define TSR_NEON_KC 256
#endif
#ifndef TSR_NEON_MC
#define TSR_NEON_MC 96
#endif
#ifndef TSR_NEON_NC
#define TSR_NEON_NC 2040 /* a multiple of NR */
#endif

/* Per-thread packing buffers, grown on demand and kept for the next call. */
static float* tsr_neon_workspace(int which, size_t floats) {
  static _Thread_local float* buffers[2];
  static _Thread_local size_t sizes[2];
  if (sizes[which] < floats) {
    tsr_free(buffers[which]);
    buffers[which] = (float*)tsr_alloc(floats * sizeof(float));
    sizes[which] = floats;
  }
  return buffers[which];
}

/* A[0:m, 0:kc] into MR-row panels: panel p holds A[p*MR + r, q] at [q*MR + r]. */
static void tsr_neon_pack_a(int64_t m, int64_t kc, const float* A, int64_t lda, float* out) {
  for (int64_t i0 = 0; i0 < m; i0 += TSR_NEON_MR, out += kc * TSR_NEON_MR) {
    int64_t rows = m - i0 < TSR_NEON_MR ? m - i0 : TSR_NEON_MR;
    const float* a = A + i0 * lda;
    int64_t q = 0;
    if (rows == TSR_NEON_MR) {
      /* Four columns at a time: two 4x4 transposes. */
      for (; q + 4 <= kc; q += 4) {
        for (int half = 0; half < 2; ++half) {
          const float* r = a + half * 4 * lda + q;
          float32x4_t r0 = vld1q_f32(r), r1 = vld1q_f32(r + lda);
          float32x4_t r2 = vld1q_f32(r + 2 * lda), r3 = vld1q_f32(r + 3 * lda);
          float32x4_t t0 = vtrn1q_f32(r0, r1), t1 = vtrn2q_f32(r0, r1);
          float32x4_t t2 = vtrn1q_f32(r2, r3), t3 = vtrn2q_f32(r2, r3);
          float* o = out + q * TSR_NEON_MR + half * 4;
          vst1q_f32(o, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0),
                                                        vreinterpretq_f64_f32(t2))));
          vst1q_f32(o + TSR_NEON_MR, vreinterpretq_f32_f64(vtrn1q_f64(
                                         vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
          vst1q_f32(o + 2 * TSR_NEON_MR, vreinterpretq_f32_f64(vtrn2q_f64(
                                             vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))));
          vst1q_f32(o + 3 * TSR_NEON_MR, vreinterpretq_f32_f64(vtrn2q_f64(
                                             vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
        }
      }
    }
    for (; q < kc; ++q) {
      for (int64_t r = 0; r < TSR_NEON_MR; ++r) {
        out[q * TSR_NEON_MR + r] = r < rows ? a[r * lda + q] : 0.0f;
      }
    }
  }
}

/* B[0:kc, 0:n] into NR-column panels: panel p holds B[q, p*NR + c] at [q*NR + c].
 * With `transposed`, B[q, j] is read from B[j * ldb + q]. */
static void tsr_neon_pack_b(int64_t kc, int64_t n, const float* B, int64_t ldb, bool transposed,
                            float* out) {
  for (int64_t j0 = 0; j0 < n; j0 += TSR_NEON_NR, out += kc * TSR_NEON_NR) {
    int64_t cols = n - j0 < TSR_NEON_NR ? n - j0 : TSR_NEON_NR;
    if (!transposed && cols == TSR_NEON_NR) {
      for (int64_t q = 0; q < kc; ++q) {
        const float* b = B + q * ldb + j0;
        float* o = out + q * TSR_NEON_NR;
        vst1q_f32(o, vld1q_f32(b));
        vst1q_f32(o + 4, vld1q_f32(b + 4));
        vst1q_f32(o + 8, vld1q_f32(b + 8));
      }
      continue;
    }
    for (int64_t q = 0; q < kc; ++q) {
      for (int64_t c = 0; c < TSR_NEON_NR; ++c) {
        float v = 0.0f;
        if (c < cols) v = transposed ? B[(j0 + c) * ldb + q] : B[q * ldb + j0 + c];
        out[q * TSR_NEON_NR + c] = v;
      }
    }
  }
}

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

/* A partial tile goes through a full-size scratch tile. */
static void tsr_neon_micro_edge(int64_t kc, const float* a, const float* b, float* c, int64_t ldc,
                                int64_t rows, int64_t cols, bool load) {
  float tile[TSR_NEON_MR * TSR_NEON_NR] __attribute__((aligned(64)));
  if (load) {
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t j = 0; j < cols; ++j) tile[r * TSR_NEON_NR + j] = c[r * ldc + j];
  }
  tsr_neon_micro_8x12(kc, a, b, tile, TSR_NEON_NR, load);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t j = 0; j < cols; ++j) c[r * ldc + j] = tile[r * TSR_NEON_NR + j];
}

static void tsr_neon_sgemm_any(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                               const float* B, int64_t ldb, bool b_transposed, float* C,
                               int64_t ldc, bool accumulate) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0) {
    if (!accumulate)
      for (int64_t i = 0; i < m; ++i) memset(C + i * ldc, 0, (size_t)n * sizeof(float));
    return;
  }
  int64_t kc_max = k < TSR_NEON_KC ? k : TSR_NEON_KC;
  int64_t mc_max = m < TSR_NEON_MC ? m : TSR_NEON_MC;
  int64_t nc_max = n < TSR_NEON_NC ? n : TSR_NEON_NC;
  float* bp = tsr_neon_workspace(
      0, (size_t)((nc_max + TSR_NEON_NR - 1) / TSR_NEON_NR * TSR_NEON_NR * kc_max));
  float* ap = tsr_neon_workspace(
      1, (size_t)((mc_max + TSR_NEON_MR - 1) / TSR_NEON_MR * TSR_NEON_MR * kc_max));
  for (int64_t j0 = 0; j0 < n; j0 += TSR_NEON_NC) {
    int64_t nc = n - j0 < TSR_NEON_NC ? n - j0 : TSR_NEON_NC;
    for (int64_t p0 = 0; p0 < k; p0 += TSR_NEON_KC) {
      int64_t kc = k - p0 < TSR_NEON_KC ? k - p0 : TSR_NEON_KC;
      bool load = accumulate || p0 > 0;
      const float* bsrc = b_transposed ? B + j0 * ldb + p0 : B + p0 * ldb + j0;
      tsr_neon_pack_b(kc, nc, bsrc, ldb, b_transposed, bp);
      for (int64_t i0 = 0; i0 < m; i0 += TSR_NEON_MC) {
        int64_t mc = m - i0 < TSR_NEON_MC ? m - i0 : TSR_NEON_MC;
        tsr_neon_pack_a(mc, kc, A + i0 * lda + p0, lda, ap);
        for (int64_t j = 0; j < nc; j += TSR_NEON_NR) {
          const float* b = bp + (j / TSR_NEON_NR) * kc * TSR_NEON_NR;
          int64_t cols = nc - j < TSR_NEON_NR ? nc - j : TSR_NEON_NR;
          for (int64_t i = 0; i < mc; i += TSR_NEON_MR) {
            const float* a = ap + (i / TSR_NEON_MR) * kc * TSR_NEON_MR;
            float* c = C + (i0 + i) * ldc + j0 + j;
            int64_t rows = mc - i < TSR_NEON_MR ? mc - i : TSR_NEON_MR;
            if (rows == TSR_NEON_MR && cols == TSR_NEON_NR) {
              tsr_neon_micro_8x12(kc, a, b, c, ldc, load);
            } else {
              tsr_neon_micro_edge(kc, a, b, c, ldc, rows, cols, load);
            }
          }
        }
      }
    }
  }
}

/* C = A B (accumulate == 0) or C += A B, for one m x n tile. */
static void tsr_neon_sgemm(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                           const float* B, int64_t ldb, float* C, int64_t ldc,
                           int64_t accumulate) {
  tsr_neon_sgemm_any(m, n, k, A, lda, B, ldb, false, C, ldc, accumulate != 0);
}

/* The same with B read transposed: B[p, j] at B[j * ldb + p]. */
static void tsr_neon_sgemm_bt(int64_t m, int64_t n, int64_t k, const float* A, int64_t lda,
                              const float* B, int64_t ldb, float* C, int64_t ldc,
                              int64_t accumulate) {
  tsr_neon_sgemm_any(m, n, k, A, lda, B, ldb, true, C, ldc, accumulate != 0);
}

#endif /* TESSERA_NEON_H */
