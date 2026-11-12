/*
 * Packing for the GEMM and convolution microkernels.
 *
 * A microkernel computing a P x Q block of C reads, for every step q of the
 * reduction, P values of A and Q values of B. Packing copies a block of each
 * operand into "panels" where those values are contiguous (k-major) and
 * zero-padded to whole panels, so the microkernel's loads are sequential and
 * it never tests for edges:
 *
 *   rows panel (A):    out[q * P + r] = A[r, q]      r < P, q < kc
 *   columns panel (B): out[q * Q + c] = B[q, c]      c < Q, q < kc
 *
 * For a convolution, A is the im2col matrix of an NHWC image, which is never
 * built: tsr_pack_im2col reads the rows (output pixels) and columns
 * (kh, kw, c) it needs straight from the image, with zeros for padding.
 * Packing uses NEON outside streaming mode, for both the NEON and the SME
 * kernels.
 */
#ifndef TESSERA_PACK_H
#define TESSERA_PACK_H

#include <arm_neon.h>

#include "tessera_runtime.h"

/* Per-thread packing buffers, grown on demand and kept for the next call. */
static float* tsr_pack_buffer(int which, size_t floats) {
  static _Thread_local float* buffers[2];
  static _Thread_local size_t sizes[2];
  if (sizes[which] < floats) {
    tsr_free(buffers[which]);
    buffers[which] = (float*)tsr_alloc(floats * sizeof(float));
    sizes[which] = floats;
  }
  return buffers[which];
}

/* out[q*4 + j] (j < 4, q < 4) = rows[j][q]: a 4 x 4 transpose into a panel of
 * height `height` (the four values for one q land at out + q * height). */
static inline void tsr_pack_transpose4(float32x4_t r0, float32x4_t r1, float32x4_t r2,
                                       float32x4_t r3, float* out, int64_t height) {
  float64x2_t t0 = vreinterpretq_f64_f32(vtrn1q_f32(r0, r1));
  float64x2_t t1 = vreinterpretq_f64_f32(vtrn2q_f32(r0, r1));
  float64x2_t t2 = vreinterpretq_f64_f32(vtrn1q_f32(r2, r3));
  float64x2_t t3 = vreinterpretq_f64_f32(vtrn2q_f32(r2, r3));
  vst1q_f32(out, vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)));
  vst1q_f32(out + height, vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)));
  vst1q_f32(out + 2 * height, vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)));
  vst1q_f32(out + 3 * height, vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)));
}

/* Rows [0, m) x columns [0, kc) of A (A[r, q] at A[r * lda + q]) into panels
 * of `height` rows (a multiple of 4). */
static void tsr_pack_rows(int64_t height, int64_t m, int64_t kc, const float* A, int64_t lda,
                          float* out) {
  for (int64_t i0 = 0; i0 < m; i0 += height, out += kc * height) {
    int64_t rows = m - i0 < height ? m - i0 : height;
    for (int64_t g = 0; g < height; g += 4) {
      const float* a = A + (i0 + g) * lda;
      int64_t q = 0;
      if (g + 4 <= rows) {
        for (; q + 4 <= kc; q += 4) {
          tsr_pack_transpose4(vld1q_f32(a + q), vld1q_f32(a + lda + q),
                              vld1q_f32(a + 2 * lda + q), vld1q_f32(a + 3 * lda + q),
                              out + q * height + g, height);
        }
      }
      for (; q < kc; ++q) {
        for (int64_t r = g; r < g + 4; ++r) {
          out[q * height + r] = r < rows ? A[(i0 + r) * lda + q] : 0.0f;
        }
      }
    }
  }
}

/* Rows [0, kc) x columns [0, n) of B (B[q, c] at B[q * ldb + c]) into panels
 * of `width` columns (a multiple of 4). */
static void tsr_pack_cols(int64_t width, int64_t kc, int64_t n, const float* B, int64_t ldb,
                          float* out) {
  for (int64_t j0 = 0; j0 < n; j0 += width, out += kc * width) {
    int64_t cols = n - j0 < width ? n - j0 : width;
    for (int64_t q = 0; q < kc; ++q) {
      const float* b = B + q * ldb + j0;
      float* o = out + q * width;
      if (cols == width) {
        for (int64_t c = 0; c < width; c += 4) vst1q_f32(o + c, vld1q_f32(b + c));
      } else {
        for (int64_t c = 0; c < width; ++c) o[c] = c < cols ? b[c] : 0.0f;
      }
    }
  }
}

/* The geometry of a 2-D convolution over one NHWC image. */
typedef struct {
  int64_t h, w, c;    /* input height, width, channels */
  int64_t kh, kw;     /* kernel height and width */
  int64_t sh, sw;     /* strides */
  int64_t ph, pw;     /* top and left padding */
  int64_t dh, dw;     /* dilations */
  int64_t ow;         /* output width */
} tsr_conv_shape;

/*
 * Rows [p0, p0 + m) x columns [k0, k0 + kc) of the im2col matrix of image x
 * into panels of `height` rows. Row p is output pixel (p / ow, p % ow);
 * column k is (kh, kw, c) with k = (kh * KW + kw) * C + c, which is also the
 * row of the HWIO weights viewed as a [KH*KW*C, O] matrix.
 */
static void tsr_pack_im2col(int64_t height, int64_t m, int64_t kc, int64_t p0, int64_t k0,
                            const float* x, const tsr_conv_shape* s, float* out) {
  int64_t c0 = k0 % s->c, kw0 = (k0 / s->c) % s->kw, kh0 = k0 / (s->c * s->kw);
  for (int64_t i0 = 0; i0 < m; i0 += height, out += kc * height) {
    for (int64_t g = 0; g < height; g += 4) {
      /* Four output pixels at a time; a pixel past the end reads as zeros. */
      int64_t ih0[4], iw0[4];
      bool live[4];
      for (int j = 0; j < 4; ++j) {
        int64_t r = i0 + g + j;
        live[j] = r < m;
        int64_t p = p0 + (live[j] ? r : 0);
        ih0[j] = (p / s->ow) * s->sh - s->ph;
        iw0[j] = (p % s->ow) * s->sw - s->pw;
      }
      int64_t q = 0, c = c0, kw = kw0, kh = kh0;
      while (q < kc) {
        int64_t run = s->c - c < kc - q ? s->c - c : kc - q;
        const float* src[4];
        for (int j = 0; j < 4; ++j) {
          int64_t ih = ih0[j] + kh * s->dh, iw = iw0[j] + kw * s->dw;
          bool inside = live[j] && ih >= 0 && ih < s->h && iw >= 0 && iw < s->w;
          src[j] = inside ? x + (ih * s->w + iw) * s->c + c : NULL;
        }
        int64_t t = 0;
        for (; t + 4 <= run; t += 4) {
          float32x4_t zero = vdupq_n_f32(0.0f);
          tsr_pack_transpose4(src[0] ? vld1q_f32(src[0] + t) : zero,
                              src[1] ? vld1q_f32(src[1] + t) : zero,
                              src[2] ? vld1q_f32(src[2] + t) : zero,
                              src[3] ? vld1q_f32(src[3] + t) : zero,
                              out + (q + t) * height + g, height);
        }
        for (; t < run; ++t) {
          for (int j = 0; j < 4; ++j) out[(q + t) * height + g + j] = src[j] ? src[j][t] : 0.0f;
        }
        q += run;
        c += run;
        if (c == s->c) {
          c = 0;
          if (++kw == s->kw) {
            kw = 0;
            ++kh;
          }
        }
      }
    }
  }
}

/* ---- the blocked driver shared by every GEMM-like kernel ----------------- */

/* Computes C[0:mc, 0:nc] (+)= packed A . packed B over kc steps. */
typedef void (*tsr_macro_fn)(int64_t mc, int64_t nc, int64_t kc, const float* ap,
                             const float* bp, float* c, int64_t ldc, bool load);

/* Panel sizes of a microkernel and the cache blocking around it. */
typedef struct {
  int64_t mr, nr;     /* panel height of A and width of B */
  int64_t mc, nc, kc; /* block of A, block of B, block of the reduction */
  tsr_macro_fn macro;
} tsr_blocking;

/* The A operand: a strided matrix (conv == NULL) or the im2col matrix of an
 * image starting at output pixel p0. */
typedef struct {
  const float* a;
  int64_t lda;
  const tsr_conv_shape* conv;
  int64_t p0;
} tsr_lhs;

/*
 * C[0:m, 0:n] (+)= A[0:m, 0:k] B[0:k, 0:n], BLIS-style:
 *
 *   for each block of nc columns:            pack B (kc x nc) into nr-wide panels
 *     for each block of kc reduction steps:
 *       for each block of mc rows:           pack A (mc x kc) into mr-high panels
 *         macro: every mr x nr block of the mc x nc block of C
 *
 * When A is a single block of rows (m <= mc), the column loop moves inside
 * the reduction loop, so A is packed once per block of the reduction rather
 * than once per block of columns.
 *
 * B[q, j] is B[q * ldb + j], or B[j * ldb + q] with b_transposed.
 */
/* Where the driver gets its packed blocks of B: packed on the spot into a
 * scratch buffer, or taken in order from panels packed earlier. */
typedef struct {
  const float* b;
  int64_t ldb, nr;
  bool transposed;
  float* scratch;
  const float* cached; /* the next block of a packing made earlier, or NULL */
} tsr_rhs;

static const float* tsr_rhs_block(tsr_rhs* r, int64_t j0, int64_t nc, int64_t p0, int64_t kc) {
  if (r->cached) {
    const float* block = r->cached;
    r->cached += (nc + r->nr - 1) / r->nr * r->nr * kc;
    return block;
  }
  if (r->transposed) {
    tsr_pack_rows(r->nr, nc, kc, r->b + j0 * r->ldb + p0, r->ldb, r->scratch);
  } else {
    tsr_pack_cols(r->nr, kc, nc, r->b + p0 * r->ldb + j0, r->ldb, r->scratch);
  }
  return r->scratch;
}

/* The order in which the driver visits blocks of B: by blocks of the
 * reduction then of columns when A is a single block of rows, else by
 * blocks of columns then of the reduction. */
static bool tsr_reduction_outer(const tsr_blocking* blk, int64_t m) { return m <= blk->mc; }

/* ---- weights packed once ------------------------------------------------ */

/*
 * A B operand that is a model constant (a weight) is packed at its first
 * call, every block in the order the driver uses them, and the packing is
 * kept for later calls: for the skinny products of a transformer, packing
 * B costs about as much as the multiplication. Entries are keyed by the
 * operand's address and geometry. tsr_forget_panels drops them all; the
 * executor calls it whenever it loads a model (and run_kernel before every
 * call), so a packing is never reused for other data that came to live at
 * the same address. The packing costs as much memory as the weights.
 *
 * The packings are found through a hash table (open addressing, linear
 * probing, at most half full): a tuned model makes thousands of them, one
 * per column strip of every weight, and looks one up at every call.
 */
typedef struct {
  const float* b;
  int64_t ldb, k, n;
  bool transposed, reduction_outer;
  const tsr_blocking* blk;
  float* panels; /* NULL: an empty slot */
} tsr_panel_entry;

static tsr_panel_entry* tsr_panel_table;
static size_t tsr_panel_count, tsr_panel_capacity; /* capacity: 0 or a power of two */
static pthread_mutex_t tsr_panel_mutex = PTHREAD_MUTEX_INITIALIZER;

static size_t tsr_panel_slot(const float* b, int64_t n, int64_t k, size_t capacity) {
  uint64_t h = (uint64_t)(uintptr_t)b * 0x9E3779B97F4A7C15ull;
  h ^= ((uint64_t)n << 32 | (uint64_t)k) * 0xC2B2AE3D27D4EB4Full;
  return (size_t)(h ^ h >> 31) & (capacity - 1);
}

static bool tsr_panel_matches(const tsr_panel_entry* e, const tsr_panel_entry* key) {
  return e->b == key->b && e->ldb == key->ldb && e->k == key->k && e->n == key->n &&
         e->transposed == key->transposed && e->reduction_outer == key->reduction_outer &&
         e->blk == key->blk;
}

/* The slot of `key` in the table, or the empty slot where it belongs. */
static tsr_panel_entry* tsr_panel_find(const tsr_panel_entry* key) {
  size_t i = tsr_panel_slot(key->b, key->n, key->k, tsr_panel_capacity);
  while (tsr_panel_table[i].panels && !tsr_panel_matches(&tsr_panel_table[i], key)) {
    i = (i + 1) & (tsr_panel_capacity - 1);
  }
  return &tsr_panel_table[i];
}

/* Makes room for one more entry, keeping the table at most half full. */
static void tsr_panel_reserve(void) {
  if (2 * (tsr_panel_count + 1) <= tsr_panel_capacity) return;
  tsr_panel_entry* old = tsr_panel_table;
  size_t old_capacity = tsr_panel_capacity;
  tsr_panel_capacity = old_capacity ? 2 * old_capacity : 256;
  tsr_panel_table = (tsr_panel_entry*)calloc(tsr_panel_capacity, sizeof(tsr_panel_entry));
  for (size_t i = 0; i < old_capacity; ++i) {
    if (old[i].panels) *tsr_panel_find(&old[i]) = old[i];
  }
  free(old);
}

static void tsr_forget_panels(void) {
  pthread_mutex_lock(&tsr_panel_mutex);
  for (size_t i = 0; i < tsr_panel_capacity; ++i) {
    tsr_free(tsr_panel_table[i].panels);
    tsr_panel_table[i].panels = NULL;
  }
  tsr_panel_count = 0;
  pthread_mutex_unlock(&tsr_panel_mutex);
}

static const float* tsr_constant_panels(const tsr_blocking* blk, int64_t m, int64_t n, int64_t k,
                                        const float* b, int64_t ldb, bool transposed) {
  tsr_panel_entry key = {b, ldb, k, n, transposed, tsr_reduction_outer(blk, m), blk, NULL};
  bool outer = key.reduction_outer;
  pthread_mutex_lock(&tsr_panel_mutex);
  tsr_panel_reserve();
  tsr_panel_entry* slot = tsr_panel_find(&key);
  if (slot->panels) {
    pthread_mutex_unlock(&tsr_panel_mutex);
    return slot->panels;
  }
  int64_t n_pad = 0;
  for (int64_t j0 = 0; j0 < n; j0 += blk->nc) {
    int64_t nc = n - j0 < blk->nc ? n - j0 : blk->nc;
    n_pad += (nc + blk->nr - 1) / blk->nr * blk->nr;
  }
  float* panels = (float*)tsr_alloc((size_t)(n_pad * k) * sizeof(float));
  float* out = panels;
  tsr_rhs r = {b, ldb, blk->nr, transposed, NULL, NULL};
  for (int64_t outer_i = 0; outer_i < (outer ? k : n); outer_i += outer ? blk->kc : blk->nc) {
    for (int64_t inner_i = 0; inner_i < (outer ? n : k); inner_i += outer ? blk->nc : blk->kc) {
      int64_t p0 = outer ? outer_i : inner_i, j0 = outer ? inner_i : outer_i;
      int64_t kc = k - p0 < blk->kc ? k - p0 : blk->kc;
      int64_t nc = n - j0 < blk->nc ? n - j0 : blk->nc;
      r.scratch = out;
      tsr_rhs_block(&r, j0, nc, p0, kc);
      out += (nc + blk->nr - 1) / blk->nr * blk->nr * kc;
    }
  }
  key.panels = panels;
  *slot = key;
  ++tsr_panel_count;
  pthread_mutex_unlock(&tsr_panel_mutex);
  return panels;
}

/* ---- the blocked driver shared by every GEMM-like kernel ----------------- */

/*
 * C[0:m, 0:n] (+)= A[0:m, 0:k] B[0:k, 0:n], BLIS-style:
 *
 *   for each block of nc columns:            pack B (kc x nc) into nr-wide panels
 *     for each block of kc reduction steps:
 *       for each block of mc rows:           pack A (mc x kc) into mr-high panels
 *         macro: every mr x nr block of the mc x nc block of C
 *
 * When A is a single block of rows (m <= mc), the column loop moves inside
 * the reduction loop, so A is packed once per block of the reduction rather
 * than once per block of columns. With `constant_b`, B's blocks come from a
 * packing kept across calls (tsr_constant_panels).
 *
 * B[q, j] is B[q * ldb + j], or B[j * ldb + q] with b_transposed.
 */
static void tsr_blocked_gemm(const tsr_blocking* blk, int64_t m, int64_t n, int64_t k,
                             const tsr_lhs* lhs, const float* B, int64_t ldb, bool b_transposed,
                             float* C, int64_t ldc, bool accumulate, bool constant_b) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0) {
    if (!accumulate)
      for (int64_t i = 0; i < m; ++i) memset(C + i * ldc, 0, (size_t)n * sizeof(float));
    return;
  }
  int64_t kc_max = k < blk->kc ? k : blk->kc;
  int64_t mc_max = m < blk->mc ? m : blk->mc;
  int64_t nc_max = n < blk->nc ? n : blk->nc;
  float* ap = tsr_pack_buffer(1, (size_t)((mc_max + blk->mr - 1) / blk->mr * blk->mr * kc_max));
  tsr_rhs rhs = {B, ldb, blk->nr, b_transposed, NULL, NULL};
  if (constant_b) {
    rhs.cached = tsr_constant_panels(blk, m, n, k, B, ldb, b_transposed);
  } else {
    rhs.scratch =
        tsr_pack_buffer(0, (size_t)((nc_max + blk->nr - 1) / blk->nr * blk->nr * kc_max));
  }
  if (tsr_reduction_outer(blk, m)) {
    /* One block of rows (the skinny products of a transformer): pack A once
     * per block of the reduction and walk every block of columns under it. */
    for (int64_t p0 = 0; p0 < k; p0 += blk->kc) {
      int64_t kc = k - p0 < blk->kc ? k - p0 : blk->kc;
      if (lhs->conv) {
        tsr_pack_im2col(blk->mr, m, kc, lhs->p0, p0, lhs->a, lhs->conv, ap);
      } else {
        tsr_pack_rows(blk->mr, m, kc, lhs->a + p0, lhs->lda, ap);
      }
      for (int64_t j0 = 0; j0 < n; j0 += blk->nc) {
        int64_t nc = n - j0 < blk->nc ? n - j0 : blk->nc;
        const float* bp = tsr_rhs_block(&rhs, j0, nc, p0, kc);
        blk->macro(m, nc, kc, ap, bp, C + j0, ldc, accumulate || p0 > 0);
      }
    }
    return;
  }
  for (int64_t j0 = 0; j0 < n; j0 += blk->nc) {
    int64_t nc = n - j0 < blk->nc ? n - j0 : blk->nc;
    for (int64_t p0 = 0; p0 < k; p0 += blk->kc) {
      int64_t kc = k - p0 < blk->kc ? k - p0 : blk->kc;
      const float* bp = tsr_rhs_block(&rhs, j0, nc, p0, kc);
      for (int64_t i0 = 0; i0 < m; i0 += blk->mc) {
        int64_t mc = m - i0 < blk->mc ? m - i0 : blk->mc;
        if (lhs->conv) {
          tsr_pack_im2col(blk->mr, mc, kc, lhs->p0 + i0, p0, lhs->a, lhs->conv, ap);
        } else {
          tsr_pack_rows(blk->mr, mc, kc, lhs->a + i0 * lhs->lda + p0, lhs->lda, ap);
        }
        blk->macro(mc, nc, kc, ap, bp, C + i0 * ldc + j0, ldc, accumulate || p0 > 0);
      }
    }
  }
}

#endif /* TESSERA_PACK_H */
