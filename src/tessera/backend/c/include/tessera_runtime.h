/*
 * The Tessera C runtime: a persistent thread pool, aligned allocation, a
 * monotonic timer and the small helpers generated kernels call.
 *
 * Every generated translation unit (a model, or a kernel's timing harness)
 * includes this header once, after defining TSR_NUM_THREADS. Everything is
 * static, so each loaded model has its own pool sized for its target.
 */
#ifndef TESSERA_RUNTIME_H
#define TESSERA_RUNTIME_H

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#else
#define _GNU_SOURCE 1
#endif

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#include <pthread/qos.h>
#endif

#ifndef TSR_NUM_THREADS
#define TSR_NUM_THREADS 1
#endif

/* How long an idle worker polls for the next parallel loop before it sleeps.
 * Kernels of one model run follow each other within microseconds, so workers
 * stay awake through a run; between runs they go to sleep. */
#ifndef TSR_SPIN_SECONDS
#define TSR_SPIN_SECONDS 0.0005
#endif

#define TSR_MAX_THREADS 64
#define TSR_ALIGNMENT 128 /* the cache line of Apple's performance cores */
#define TSR_EXPORT __attribute__((visibility("default")))

/* ---- arithmetic helpers ------------------------------------------------- */

/* Division and remainder rounding toward negative infinity (Python's // and %). */
static inline int64_t tsr_floordiv(int64_t a, int64_t b) {
  int64_t q = a / b, r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

static inline int64_t tsr_floormod(int64_t a, int64_t b) {
  int64_t r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}

static inline float tsr_floordivf(float a, float b) { return floorf(a / b); }

static inline float tsr_floormodf(float a, float b) {
  float r = fmodf(a, b);
  return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r;
}

static inline int64_t tsr_min(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t tsr_max(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t tsr_absi(int64_t a) { return a < 0 ? -a : a; }

static inline int64_t tsr_powi(int64_t a, int64_t b) {
  int64_t r = 1;
  for (; b > 0; --b) r *= a;
  return r;
}

static inline float tsr_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float tsr_rsqrtf(float x) { return 1.0f / sqrtf(x); }

/* ---- memory and time ---------------------------------------------------- */

static void* tsr_alloc(size_t nbytes) {
  void* p = NULL;
  if (posix_memalign(&p, TSR_ALIGNMENT, nbytes ? nbytes : 1) != 0) {
    fprintf(stderr, "tessera: cannot allocate %zu bytes\n", nbytes);
    abort();
  }
  return p;
}

static void tsr_free(void* p) { free(p); }

/* Seconds from an arbitrary origin, monotonic. */
static double tsr_now(void) {
#if defined(__APPLE__)
  static mach_timebase_info_data_t timebase;
  if (timebase.denom == 0) mach_timebase_info(&timebase);
  return (double)mach_absolute_time() * timebase.numer / timebase.denom * 1e-9;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ---- thread pool -------------------------------------------------------- */

/* A parallel loop body: runs iterations [begin, end). */
typedef void (*tsr_task)(void* args, int64_t begin, int64_t end);

/*
 * One pool of TSR_NUM_THREADS - 1 workers; the thread that starts a parallel
 * loop runs the first chunk itself. The iteration range is split statically
 * into one contiguous chunk per thread, which suits the loops templates
 * parallelize (equal-cost tiles).
 *
 * A loop is published by writing the job and bumping `generation`. Workers
 * poll `generation` for TSR_SPIN_SECONDS, then sleep on `wake`; the publisher
 * only takes the mutex when some worker sleeps. Every worker acknowledges
 * every job through `pending`, so the job fields are never rewritten while a
 * worker may still read them.
 */
typedef struct {
  _Atomic(uint64_t) generation;
  _Atomic(int) pending;
  _Atomic(int) sleepers;
  tsr_task task;
  void* args;
  int64_t begin, end;
  int nchunks;
  int nthreads;
  bool forked;
  pthread_mutex_t mutex;
  pthread_cond_t wake;
  pthread_mutex_t owner; /* held while a loop runs; a second caller runs serially */
} tsr_pool_t;

static tsr_pool_t tsr_pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .wake = PTHREAD_COND_INITIALIZER,
    .owner = PTHREAD_MUTEX_INITIALIZER,
};
static pthread_once_t tsr_pool_once = PTHREAD_ONCE_INIT;
static _Thread_local bool tsr_in_parallel = false;

static inline void tsr_pause(void) {
#if defined(__aarch64__) || defined(__arm64__)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

static void tsr_run_chunk(int chunk) {
  int64_t n = tsr_pool.end - tsr_pool.begin;
  int64_t lo = tsr_pool.begin + n * chunk / tsr_pool.nchunks;
  int64_t hi = tsr_pool.begin + n * (chunk + 1) / tsr_pool.nchunks;
  if (lo < hi) tsr_pool.task(tsr_pool.args, lo, hi);
}

static uint64_t tsr_wait_for_job(uint64_t seen) {
  uint64_t gen;
  double start = tsr_now();
  unsigned spins = 0;
  while ((gen = atomic_load_explicit(&tsr_pool.generation, memory_order_acquire)) == seen) {
    tsr_pause();
    if (++spins % 256 == 0 && tsr_now() - start > TSR_SPIN_SECONDS) {
      pthread_mutex_lock(&tsr_pool.mutex);
      atomic_fetch_add(&tsr_pool.sleepers, 1);
      while ((gen = atomic_load(&tsr_pool.generation)) == seen) {
        pthread_cond_wait(&tsr_pool.wake, &tsr_pool.mutex);
      }
      atomic_fetch_sub(&tsr_pool.sleepers, 1);
      pthread_mutex_unlock(&tsr_pool.mutex);
      break;
    }
  }
  return gen;
}

static void* tsr_worker(void* arg) {
  int id = (int)(intptr_t)arg;
  uint64_t seen = 0;
  tsr_in_parallel = true;
  for (;;) {
    seen = tsr_wait_for_job(seen);
    if (id < tsr_pool.nchunks) tsr_run_chunk(id);
    atomic_fetch_sub_explicit(&tsr_pool.pending, 1, memory_order_release);
  }
  return NULL;
}

/* After fork() the child has no workers: run every loop serially there. */
static void tsr_after_fork(void) {
  tsr_pool.forked = true;
  pthread_mutex_init(&tsr_pool.owner, NULL);
}

static void tsr_pool_start(void) {
  int n = TSR_NUM_THREADS < TSR_MAX_THREADS ? TSR_NUM_THREADS : TSR_MAX_THREADS;
#if defined(__APPLE__)
  /* The caller runs a chunk too; keep it off the efficiency cores. */
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, (size_t)8 << 20);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
#if defined(__APPLE__)
  pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  int started = 1;
  for (int i = 1; i < n; ++i) {
    pthread_t thread;
    if (pthread_create(&thread, &attr, tsr_worker, (void*)(intptr_t)i) != 0) break;
    ++started;
  }
  pthread_attr_destroy(&attr);
  tsr_pool.nthreads = started;
  pthread_atfork(NULL, NULL, tsr_after_fork);
}

/* Run task(args, lo, hi) over [begin, end) split across the pool. */
static void tsr_parallel_for(int64_t begin, int64_t end, tsr_task task, void* args) {
  int64_t n = end - begin;
  if (n <= 0) return;
  if (TSR_NUM_THREADS <= 1 || n == 1 || tsr_in_parallel || tsr_pool.forked ||
      pthread_mutex_trylock(&tsr_pool.owner) != 0) {
    task(args, begin, end);
    return;
  }
  pthread_once(&tsr_pool_once, tsr_pool_start);
  if (tsr_pool.nthreads <= 1) {
    pthread_mutex_unlock(&tsr_pool.owner);
    task(args, begin, end);
    return;
  }
  tsr_pool.task = task;
  tsr_pool.args = args;
  tsr_pool.begin = begin;
  tsr_pool.end = end;
  tsr_pool.nchunks = n < tsr_pool.nthreads ? (int)n : tsr_pool.nthreads;
  atomic_store_explicit(&tsr_pool.pending, tsr_pool.nthreads - 1, memory_order_relaxed);
  atomic_fetch_add(&tsr_pool.generation, 1);
  if (atomic_load(&tsr_pool.sleepers) > 0) {
    pthread_mutex_lock(&tsr_pool.mutex);
    pthread_cond_broadcast(&tsr_pool.wake);
    pthread_mutex_unlock(&tsr_pool.mutex);
  }
  tsr_in_parallel = true;
  tsr_run_chunk(0);
  tsr_in_parallel = false;
  while (atomic_load_explicit(&tsr_pool.pending, memory_order_acquire) > 0) tsr_pause();
  pthread_mutex_unlock(&tsr_pool.owner);
}

#endif /* TESSERA_RUNTIME_H */
