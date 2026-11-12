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
#include <sys/sysctl.h>
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

static inline float tsr_rsqrtf(float x) { return 1.0f / sqrtf(x); }

/* ---- vectorizable math -------------------------------------------------- */

/*
 * exp, log, tanh and erf without branches or calls, so that clang vectorizes
 * the loops that use them (libm calls keep a loop scalar). Every case is
 * computed and the answer picked with selects. exp and log use Cephes'
 * single-precision polynomials; tests/c checks the error of every function
 * against float64 over its whole range.
 */

static inline float tsr_bits_float(int32_t bits) {
  union { int32_t i; float f; } u = {bits};
  return u.f;
}

static inline int32_t tsr_float_bits(float x) {
  union { float f; int32_t i; } u = {x};
  return u.i;
}

static inline float tsr_expf(float x) {
  /* x = n ln2 + r with |r| <= ln2 / 2; exp(x) = 2^n exp(r). */
  float xc = fminf(fmaxf(x, -104.0f), 88.7228394f);
  float n = rintf(xc * 1.44269504088896341f);
  float r = fmaf(n, -0.693359375f, xc);
  r = fmaf(n, 2.12194440e-4f, r);
  float p = 1.9875691500e-4f;
  p = fmaf(p, r, 1.3981999507e-3f);
  p = fmaf(p, r, 8.3334519073e-3f);
  p = fmaf(p, r, 4.1665795894e-2f);
  p = fmaf(p, r, 1.6666665459e-1f);
  p = fmaf(p, r, 5.0000001201e-1f);
  p = fmaf(p, r * r, r) + 1.0f;
  /* Multiply by 2^n by adding n to the exponent of p (in [0.7, 1.42]): one
   * integer add, which reassociation cannot turn into an overflowing 2^128.
   * Results below the normal range are made 2^64 times larger, then scaled
   * by one multiplication, which underflows gradually. */
  int32_t ni = (int32_t)n;
  bool deep = ni < -125;
  float y = tsr_bits_float(tsr_float_bits(p) + ((deep ? ni + 64 : ni) << 23));
  y = deep ? y * 5.42101086e-20f : y;
  y = x > 88.7228394f ? INFINITY : y;
  y = x < -104.0f ? 0.0f : y;
  return x != x ? x : y;
}

static inline float tsr_logf(float x) {
  /* x = m 2^e with m in [sqrt(1/2), sqrt(2)); log(x) = e ln2 + log(m). */
  bool tiny = x < 1.17549435e-38f;
  float xs = tiny ? x * 8388608.0f : x; /* subnormals: scale by 2^23 */
  int32_t bits = tsr_float_bits(xs);
  float e = (float)(((bits >> 23) & 0xff) - 126) - (tiny ? 23.0f : 0.0f);
  float m = tsr_bits_float((bits & 0x007fffff) | 0x3f000000); /* in [0.5, 1) */
  bool low = m < 0.707106781186547524f;
  e = low ? e - 1.0f : e;
  float f = (low ? m + m : m) - 1.0f;
  float z = f * f;
  float p = 7.0376836292e-2f;
  p = fmaf(p, f, -1.1514610310e-1f);
  p = fmaf(p, f, 1.1676998740e-1f);
  p = fmaf(p, f, -1.2420140846e-1f);
  p = fmaf(p, f, 1.4249322787e-1f);
  p = fmaf(p, f, -1.6668057665e-1f);
  p = fmaf(p, f, 2.0000714765e-1f);
  p = fmaf(p, f, -2.4999993993e-1f);
  p = fmaf(p, f, 3.3333331174e-1f);
  float y = f * z * p;
  y = fmaf(e, -2.12194440e-4f, y);
  y = fmaf(z, -0.5f, y);
  y = f + y;
  y = fmaf(e, 0.693359375f, y);
  y = x == INFINITY ? INFINITY : y;
  y = x == 0.0f ? -INFINITY : y;
  return x < 0.0f || x != x ? NAN : y;
}

/* tanh and erf as x P(x^2) / Q(x^2) on a clamped range (the rational forms
 * Eigen uses): a dozen FMAs and one division, no exp. Their error is a few
 * ulp where the result approaches +-1 (absolute error below 4e-7). */
static inline float tsr_tanhf(float x) {
  float xc = fminf(fmaxf(x, -7.90531110763549805f), 7.90531110763549805f);
  float z = xc * xc;
  float p = -2.76076847742355e-16f;
  p = fmaf(p, z, 2.00018790482477e-13f);
  p = fmaf(p, z, -8.60467152213735e-11f);
  p = fmaf(p, z, 5.12229709037114e-08f);
  p = fmaf(p, z, 1.48572235717979e-05f);
  p = fmaf(p, z, 6.37261928875436e-04f);
  p = fmaf(p, z, 4.89352455891786e-03f);
  float q = 1.19825839466702e-06f;
  q = fmaf(q, z, 1.18534705686654e-04f);
  q = fmaf(q, z, 2.26843463243900e-03f);
  q = fmaf(q, z, 4.89352518554385e-03f);
  float y = xc * p / q;
  y = fabsf(x) < 0.0004f ? x : y;
  y = fabsf(x) > 9.0f ? copysignf(1.0f, x) : y; /* tanh(9) rounds to 1 */
  return x != x ? x : y;
}

static inline float tsr_erff(float x) {
  float xc = fminf(fmaxf(x, -4.0f), 4.0f);
  float z = xc * xc;
  float p = -2.72614225801306e-10f;
  p = fmaf(p, z, 2.77068142495902e-08f);
  p = fmaf(p, z, -2.10102402082508e-06f);
  p = fmaf(p, z, -5.69250639462346e-05f);
  p = fmaf(p, z, -7.34990630326855e-04f);
  p = fmaf(p, z, -2.95459980854025e-03f);
  p = fmaf(p, z, -1.60960333262415e-02f);
  float q = -1.45660718464996e-05f;
  q = fmaf(q, z, -2.13374055278905e-04f);
  q = fmaf(q, z, -1.68282697438203e-03f);
  q = fmaf(q, z, -7.37332916720468e-03f);
  q = fmaf(q, z, -1.42647390514189e-02f);
  float y = xc * p / q;
  y = fabsf(x) > 3.92f ? copysignf(1.0f, x) : y; /* erf(3.92) rounds to 1 */
  return x != x ? x : y;
}

static inline float tsr_sigmoidf(float x) { return 1.0f / (1.0f + tsr_expf(-x)); }

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
 * loop works on it too. Threads claim iterations from a shared counter, a
 * few at a time (an eighth of an even share), so a thread that runs slower
 * than the others (an efficiency core, a preempted core) takes fewer.
 *
 * A loop is published by writing the job and bumping `generation`. Workers
 * poll `generation` for TSR_SPIN_SECONDS, then sleep on `wake`; the publisher
 * only takes the mutex when some worker sleeps. Every worker acknowledges
 * every job through `pending`, so the job fields are never rewritten while a
 * worker may still read them.
 *
 * A worker that finds itself on an efficiency core when a loop starts leaves
 * the loop to the others and sleeps until the next one, so that its wakeup
 * lets the scheduler place it again. macOS moves threads that keep a shared
 * SME unit busy to the efficiency cluster; a worker kept there would run an
 * SME tile three times slower than the others (docs/cpu.md).
 */
typedef struct {
  _Atomic(uint64_t) generation;
  _Atomic(int) pending;
  _Atomic(int) sleepers;
  _Atomic(int64_t) next; /* the first iteration nobody has claimed */
  tsr_task task;
  void* args;
  int64_t end, grain;
  int nchunks; /* how many threads may work on the loop, the caller included */
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

/* On Apple silicon the efficiency cores are the first hw.perflevel1.logicalcpu
 * logical CPUs; 0 elsewhere (no core is treated as one). Set at pool start. */
static int tsr_efficiency_cpus = 0;

static void tsr_find_efficiency_cores(void) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  int levels = 0, count = 0;
  size_t len = sizeof levels;
  if (sysctlbyname("hw.nperflevels", &levels, &len, NULL, 0) != 0 || levels != 2) return;
  len = sizeof count;
  if (sysctlbyname("hw.perflevel1.logicalcpu", &count, &len, NULL, 0) == 0) {
    tsr_efficiency_cpus = count;
  }
#endif
}

static bool tsr_on_efficiency_core(void) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  size_t cpu = 0;
  return tsr_efficiency_cpus > 0 && pthread_cpu_number_np(&cpu) == 0 &&
         cpu < (size_t)tsr_efficiency_cpus;
#else
  return false;
#endif
}

static inline void tsr_pause(void) {
#if defined(__aarch64__) || defined(__arm64__)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

/* Claim and run iterations of the current loop until none is left. */
static void tsr_work(void) {
  for (;;) {
    int64_t lo = atomic_fetch_add_explicit(&tsr_pool.next, tsr_pool.grain, memory_order_relaxed);
    if (lo >= tsr_pool.end) return;
    int64_t hi = tsr_pool.end - lo < tsr_pool.grain ? tsr_pool.end : lo + tsr_pool.grain;
    tsr_pool.task(tsr_pool.args, lo, hi);
  }
}

/* Wait for a job newer than `seen`, polling for TSR_SPIN_SECONDS first when
 * `poll` is set, and return its generation. */
static uint64_t tsr_wait_for_job(uint64_t seen, bool poll) {
  uint64_t gen;
  double start = tsr_now();
  unsigned spins = 0;
  while ((gen = atomic_load_explicit(&tsr_pool.generation, memory_order_acquire)) == seen) {
    tsr_pause();
    if (!poll || (++spins % 256 == 0 && tsr_now() - start > TSR_SPIN_SECONDS)) {
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
  bool poll = true;
  tsr_in_parallel = true;
  for (;;) {
    seen = tsr_wait_for_job(seen, poll);
    poll = !tsr_on_efficiency_core();
    if (poll && id < tsr_pool.nchunks) tsr_work();
    atomic_fetch_sub_explicit(&tsr_pool.pending, 1, memory_order_release);
  }
  return NULL;
}

/* After fork() the child has no workers: run every loop serially there. */
static void tsr_after_fork(void) {
  tsr_pool.forked = true;
  pthread_mutex_init(&tsr_pool.owner, NULL);
}

/* Ask the scheduler to keep the calling thread on the performance cores
 * (macOS places threads by QoS class). Every entry point of a generated
 * library calls it, once per thread: the caller runs the serial parts of a
 * model and works on every parallel loop. */
static void tsr_prefer_performance_cores(void) {
#if defined(__APPLE__)
  static _Thread_local bool done = false;
  if (!done) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    done = true;
  }
#endif
}

static void tsr_pool_start(void) {
  int n = TSR_NUM_THREADS < TSR_MAX_THREADS ? TSR_NUM_THREADS : TSR_MAX_THREADS;
  tsr_prefer_performance_cores();
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, (size_t)8 << 20);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
#if defined(__APPLE__)
  pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  tsr_find_efficiency_cores();
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

/* Run task(args, lo, hi) over [begin, end) split across at most `threads`
 * threads of the pool (the others stay idle: SME kernels use fewer threads
 * than there are cores, because the cores share one SME unit). */
static void tsr_parallel_for_n(int64_t begin, int64_t end, int threads, tsr_task task,
                               void* args) {
  int64_t n = end - begin;
  if (n <= 0) return;
  if (TSR_NUM_THREADS <= 1 || threads <= 1 || n == 1 || tsr_in_parallel || tsr_pool.forked ||
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
  tsr_pool.end = end;
  int most = threads < tsr_pool.nthreads ? threads : tsr_pool.nthreads;
  tsr_pool.nchunks = n < most ? (int)n : most;
  int64_t grain = n / (8 * (int64_t)tsr_pool.nchunks);
  tsr_pool.grain = grain > 1 ? grain : 1;
  atomic_store_explicit(&tsr_pool.next, begin, memory_order_relaxed);
  atomic_store_explicit(&tsr_pool.pending, tsr_pool.nthreads - 1, memory_order_relaxed);
  atomic_fetch_add(&tsr_pool.generation, 1);
  if (atomic_load(&tsr_pool.sleepers) > 0) {
    pthread_mutex_lock(&tsr_pool.mutex);
    pthread_cond_broadcast(&tsr_pool.wake);
    pthread_mutex_unlock(&tsr_pool.mutex);
  }
  tsr_in_parallel = true;
  tsr_work();
  tsr_in_parallel = false;
  while (atomic_load_explicit(&tsr_pool.pending, memory_order_acquire) > 0) tsr_pause();
  pthread_mutex_unlock(&tsr_pool.owner);
}

/* Run task(args, lo, hi) over [begin, end) split across the whole pool. */
static void tsr_parallel_for(int64_t begin, int64_t end, tsr_task task, void* args) {
  tsr_parallel_for_n(begin, end, TSR_NUM_THREADS, task, args);
}

#endif /* TESSERA_RUNTIME_H */
