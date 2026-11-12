"""The C runtime: thread pool, allocation and timer, exercised from a C program."""

from __future__ import annotations

import subprocess

import pytest

from tessera.backend.c import build as cbuild
from tessera.backend.c.microkernels import header_text
from tessera.target import Target

POOL_TEST = r"""
#define TSR_NUM_THREADS 4
#define TSR_SPIN_SECONDS 0.0002
#include "tessera_runtime.h"
#include <assert.h>
#include <sys/wait.h>
#include <unistd.h>

static _Atomic(int) hits[4096];

static void count(void* args, int64_t begin, int64_t end) {
  (void)args;
  for (int64_t i = begin; i < end; ++i) atomic_fetch_add(&hits[i], 1);
}

/* Every index in [lo, hi) ran `expected` times; reset the counts. */
static void check(int64_t lo, int64_t hi, int expected) {
  for (int64_t i = lo; i < hi; ++i) {
    if (hits[i] != expected) {
      fprintf(stderr, "[%lld, %lld): index %lld ran %d times\n", (long long)lo, (long long)hi,
              (long long)i, hits[i]);
      exit(1);
    }
    hits[i] = 0;
  }
}

static void nested(void* args, int64_t begin, int64_t end) {
  (void)args;
  /* A loop started from inside a parallel loop runs on the calling thread. */
  for (int64_t i = begin; i < end; ++i) tsr_parallel_for(i * 8, i * 8 + 8, count, NULL);
}

static void* caller(void* arg) {
  (void)arg;
  for (int r = 0; r < 500; ++r) tsr_parallel_for(0, 64, count, NULL);
  return NULL;
}

int main(void) {
  static const int64_t sizes[] = {1, 2, 3, 4, 5, 7, 64, 1000, 4096};
  for (int r = 0; r < 2000; ++r) {
    int64_t n = sizes[r % 9];
    tsr_parallel_for(0, n, count, NULL);
    check(0, n, 1);
  }
  /* Offset ranges. */
  tsr_parallel_for(100, 200, count, NULL);
  check(100, 200, 1);
  /* Workers asleep after the spin period wake up for the next loop. */
  for (int r = 0; r < 20; ++r) {
    usleep(2000);
    tsr_parallel_for(0, 1000, count, NULL);
    check(0, 1000, 1);
  }
  tsr_parallel_for(0, 16, nested, NULL);
  check(0, 128, 1);
  /* Two threads starting loops at once: one owns the pool, the other runs serially. */
  pthread_t a, b;
  pthread_create(&a, NULL, caller, NULL);
  pthread_create(&b, NULL, caller, NULL);
  pthread_join(a, NULL);
  pthread_join(b, NULL);
  check(0, 64, 1000);
  /* A forked child has no workers and runs loops serially. */
  pid_t pid = fork();
  if (pid == 0) {
    tsr_parallel_for(0, 1000, count, NULL);
    for (int i = 0; i < 1000; ++i) if (hits[i] != 1) _exit(2);
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

  void* p = tsr_alloc(1000);
  assert(((uintptr_t)p & (TSR_ALIGNMENT - 1)) == 0);
  tsr_free(p);
  double t0 = tsr_now();
  usleep(10000);
  double dt = tsr_now() - t0;
  assert(dt >= 0.009 && dt < 1.0);

  assert(tsr_floordiv(-7, 2) == -4 && tsr_floormod(-7, 2) == 1);
  assert(tsr_floordiv(7, -2) == -4 && tsr_floormod(7, -2) == -1);
  assert(tsr_floordiv(-8, 2) == -4 && tsr_floormod(-8, 2) == 0);
  printf("ok\n");
  return 0;
}
"""


def test_thread_pool() -> None:
    files = {"pool_test.c": POOL_TEST, "tessera_runtime.h": header_text("tessera_runtime.h")}
    binary = cbuild.build(files, "pool_test.c", Target(), kind="executable")
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n"


@pytest.mark.parametrize("threads", [1, 3])
def test_runtime_compiles_for_any_thread_count(threads: int) -> None:
    source = (
        f"#define TSR_NUM_THREADS {threads}\n"
        '#include "tessera_runtime.h"\n'
        "static void f(void* a, int64_t b, int64_t e) { (void)a; (void)b; (void)e; }\n"
        "int main(void) { tsr_parallel_for(0, 10, f, NULL); return 0; }\n"
    )
    files = {"t.c": source, "tessera_runtime.h": header_text("tessera_runtime.h")}
    binary = cbuild.build(files, "t.c", Target(), kind="executable")
    assert subprocess.run([str(binary)], timeout=60).returncode == 0
