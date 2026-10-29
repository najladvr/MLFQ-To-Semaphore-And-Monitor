#!/usr/bin/env python3
"""
MLFQ (Multi-Level Feedback Queue) scheduler combining a MONITOR and SEMAPHORES.

Division of labour (matches the article "MLFQ by Combining Semaphores and
Monitors"):

    Monitor MLFQMonitor   owns the queues, ready_count and the MLFQ policy.
                          Condition variable `nonEmpty` = workers sleep here.
    cpu_slots             counting semaphore, initial value M (number of
                          cores): at most M processes run at the same time.
    boost_sem             binary-style semaphore, initial value 0: the timer
                          "interrupt handler" only calls release() on it; the
                          booster thread waits on it and calls monitor.boost().

Golden rule: a thread NEVER waits on a semaphore while inside the monitor.
Lock order is always  cpu_slots -> monitor lock  (never the reverse).

Threads:  W workers (W >= M), 1 booster, 1 timer ("interrupt handler").

Run:
    python3 mlfq_hybrid.py
    python3 mlfq_hybrid.py --cores 1 --workers 1
    python3 mlfq_hybrid.py --boost-period 8
"""
import argparse
import threading
import time
from collections import deque

UNIT = 0.02                 # real seconds per simulated millisecond
QUANTA = [4, 8, 16]
N = len(QUANTA)


class Process:
    def __init__(self, name, burst):
        self.name = name
        self.burst = burst
        self.remaining = burst
        self.level = 0
        self.state = "NEW"
        self.completion = None


# ---------------------------------------------------------------- monitor
class MLFQMonitor:
    def __init__(self, total):
        self._cond = threading.Condition()      # monitor lock + `nonEmpty`
        self._q = [deque() for _ in range(N)]
        self._ready_count = 0
        self._total = total
        self._finished = 0
        self._closed = False
        self.boosts = 0

    def _enqueue(self, p):
        self._q[p.level].append(p)
        p.state = "READY"
        self._ready_count += 1
        self._cond.notify()                     # signal(nonEmpty)

    def make_ready(self, p):
        with self._cond:
            self._enqueue(p)

    def get_next(self):
        with self._cond:
            while self._ready_count == 0 and not self._closed:   # Mesa: while
                self._cond.wait()
            if self._ready_count == 0:
                return None                     # everything finished
            for i in range(N):
                if self._q[i]:
                    p = self._q[i].popleft()
                    self._ready_count -= 1
                    p.state = "RUNNING"
                    return p

    def update_after_run(self, p, end):
        with self._cond:
            if p.remaining == 0:
                p.state = "TERMINATED"
                p.completion = end
                self._finished += 1
                if self._finished == self._total:
                    self._closed = True
                    self._cond.notify_all()
            else:
                p.level = min(p.level + 1, N - 1)
                self._enqueue(p)

    def boost(self):
        with self._cond:
            for i in range(1, N):
                while self._q[i]:
                    p = self._q[i].popleft()
                    p.level = 0
                    self._q[0].append(p)
            self.boosts += 1


# -------------------------------------------------------------- scheduler
class HybridScheduler:
    def __init__(self, processes, cores, workers, boost_period):
        self.processes = processes
        self.monitor = MLFQMonitor(len(processes))
        self.cores = cores
        self.workers = max(workers, cores)      # W >= M
        self.boost_period = boost_period

        self.cpu_slots = threading.Semaphore(cores)     # counted resource
        self.boost_sem = threading.Semaphore(0)         # timer -> booster
        self.stop_evt = threading.Event()

        self._free_cores = list(range(1, cores + 1))    # only for logging
        self._core_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.log = []                                   # (start, end, core, name, level)
        self.t0 = 0.0

    def now(self):
        return (time.monotonic() - self.t0) / UNIT

    # --- worker: semaphore -> monitor -> run -> monitor -> semaphore -----
    # The core slot is taken BEFORE asking the monitor for a process, so the
    # process is chosen at the moment a core is really free.  (If the process
    # were dequeued first and the slot acquired afterwards, a worker could sit
    # on a high-priority process while a lower-priority one runs: a violation
    # of rule R1.)  Lock order is always: cpu_slots -> monitor lock; the
    # monitor never waits on a semaphore, so no circular wait is possible.
    def _worker(self):
        while True:
            self.cpu_slots.acquire()            # OUTSIDE the monitor: take a core
            p = self.monitor.get_next()         # may sleep INSIDE the monitor
            if p is None:                       # everything finished
                self.cpu_slots.release()
                break
            slice_ms = min(QUANTA[p.level], p.remaining)

            with self._core_lock:
                core = self._free_cores.pop(0)
            start = self.now()
            time.sleep(slice_ms * UNIT)         # run one quantum
            end = self.now()
            with self._core_lock:
                self._free_cores.append(core)

            p.remaining -= slice_ms
            with self._log_lock:
                self.log.append((start, end, core, p.name, p.level))
            self.monitor.update_after_run(p, end)   # requeue before freeing the core
            self.cpu_slots.release()            # give the core back

    # --- timer "interrupt handler": only signals, never blocks -----------
    def _timer(self):
        while not self.stop_evt.wait(self.boost_period * UNIT):
            self.boost_sem.release()

    # --- booster: sleeps on the semaphore, then calls the monitor --------
    def _booster(self):
        while True:
            self.boost_sem.acquire()
            if self.stop_evt.is_set():
                break
            self.monitor.boost()

    def run(self):
        self.t0 = time.monotonic()
        timer = threading.Thread(target=self._timer, daemon=True)
        booster = threading.Thread(target=self._booster, daemon=True)
        workers = [threading.Thread(target=self._worker) for _ in range(self.workers)]

        timer.start()
        booster.start()
        for p in self.processes:
            self.monitor.make_ready(p)
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        self.stop_evt.set()
        self.boost_sem.release()                # wake the booster so it can exit
        timer.join()
        booster.join()


def report(s):
    print(f"\nCores: {s.cores}   Workers: {s.workers}")
    print("\nExecution trace")
    print(f"{'Interval':<10}{'Core':<6}{'Process':<9}{'Queue':<6}")
    for start, end, core, name, level in sorted(s.log, key=lambda r: (round(r[0]), r[2])):
        print(f"{round(start):>3}-{round(end):<6}{core:<6}{name:<9}Q{level}")

    print("\nMetrics")
    print(f"{'Process':<9}{'Completion':>11}{'Turnaround':>12}{'Waiting':>9}")
    tt = wt = 0
    for p in s.processes:
        comp = round(p.completion)
        turn = comp
        wait = turn - p.burst
        tt += turn
        wt += wait
        print(f"{p.name:<9}{comp:>11}{turn:>12}{wait:>9}")
    n = len(s.processes)
    print(f"{'Average':<9}{'':>11}{tt / n:>12.1f}{wt / n:>9.1f}")
    print(f"\nPriority boosts performed: {s.monitor.boosts}")


def main():
    ap = argparse.ArgumentParser(description="MLFQ with monitor + semaphores")
    ap.add_argument("--cores", type=int, default=2, help="number of cores M (default 2)")
    ap.add_argument("--workers", type=int, default=3, help="worker threads W >= M (default 3)")
    ap.add_argument("--boost-period", type=int, default=100,
                    help="priority boost period S in simulated ms (default 100)")
    args = ap.parse_args()

    processes = [Process("A", 20), Process("B", 6), Process("C", 12)]
    sched = HybridScheduler(processes, args.cores, args.workers, args.boost_period)
    sched.run()
    report(sched)


if __name__ == "__main__":
    main()
