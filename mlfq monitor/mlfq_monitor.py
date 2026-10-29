#!/usr/bin/env python3
"""
MLFQ (Multi-Level Feedback Queue) scheduler implemented with a MONITOR.

Python has no `monitor` keyword, so the monitor is a class whose public
methods all run inside `with self._cond:` (implicit mutual exclusion) and
which uses a condition variable for waiting.  Python's threading.Condition
has MESA semantics, therefore every wait is inside a `while` loop.

Design (matches the article "Implementing MLFQ Using Monitors"):

    class MLFQMonitor
        Q[0..N-1]      private ready queues
        ready_count    number of processes waiting in all queues
        nonEmpty       condition variable: dispatcher sleeps while count == 0
        make_ready(P) / get_next() / update_after_run(P) / boost()

The process runs OUTSIDE the monitor, so the lock is held only briefly.

Run:
    python3 mlfq_monitor.py
    python3 mlfq_monitor.py --boost-period 20
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


class MLFQMonitor:
    def __init__(self, total):
        self._cond = threading.Condition()      # monitor lock + condition `nonEmpty`
        self._q = [deque() for _ in range(N)]
        self._ready_count = 0
        self._total = total
        self._finished = 0
        self._closed = False
        self.boosts = 0

    # --- internal (caller must already be inside the monitor) ------------
    def _enqueue(self, p):
        self._q[p.level].append(p)
        p.state = "READY"
        self._ready_count += 1
        self._cond.notify()                     # signal(nonEmpty)

    # --- entry procedures -------------------------------------------------
    def make_ready(self, p):
        with self._cond:
            self._enqueue(p)

    def get_next(self):
        """Block until a process exists; return the highest-priority one.
        Returns None once every process has terminated."""
        with self._cond:
            while self._ready_count == 0 and not self._closed:   # Mesa: use while
                self._cond.wait()                                # releases the lock
            if self._ready_count == 0:
                return None
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
                    self._cond.notify_all()     # wake the dispatcher so it can exit
            else:                               # used the full quantum (R4)
                p.level = min(p.level + 1, N - 1)
                self._enqueue(p)

    def boost(self):
        """Rule R5: move every process to Q0."""
        with self._cond:
            for i in range(1, N):
                while self._q[i]:
                    p = self._q[i].popleft()
                    p.level = 0
                    self._q[0].append(p)
            self.boosts += 1


class Scheduler:
    def __init__(self, processes, boost_period):
        self.processes = processes
        self.monitor = MLFQMonitor(len(processes))
        self.boost_period = boost_period
        self.stop_evt = threading.Event()
        self.log = []
        self.t0 = 0.0

    def now(self):
        return (time.monotonic() - self.t0) / UNIT

    def _dispatcher(self):
        while True:
            p = self.monitor.get_next()         # may sleep inside the monitor
            if p is None:
                break
            slice_ms = min(QUANTA[p.level], p.remaining)
            start = self.now()
            time.sleep(slice_ms * UNIT)         # run OUTSIDE the monitor
            end = self.now()
            p.remaining -= slice_ms
            self.log.append((start, end, p.name, p.level))
            self.monitor.update_after_run(p, end)

    def _timer_loop(self):
        while not self.stop_evt.wait(self.boost_period * UNIT):
            self.monitor.boost()

    def run(self):
        self.t0 = time.monotonic()
        timer = threading.Thread(target=self._timer_loop, daemon=True)
        timer.start()
        for p in self.processes:
            self.monitor.make_ready(p)
        self._dispatcher()
        self.stop_evt.set()
        timer.join()


def report(s):
    print("\nExecution trace")
    print(f"{'Interval':<10}{'Process':<9}{'Queue':<6}")
    for start, end, name, level in s.log:
        print(f"{round(start):>3}-{round(end):<6}{name:<9}Q{level}")

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
    ap = argparse.ArgumentParser(description="MLFQ with a monitor")
    ap.add_argument("--boost-period", type=int, default=100,
                    help="priority boost period S in simulated ms (default 100)")
    args = ap.parse_args()

    processes = [Process("A", 20), Process("B", 6), Process("C", 12)]
    sched = Scheduler(processes, args.boost_period)
    sched.run()
    report(sched)


if __name__ == "__main__":
    main()
