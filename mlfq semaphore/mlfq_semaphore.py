#!/usr/bin/env python3
"""
MLFQ (Multi-Level Feedback Queue) scheduler implemented with SEMAPHORES ONLY.

Design (matches the article "Implementing MLFQ Using Semaphores"):

    mutex[i]     binary semaphore (1)  protects queue Qi
    ready        counting semaphore (0) number of ready processes; the
                                         dispatcher sleeps on it when 0
    cpu          binary semaphore (1)  only one process owns the CPU
    boost_mutex  binary semaphore (1)  keeps the priority boost away from
                                         enqueue / dequeue operations

Lock order (prevents deadlock):  boost_mutex -> mutex[0] -> mutex[1] -> mutex[2]

Time is simulated: 1 "ms" of CPU burst = UNIT seconds of real sleeping.

Run:
    python3 mlfq_semaphore.py
    python3 mlfq_semaphore.py --boost-period 20
"""
import argparse
import threading
import time
from collections import deque

UNIT = 0.02                 # real seconds per simulated millisecond
QUANTA = [4, 8, 16]         # time quantum of Q0, Q1, Q2
N = len(QUANTA)


class Process:
    def __init__(self, name, burst):
        self.name = name
        self.burst = burst
        self.remaining = burst
        self.level = 0              # rule R3: new process enters Q0
        self.state = "NEW"
        self.completion = None


class SemaphoreMLFQ:
    def __init__(self, boost_period):
        self.queues = [deque() for _ in range(N)]
        self.mutex = [threading.Semaphore(1) for _ in range(N)]
        self.ready = threading.Semaphore(0)
        self.cpu = threading.Semaphore(1)
        self.boost_mutex = threading.Semaphore(1)

        self.boost_period = boost_period
        self.stop_evt = threading.Event()
        self.total = 0
        self.finished = 0
        self.boosts = 0
        self.log = []               # (start, end, process name, level)
        self.t0 = 0.0

    # ------------------------------------------------------------ helpers
    def now(self):
        return (time.monotonic() - self.t0) / UNIT

    # -------------------------------------------------------- make_ready
    def make_ready(self, p):
        self.boost_mutex.acquire()
        i = p.level
        self.mutex[i].acquire()
        self.queues[i].append(p)
        p.state = "READY"
        self.mutex[i].release()
        self.boost_mutex.release()
        self.ready.release()        # wake the dispatcher

    # -------------------------------------------------------- dispatcher
    def _pick_next(self):
        """Dequeue the first process of the highest non-empty queue."""
        self.boost_mutex.acquire()
        chosen = None
        for i in range(N):
            self.mutex[i].acquire()
            if self.queues[i]:
                chosen = self.queues[i].popleft()
                self.mutex[i].release()
                break
            self.mutex[i].release()
        self.boost_mutex.release()
        return chosen

    def dispatcher(self):
        while self.finished < self.total:
            self.ready.acquire()                    # sleep if nothing is ready
            p = self._pick_next()
            if p is None:                           # defensive: should not happen
                continue
            p.state = "RUNNING"
            slice_ms = min(QUANTA[p.level], p.remaining)

            self.cpu.acquire()                      # own the CPU
            start = self.now()
            time.sleep(slice_ms * UNIT)             # "run" the process
            end = self.now()
            self.cpu.release()

            p.remaining -= slice_ms
            self.log.append((start, end, p.name, p.level))
            self._update_after_run(p, end)

    def _update_after_run(self, p, end):
        if p.remaining == 0:
            p.state = "TERMINATED"
            p.completion = end
            self.finished += 1
        else:                                       # used its full quantum (R4)
            p.level = min(p.level + 1, N - 1)
            self.make_ready(p)

    # ------------------------------------------------------------- boost
    def boost(self):
        """Rule R5: move every process to Q0.  Locks taken in fixed order."""
        self.boost_mutex.acquire()
        for i in range(N):
            self.mutex[i].acquire()
        for i in range(1, N):
            while self.queues[i]:
                p = self.queues[i].popleft()
                p.level = 0
                self.queues[0].append(p)
        for i in reversed(range(N)):
            self.mutex[i].release()
        self.boost_mutex.release()
        self.boosts += 1

    def _boost_loop(self):
        while not self.stop_evt.wait(self.boost_period * UNIT):
            self.boost()

    # --------------------------------------------------------------- run
    def run(self, processes):
        self.total = len(processes)
        self.t0 = time.monotonic()
        booster = threading.Thread(target=self._boost_loop, daemon=True)
        booster.start()
        for p in processes:
            self.make_ready(p)
        self.dispatcher()
        self.stop_evt.set()
        booster.join()


def report(sched, processes):
    print("\nExecution trace")
    print(f"{'Interval':<10}{'Process':<9}{'Queue':<6}")
    for start, end, name, level in sched.log:
        print(f"{round(start):>3}-{round(end):<6}{name:<9}Q{level}")

    print("\nMetrics")
    print(f"{'Process':<9}{'Completion':>11}{'Turnaround':>12}{'Waiting':>9}")
    tt = wt = 0
    for p in processes:
        comp = round(p.completion)
        turn = comp                                  # all arrive at time 0
        wait = turn - p.burst
        tt += turn
        wt += wait
        print(f"{p.name:<9}{comp:>11}{turn:>12}{wait:>9}")
    n = len(processes)
    print(f"{'Average':<9}{'':>11}{tt / n:>12.1f}{wt / n:>9.1f}")
    print(f"\nPriority boosts performed: {sched.boosts}")


def main():
    ap = argparse.ArgumentParser(description="MLFQ with semaphores")
    ap.add_argument("--boost-period", type=int, default=100,
                    help="priority boost period S in simulated ms (default 100)")
    args = ap.parse_args()

    processes = [Process("A", 20), Process("B", 6), Process("C", 12)]
    sched = SemaphoreMLFQ(args.boost_period)
    sched.run(processes)
    report(sched, processes)


if __name__ == "__main__":
    main()
