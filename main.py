# ==============================
# CPU Scheduling Visualizer
# Phase 1: Pygame loop + auto-tick
# ==============================

from dataclasses import dataclass
from typing import List, Optional

import pygame

# ------------------------------
# CONFIG
# ------------------------------
W, H = 1100, 650
FPS = 60
TICK_MS_DEFAULT = 500  # 0.5s per time unit


# ------------------------------
# Process definition
# ------------------------------
@dataclass
class Process:
    pid: str
    arrival_time: int
    burst_time: int

    remaining_time: int = 0
    start_time: Optional[int] = None
    completion_time: Optional[int] = None

    def __post_init__(self):
        self.remaining_time = self.burst_time


# ------------------------------
# FCFS Scheduler
# ------------------------------
class FCFSScheduler:
    def __init__(self, processes: List[Process]):
        self.processes = processes
        self.reset()

    def reset(self):
        self.time = 0
        self.ready_queue: List[Process] = []
        self.running: Optional[Process] = None
        self.completed: List[Process] = []
        self.gantt_chart: List[str] = []

        # Keep a stable list we can scan for arrivals
        self._all = self.processes

    def add_arrived_processes(self):
        for p in self._all:
            if p.arrival_time == self.time:
                self.ready_queue.append(p)

    def schedule(self):
        # If CPU is idle, pick next process
        if self.running is None and self.ready_queue:
            self.running = self.ready_queue.pop(0)
            if self.running.start_time is None:
                self.running.start_time = self.time

    def execute(self):
        if self.running:
            self.running.remaining_time -= 1
            self.gantt_chart.append(self.running.pid)

            if self.running.remaining_time == 0:
                self.running.completion_time = self.time + 1
                self.completed.append(self.running)
                self.running = None
        else:
            self.gantt_chart.append("IDLE")

    def tick(self):
        self.add_arrived_processes()
        self.schedule()
        self.execute()
        self.time += 1

    def done(self) -> bool:
        return len(self.completed) == len(self.processes)


def build_default_processes() -> List[Process]:
    return [
        Process("P1", arrival_time=0, burst_time=5),
        Process("P2", arrival_time=1, burst_time=3),
        Process("P3", arrival_time=2, burst_time=6),
        Process("P4", arrival_time=4, burst_time=2),
    ]


# ------------------------------
# Pygame UI (Phase 1: text only)
# ------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CPU Scheduling Visualizer (FCFS) - Phase 1")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 26)
    small = pygame.font.SysFont("Arial", 20)

    scheduler = FCFSScheduler(build_default_processes())

    paused = False
    tick_ms = TICK_MS_DEFAULT
    last_tick = pygame.time.get_ticks()

    running = True
    while running:
        clock.tick(FPS)
        now = pygame.time.get_ticks()

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    scheduler = FCFSScheduler(build_default_processes())
                elif event.key == pygame.K_UP:
                    tick_ms = min(1500, tick_ms + 100)  # slower
                elif event.key == pygame.K_DOWN:
                    tick_ms = max(100, tick_ms - 100)   # faster

        # Auto tick
        if (not paused) and (not scheduler.done()) and (now - last_tick >= tick_ms):
            scheduler.tick()
            last_tick = now

        # Draw
        screen.fill((18, 18, 20))

        header = [
            "CPU Scheduling Visualizer (FCFS) - Phase 1",
            f"Time: {scheduler.time} | Completed: {len(scheduler.completed)}/{len(scheduler.processes)} | Tick: {tick_ms}ms",
            "Controls: SPACE Pause/Resume | R Reset | UP Slow | DOWN Fast",
        ]
        y = 18
        for ln in header:
            surf = small.render(ln, True, (235, 235, 235))
            screen.blit(surf, (18, y))
            y += 26

        # CPU info
        cpu_title = font.render("CPU", True, (240, 240, 240))
        screen.blit(cpu_title, (50, 120))

        if scheduler.running:
            cpu_lines = [
                f"Running: {scheduler.running.pid}",
                f"Remaining: {scheduler.running.remaining_time}/{scheduler.running.burst_time}",
                f"Start time: {scheduler.running.start_time}",
            ]
        else:
            cpu_lines = ["CPU is IDLE"]

        y = 165
        for ln in cpu_lines:
            screen.blit(small.render(ln, True, (220, 220, 220)), (50, y))
            y += 26

        # Ready Queue info
        rq_title = font.render("Ready Queue (front → back)", True, (240, 240, 240))
        screen.blit(rq_title, (450, 120))

        rq = scheduler.ready_queue
        rq_text = "  ".join([f"{p.pid}(rem:{p.remaining_time})" for p in rq]) or "(empty)"
        screen.blit(small.render(rq_text, True, (220, 220, 220)), (450, 165))

        # Gantt text preview (temporary)
        gantt_title = font.render("Gantt (text preview)", True, (240, 240, 240))
        screen.blit(gantt_title, (50, 260))

        preview = scheduler.gantt_chart[-25:]  # show last 25 ticks
        gantt_text = " | ".join(preview) if preview else "(no ticks yet)"
        screen.blit(small.render(gantt_text, True, (220, 220, 220)), (50, 305))

        if paused:
            screen.blit(font.render("PAUSED", True, (255, 255, 255)), (950, 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()