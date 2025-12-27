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
# COLORS
# ------------------------------
BG = (18, 18, 20)
PANEL = (35, 35, 38)
BORDER = (180, 180, 180)
TEXT = (235, 235, 235)
MUTED = (220, 220, 220)
CPU_RUN = (70, 170, 110)
CPU_IDLE = (120, 120, 120)
READY_BOX = (70, 110, 170)
GANTT_BG = (24, 24, 26)
GRID = (70, 70, 75)


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
# Helper drawing functions
# ------------------------------
def draw_panel(screen, rect, title, font, small):
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    t = font.render(title, True, TEXT)
    screen.blit(t, (rect.x + 12, rect.y + 10))

def draw_process_chip(screen, rect, label, color, small):
    pygame.draw.rect(screen, color, rect, border_radius=10)
    pygame.draw.rect(screen, (20, 20, 20), rect, 2, border_radius=10)
    txt = small.render(label, True, (10, 10, 10))
    screen.blit(txt, (rect.x + 10, rect.y + 14))


def pid_color(pid: str):
    # Consistent color for each PID
    if pid == "IDLE":
        return CPU_IDLE
    h = 0
    for ch in pid:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (60 + (h % 160), 60 + ((h // 3) % 160), 60 + ((h // 7) % 160))

def compress_gantt(gantt: List[str]):
    # Convert tick-by-tick list into segments: (pid, start, end)
    segs = []
    if not gantt:
        return segs
    cur = gantt[0]
    start = 0
    for i in range(1, len(gantt)):
        if gantt[i] != cur:
            segs.append((cur, start, i))
            cur = gantt[i]
            start = i
    segs.append((cur, start, len(gantt)))
    return segs

def draw_gantt(screen, rect, gantt: List[str], font, small):
    # Panel
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    title = font.render("Gantt Chart", True, TEXT)
    screen.blit(title, (rect.x + 12, rect.y + 10))

    inner = pygame.Rect(rect.x + 12, rect.y + 52, rect.w - 24, rect.h - 72)
    pygame.draw.rect(screen, GANTT_BG, inner, border_radius=10)

    if not gantt:
        msg = small.render("(no ticks yet)", True, MUTED)
        screen.blit(msg, (inner.x + 10, inner.y + 10))
        return

    segs = compress_gantt(gantt)
    total_t = len(gantt)

    # pixels per time unit (keep readable)
    px_per_t = max(10, min(40, inner.w // max(1, total_t)))

    x0 = inner.x + 10
    y0 = inner.y + 18
    h = 54

    # Draw segments
    for pid, s, e in segs:
        bx = x0 + s * px_per_t
        bw = max(1, (e - s) * px_per_t)
        block = pygame.Rect(bx, y0, bw, h)
        pygame.draw.rect(screen, pid_color(pid), block, border_radius=8)
        pygame.draw.rect(screen, (20, 20, 20), block, 2, border_radius=8)

        # Label if block wide enough
        if bw >= 40:
            label = small.render(pid, True, (10, 10, 10))
            screen.blit(label, (bx + 6, y0 + 16))

    # Time markers
    max_markers = 16
    step = max(1, total_t // max_markers)
    for t in range(0, total_t + 1, step):
        mx = x0 + t * px_per_t
        pygame.draw.line(screen, GRID, (mx, y0 + h + 8), (mx, y0 + h + 22), 2)
        tt = small.render(str(t), True, MUTED)
        screen.blit(tt, (mx - 6, y0 + h + 26))


# ------------------------------
# Pygame UI (Phase 1: text only)
# ------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CPU Scheduling Visualizer (FCFS) - Phase 3")
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
        screen.fill(BG)

        header = [
            "CPU Scheduling Visualizer (FCFS) - Phase 3",
            f"Time: {scheduler.time} | Completed: {len(scheduler.completed)}/{len(scheduler.processes)} | Tick: {tick_ms}ms",
            "Controls: SPACE Pause/Resume | R Reset | UP Slow | DOWN Fast",
        ]
        y = 18
        for ln in header:
            surf = small.render(ln, True, TEXT)
            screen.blit(surf, (18, y))
            y += 26

        # Layout panels
        cpu_panel = pygame.Rect(40, 120, 360, 180)
        rq_panel = pygame.Rect(430, 120, 630, 180)

        # CPU panel
        draw_panel(screen, cpu_panel, "CPU", font, small)
        if scheduler.running:
            chip = pygame.Rect(cpu_panel.x + 20, cpu_panel.y + 70, 220, 56)
            label = f"{scheduler.running.pid}  rem:{scheduler.running.remaining_time}"
            draw_process_chip(screen, chip, label, CPU_RUN, small)
        else:
            chip = pygame.Rect(cpu_panel.x + 20, cpu_panel.y + 70, 220, 56)
            draw_process_chip(screen, chip, "IDLE", CPU_IDLE, small)

        # Ready Queue panel
        draw_panel(screen, rq_panel, "Ready Queue (front → back)", font, small)
        rx, ry = rq_panel.x + 20, rq_panel.y + 70
        for i, p in enumerate(scheduler.ready_queue[:6]):
            chip = pygame.Rect(rx + (i % 3) * 200, ry + (i // 3) * 70, 180, 56)
            label = f"{p.pid}  rem:{p.remaining_time}"
            draw_process_chip(screen, chip, label, READY_BOX, small)

        # Gantt chart panel (visual)
        gantt_panel = pygame.Rect(40, 330, 1020, 260)
        draw_gantt(screen, gantt_panel, scheduler.gantt_chart, font, small)

        if paused:
            screen.blit(font.render("PAUSED", True, (255, 255, 255)), (950, 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()