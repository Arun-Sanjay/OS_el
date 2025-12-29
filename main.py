# ==============================
# CPU Scheduling Visualizer
# Phase 3: Visual UI + Multi-Algorithm Scheduler
# ==============================

from dataclasses import dataclass
from typing import List, Optional
import json
import math

import pygame

# ------------------------------
# CONFIG
# ------------------------------
W, H = 1100, 900
FPS = 60
TICK_MS_DEFAULT = 500  # 0.5s per time unit

# ------------------------------
# COLORS (Neo-dark dashboard)
# ------------------------------
BG = (14, 15, 18)            # app background
PANEL = (26, 28, 34)         # primary surface
PANEL_2 = (32, 35, 42)       # secondary surface (reserved for later)
BORDER = (70, 74, 88)        # subtle border (no bright white)
OUTLINE = (10, 11, 13)       # dark outline for chips/buttons
TEXT = (240, 242, 248)
MUTED = (170, 176, 192)

ACCENT = (92, 145, 255)      # reserved for later polish
GOOD = (80, 200, 140)
BAD = (255, 120, 120)

CPU_RUN = GOOD
CPU_IDLE = (110, 114, 126)
READY_BOX = (92, 145, 255)   # ready chips use accent tone
GANTT_BG = (18, 19, 22)
GRID = (60, 62, 72)

# Depth / polish
SHADOW = (0, 0, 0)
SHADOW_ALPHA = 120
SHADOW_OFFSET = (0, 6)
HILITE = (255, 255, 255)
HILITE_ALPHA = 18
HEADER_STRIP_ALPHA = 170


# ------------------------------
# Process definition
# ------------------------------
@dataclass
class Process:
    pid: str
    arrival_time: int

    # Total CPU time across all CPU bursts (used for metrics)
    burst_time: int

    priority: int = 0          # lower number = higher priority
    queue: str = "USER"        # for MLQ: "SYS" or "USER"
    arrived: bool = False      # internal: has this process been enqueued yet?

    # Burst model
    # cpu_bursts = [cpu1, cpu2, ...]
    # io_bursts  = [io1,  io2,  ...] where io_i occurs after cpu_i
    cpu_bursts: Optional[List[int]] = None
    io_bursts: Optional[List[int]] = None

    # Runtime state
    remaining_time: int = 0          # remaining in current CPU burst
    cpu_index: int = 0              # current CPU burst index
    io_index: int = 0               # current IO burst index
    io_remaining: int = 0           # remaining IO time if in IO

    start_time: Optional[int] = None
    completion_time: Optional[int] = None

    def __post_init__(self):
        # Back-compat: if cpu_bursts not provided, treat burst_time as a single CPU burst
        if self.cpu_bursts is None:
            self.cpu_bursts = [int(self.burst_time)]
        else:
            self.cpu_bursts = [max(1, int(x)) for x in self.cpu_bursts] if self.cpu_bursts else [1]

        if self.io_bursts is None:
            self.io_bursts = []
        else:
            self.io_bursts = [max(1, int(x)) for x in self.io_bursts]

        # Ensure burst_time equals total CPU time
        self.burst_time = int(sum(self.cpu_bursts))

        # Initialize runtime fields
        self.cpu_index = 0
        self.io_index = 0
        self.io_remaining = 0
        self.remaining_time = int(self.cpu_bursts[0]) if self.cpu_bursts else 0


# ------------------------------
# Multi-Algorithm CPU Scheduler
# ------------------------------
class CPUScheduler:
    """
    Supported algorithms:
      - FCFS (non-preemptive)
      - SJF  (non-preemptive)
      - PRIORITY (non-preemptive; lower priority number runs first)
      - RR (Round Robin; time quantum)
      - MLQ (Multilevel Queue):
          SYS queue  = Round Robin (quantum_sys)
          USER queue = Round Robin (quantum_user)
          SYS has strict priority over USER (preempts USER at tick boundary)
    """

    def __init__(self, processes: List[Process], algorithm: str = "FCFS", quantum: int = 2):
        self.processes = processes
        self.algorithm = algorithm
        self.quantum = quantum          # RR quantum (time units)
        self.quantum_sys = 2            # MLQ SYS quantum (time units)
        self.quantum_user = 4           # MLQ USER quantum (time units)
        self.preemptive_priority = True
        self.reset()

    def reset(self):
        self.time = 0

        # Single-queue algorithms
        self.ready_queue: List[Process] = []

        # MLQ queues
        self.sys_queue: List[Process] = []
        self.user_queue: List[Process] = []

        self.running: Optional[Process] = None
        self.completed: List[Process] = []
        self.gantt_chart: List[str] = []

        # I/O device model (single device): io_active runs, io_queue waits
        self.io_queue: List[Process] = []
        self.io_active: Optional[Process] = None
        self.io_gantt_chart: List[str] = []
        # RR/MLQ time-slice tracking
        self.slice_left: int = 0

        # Stable list for arrival checks
        self._all = self.processes

        # Reset runtime fields on processes
        for p in self._all:
            p.cpu_index = 0
            p.io_index = 0
            p.io_remaining = 0
            p.remaining_time = int(p.cpu_bursts[0]) if p.cpu_bursts else 0

            p.start_time = None
            p.completion_time = None
            p.arrived = False

    def set_algorithm(self, algorithm: str):
        # Switch algorithm and reset simulation for clean demo
        self.algorithm = algorithm
        self.reset()

    def add_arrived_processes(self):
        for p in self._all:
            if (not p.arrived) and p.arrival_time <= self.time:
                if self.algorithm == "MLQ":
                    if p.queue.upper() == "SYS":
                        self.sys_queue.append(p)
                    else:
                        self.user_queue.append(p)
                else:
                    self.ready_queue.append(p)
                p.arrived = True

    def done(self) -> bool:
        return len(self.completed) == len(self.processes)

    # -------- Scheduling helpers --------
    def _dispatch_fcfs(self):
        if self.running is None and self.ready_queue:
            self.running = self.ready_queue.pop(0)
            if self.running.start_time is None:
                self.running.start_time = self.time

    def _dispatch_sjf(self):
        if self.running is None and self.ready_queue:
            idx = min(
                range(len(self.ready_queue)),
                key=lambda i: (
                    self.ready_queue[i].burst_time,
                    self.ready_queue[i].arrival_time,
                    self.ready_queue[i].pid,
                ),
            )
            self.running = self.ready_queue.pop(idx)
            if self.running.start_time is None:
                self.running.start_time = self.time

    def _dispatch_priority(self):
        if self.running is None and self.ready_queue:
            idx = min(
                range(len(self.ready_queue)),
                key=lambda i: (
                    self.ready_queue[i].priority,
                    self.ready_queue[i].arrival_time,
                    self.ready_queue[i].pid,
                ),
            )
            self.running = self.ready_queue.pop(idx)
            if self.running.start_time is None:
                self.running.start_time = self.time
        elif self.running and self.preemptive_priority and self.ready_queue:
            # Preempt if a strictly higher priority process exists in the ready queue.
            best_idx = min(
                range(len(self.ready_queue)),
                key=lambda i: (
                    self.ready_queue[i].priority,
                    self.ready_queue[i].arrival_time,
                    self.ready_queue[i].pid,
                ),
            )
            best = self.ready_queue[best_idx]
            if best.priority < self.running.priority:
                self.ready_queue.pop(best_idx)
                # Put the current running process back into the ready queue (no data loss).
                self.ready_queue.append(self.running)
                self.running = best
                if self.running.start_time is None:
                    self.running.start_time = self.time

    def _dispatch_rr(self):
        if self.running is None and self.ready_queue:
            self.running = self.ready_queue.pop(0)
            if self.running.start_time is None:
                self.running.start_time = self.time
            self.slice_left = self.quantum

    def _dispatch_mlq(self):
        # SYS always wins; preempt USER at tick boundary if SYS becomes non-empty
        if self.running is not None:
            if self.running.queue.upper() != "SYS" and self.sys_queue:
                self.user_queue.insert(0, self.running)
                self.running = None

        if self.running is None:
            if self.sys_queue:
                self.running = self.sys_queue.pop(0)
                if self.running.start_time is None:
                    self.running.start_time = self.time
                self.slice_left = self.quantum_sys
            elif self.user_queue:
                self.running = self.user_queue.pop(0)
                if self.running.start_time is None:
                    self.running.start_time = self.time
                # USER is also time-sliced (more OS-realistic).
                self.slice_left = self.quantum_user

    def schedule(self):
        if self.algorithm == "FCFS":
            self._dispatch_fcfs()
        elif self.algorithm == "SJF":
            self._dispatch_sjf()
        elif self.algorithm == "PRIORITY":
            self._dispatch_priority()
        elif self.algorithm == "RR":
            self._dispatch_rr()
        elif self.algorithm == "MLQ":
            self._dispatch_mlq()
        else:
            self._dispatch_fcfs()
    
    def _tick_io(self):
        # Start IO if device is idle
        if self.io_active is None and self.io_queue:
            self.io_active = self.io_queue.pop(0)

        if self.io_active is not None:
            self.io_active.io_remaining -= 1
            self.io_gantt_chart.append(self.io_active.pid)

            # IO completed -> go back to READY for the next CPU burst
            if self.io_active.io_remaining <= 0:
                p = self.io_active
                self.io_active = None

                if p.cpu_index < len(p.cpu_bursts):
                    p.remaining_time = int(p.cpu_bursts[p.cpu_index])

                if p.completion_time is None:
                    if self.algorithm == "MLQ":
                        if p.queue.upper() == "SYS":
                            self.sys_queue.append(p)
                        else:
                            self.user_queue.append(p)
                    else:
                        self.ready_queue.append(p)
        else:
            self.io_gantt_chart.append("IDLE")

    def execute(self):
        if self.running:
            self.running.remaining_time -= 1
            self.gantt_chart.append(self.running.pid)

            # time-slice tracking
            if self.algorithm == "RR":
                self.slice_left -= 1
            elif self.algorithm == "MLQ":
                self.slice_left -= 1

            # CPU burst finished
            if self.running.remaining_time == 0:
                p = self.running
                self.running = None
                self.slice_left = 0

                # advance CPU burst index
                p.cpu_index += 1

                # If there is an IO burst after this CPU burst -> WAIT
                if p.io_index < len(p.io_bursts):
                    p.io_remaining = int(p.io_bursts[p.io_index])
                    p.io_index += 1
                    self.io_queue.append(p)
                    return

                # If more CPU bursts remain (edge case: CPU bursts without IO) -> READY
                if p.cpu_index < len(p.cpu_bursts):
                    p.remaining_time = int(p.cpu_bursts[p.cpu_index])
                    if self.algorithm == "MLQ":
                        if p.queue.upper() == "SYS":
                            self.sys_queue.append(p)
                        else:
                            self.user_queue.append(p)
                    else:
                        self.ready_queue.append(p)
                    return

                # Otherwise process completed
                p.completion_time = self.time + 1
                self.completed.append(p)
                return

            # RR time slice ended
            if self.algorithm == "RR" and self.slice_left == 0:
                self.ready_queue.append(self.running)
                self.running = None
            elif self.algorithm == "MLQ" and self.running and self.slice_left == 0:
                if self.running.queue.upper() == "SYS":
                    self.sys_queue.append(self.running)
                else:
                    self.user_queue.append(self.running)
                self.running = None
        else:
            self.gantt_chart.append("IDLE")

    def tick(self):
        self.add_arrived_processes()
        self._tick_io()
        self.schedule()
        self.execute()
        self.time += 1


def build_default_processes() -> List[Process]:
    return load_processes_json("processes.json")


# Helper: clone process list (no runtime fields)
def clone_processes(procs: List[Process]) -> List[Process]:
    return [
        Process(
            p.pid,
            p.arrival_time,
            p.burst_time,
            priority=p.priority,
            queue=p.queue,
            cpu_bursts=list(p.cpu_bursts) if p.cpu_bursts is not None else None,
            io_bursts=list(p.io_bursts) if p.io_bursts is not None else None,
        )
        for p in procs
    ]


# ------------------------------
# Dataset loaders: presets + JSON
# ------------------------------
def load_preset(preset_id: int) -> List[Process]:
    if preset_id == 1:
        return [
            Process("P1", arrival_time=0, burst_time=5, priority=2, queue="USER"),
            Process("P2", arrival_time=1, burst_time=3, priority=1, queue="SYS"),
            Process("P3", arrival_time=2, burst_time=6, priority=3, queue="USER"),
            Process("P4", arrival_time=4, burst_time=2, priority=0, queue="SYS"),
        ]

    if preset_id == 2:
        return [
            Process("P1", arrival_time=0, burst_time=3, priority=1, queue="USER"),
            Process("P2", arrival_time=6, burst_time=2, priority=0, queue="SYS"),
            Process("P3", arrival_time=8, burst_time=4, priority=2, queue="USER"),
            Process("P4", arrival_time=12, burst_time=2, priority=1, queue="SYS"),
        ]

    if preset_id == 3:
        return [
            Process("P1", arrival_time=0, burst_time=4, priority=3, queue="USER"),
            Process("P2", arrival_time=1, burst_time=3, priority=0, queue="SYS"),
            Process("P3", arrival_time=2, burst_time=5, priority=2, queue="USER"),
            Process("P4", arrival_time=3, burst_time=2, priority=1, queue="SYS"),
        ]

    if preset_id == 4:
        return [
            Process("P1", arrival_time=0, burst_time=6, priority=1, queue="USER"),
            Process("P2", arrival_time=0, burst_time=5, priority=2, queue="USER"),
            Process("P3", arrival_time=0, burst_time=4, priority=3, queue="USER"),
            Process("P4", arrival_time=0, burst_time=3, priority=0, queue="USER"),
        ]

    if preset_id == 5:
        return [
            Process("S1", arrival_time=0, burst_time=4, priority=0, queue="SYS"),
            Process("U1", arrival_time=0, burst_time=6, priority=3, queue="USER"),
            Process("S2", arrival_time=2, burst_time=3, priority=1, queue="SYS"),
            Process("U2", arrival_time=3, burst_time=4, priority=2, queue="USER"),
        ]

    return load_preset(1)


def load_processes_json(path: str = "processes.json") -> List[Process]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processes: List[Process] = []
    for item in data:
        cpu_bursts = None
        io_bursts = None

        # Optional: bursts = [cpu, io, cpu, io, cpu] (must start with CPU)
        if "bursts" in item and item["bursts"] is not None:
            seq = [max(1, int(x)) for x in item["bursts"]]
            cpu_bursts = seq[0::2]
            io_bursts = seq[1::2]
            bt = int(sum(cpu_bursts)) if cpu_bursts else int(item.get("burst_time", 1))
        else:
            bt = int(item["burst_time"])

        processes.append(
            Process(
                pid=str(item["pid"]),
                arrival_time=int(item["arrival_time"]),
                burst_time=bt,
                priority=int(item.get("priority", 0)),
                queue=str(item.get("queue", "USER")),
                cpu_bursts=cpu_bursts,
                io_bursts=io_bursts,
            )
        )

    return processes


# ------------------------------
# Helper drawing functions
# ------------------------------
def draw_shadow_rect(screen, rect, radius=14, alpha=SHADOW_ALPHA, offset=SHADOW_OFFSET):
    shadow_surf = pygame.Surface((rect.w + 14, rect.h + 14), pygame.SRCALPHA)
    shadow_rect = pygame.Rect(7, 7, rect.w, rect.h)
    pygame.draw.rect(shadow_surf, (*SHADOW, alpha), shadow_rect, border_radius=radius)
    screen.blit(shadow_surf, (rect.x + offset[0] - 7, rect.y + offset[1] - 7))


def draw_inner_highlight(screen, rect, radius=14, alpha=HILITE_ALPHA):
    band = pygame.Surface((rect.w - 6, 26), pygame.SRCALPHA)
    pygame.draw.rect(band, (*HILITE, alpha), pygame.Rect(0, 0, band.get_width(), band.get_height()), border_radius=radius)
    screen.blit(band, (rect.x + 3, rect.y + 3))


def build_background_surface():
    surf = pygame.Surface((W, H))
    top = (
        min(255, BG[0] + 6),
        min(255, BG[1] + 7),
        min(255, BG[2] + 9),
    )
    bottom = (
        max(0, BG[0] - 6),
        max(0, BG[1] - 6),
        max(0, BG[2] - 8),
    )

    grad = pygame.Surface((1, H))
    for y in range(H):
        t = y / max(1, H - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        grad.set_at((0, y), (r, g, b))
    grad = pygame.transform.scale(grad, (W, H))
    surf.blit(grad, (0, 0))

    vignette = pygame.Surface((W, H), pygame.SRCALPHA)
    cx, cy = W / 2, H / 2
    max_dist = math.hypot(cx, cy)
    for y in range(H):
        dy = y - cy
        for x in range(W):
            dx = x - cx
            d = math.hypot(dx, dy) / max_dist
            alpha = int(110 * (d ** 1.8))
            vignette.set_at((x, y), (0, 0, 0, alpha))
    surf.blit(vignette, (0, 0))
    return surf


def draw_header_strip(screen, height):
    """Soft top fade (no hard horizontal line)."""
    strip = pygame.Surface((W, height), pygame.SRCALPHA)
    # Fade from HEADER_STRIP_ALPHA at the top to 0 at the bottom.
    # The exponent eases the fade so it feels natural.
    denom = max(1, height - 1)
    for y in range(height):
        t = y / denom
        a = int(HEADER_STRIP_ALPHA * (1.0 - t) ** 1.6)
        # draw 1px row with alpha
        strip.fill((0, 0, 0, a), rect=pygame.Rect(0, y, W, 1))
    screen.blit(strip, (0, 0))


def draw_panel(screen, rect, title, font, small):
    draw_shadow_rect(screen, rect)
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    draw_inner_highlight(screen, rect)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    t = font.render(title, True, TEXT)
    screen.blit(t, (rect.x + 12, rect.y + 10))


def draw_process_chip(screen, rect, label, color, small, glow_alpha: int = 0):
    draw_shadow_rect(screen, rect, radius=10, alpha=SHADOW_ALPHA - 20, offset=(0, 4))
    pygame.draw.rect(screen, color, rect, border_radius=10)
    pygame.draw.rect(screen, OUTLINE, rect, 2, border_radius=10)
    if glow_alpha > 0:
        glow = pygame.Surface((rect.w + 10, rect.h + 10), pygame.SRCALPHA)
        pygame.draw.rect(
            glow,
            (*ACCENT, glow_alpha),
            pygame.Rect(0, 0, glow.get_width(), glow.get_height()),
            width=3,
            border_radius=12,
        )
        screen.blit(glow, (rect.x - 5, rect.y - 5))
    txt = small.render(label, True, (10, 10, 10))
    screen.blit(txt, (rect.x + 10, rect.y + 14))


# ------------------------------
# Tooltip renderer (reusable)
# ------------------------------
def draw_tooltip(screen, pos, lines, tiny, max_w=460):
    """Simple hover tooltip. `lines` is a list[str]."""
    if not lines:
        return

    pad_x, pad_y = 10, 8
    line_h = tiny.get_height() + 4

    rendered = [tiny.render(str(ln), True, TEXT) for ln in lines]
    w = min(max(s.get_width() for s in rendered) + pad_x * 2, max_w)

    h = len(rendered) * line_h + pad_y * 2

    mx, my = pos
    x = mx + 14
    y = my + 14

    # keep inside window
    if x + w > W - 8:
        x = mx - w - 14
    if y + h > H - 8:
        y = my - h - 14
    x = max(8, min(W - w - 8, x))
    y = max(8, min(H - h - 8, y))

    box = pygame.Rect(x, y, w, h)

    # shadow
    draw_shadow_rect(screen, box, radius=10, alpha=SHADOW_ALPHA - 10, offset=(0, 4))

    # body
    body = pygame.Surface((w, h), pygame.SRCALPHA)
    body.fill((18, 19, 22, 240))
    pygame.draw.rect(body, (*BORDER, 230), pygame.Rect(0, 0, w, h), 2, border_radius=10)

    # subtle top highlight
    top_band = pygame.Surface((w - 6, 18), pygame.SRCALPHA)
    pygame.draw.rect(top_band, (255, 255, 255, 18), pygame.Rect(0, 0, top_band.get_width(), top_band.get_height()), border_radius=10)
    body.blit(top_band, (3, 3))

    screen.blit(body, (x, y))

    ty = y + pad_y
    for surf in rendered:
        screen.blit(surf, (x + pad_x, ty))
        ty += line_h


def pid_color(pid: str):
    if pid == "IDLE":
        return CPU_IDLE
    h = 0
    for ch in pid:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (60 + (h % 160), 60 + ((h // 3) % 160), 60 + ((h // 7) % 160))


def compress_gantt(gantt: List[str]):
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


def draw_gantt(screen, rect, gantt: List[str], font, small, hover_items=None, proc_map=None):
    draw_shadow_rect(screen, rect)
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    draw_inner_highlight(screen, rect)
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

    px_per_t = max(10, min(40, inner.w // max(1, total_t)))

    x0 = inner.x + 10
    y0 = inner.y + 18
    h = 54

    for pid, s, e in segs:
        bx = x0 + s * px_per_t
        bw = max(1, (e - s) * px_per_t)
        block = pygame.Rect(bx, y0, bw, h)
        pygame.draw.rect(screen, pid_color(pid), block, border_radius=8)
        pygame.draw.rect(screen, OUTLINE, block, 2, border_radius=8)

        if bw >= 40:
            label = small.render(pid, True, (10, 10, 10))
            screen.blit(label, (bx + 6, y0 + 16))

        if hover_items is not None:
            if pid == "IDLE":
                hover_items.append((block, ["IDLE", f"Segment: {s} → {e}"]))
            else:
                p = proc_map.get(pid) if proc_map else None
                if p is not None:
                    hover_items.append((
                        block,
                        [
                            f"PID: {p.pid}",
                            f"AT: {p.arrival_time}   BT: {p.burst_time}   PR: {p.priority}",
                            f"Queue: {p.queue}   Rem: {p.remaining_time}",
                            f"Segment: {s} → {e}",
                        ],
                    ))
                else:
                    hover_items.append((block, [f"PID: {pid}", f"Segment: {s} → {e}"]))

    max_markers = 16
    step = max(1, total_t // max_markers)
    for t in range(0, total_t + 1, step):
        mx = x0 + t * px_per_t
        pygame.draw.line(screen, GRID, (mx, y0 + h + 8), (mx, y0 + h + 22), 2)
        tt = small.render(str(t), True, MUTED)
        screen.blit(tt, (mx - 6, y0 + h + 26))


# ------------------------------
# I/O Timeline panel
# ------------------------------
def draw_io_timeline(screen, rect, io_gantt: List[str], font, small, hover_items=None, proc_map=None, ref_len: Optional[int] = None):
    """I/O device timeline (single device). `ref_len` keeps the time scale identical to CPU Gantt."""
    draw_shadow_rect(screen, rect)
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    draw_inner_highlight(screen, rect)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)

    title = font.render("I/O Timeline", True, TEXT)
    screen.blit(title, (rect.x + 12, rect.y + 10))

    inner = pygame.Rect(rect.x + 12, rect.y + 52, rect.w - 24, rect.h - 72)
    pygame.draw.rect(screen, GANTT_BG, inner, border_radius=10)

    if not io_gantt:
        msg = small.render("(no ticks yet)", True, MUTED)
        screen.blit(msg, (inner.x + 10, inner.y + 10))
        return

    # Use CPU gantt length for scaling so labels/blocks line up visually.
    total_t = int(ref_len) if (ref_len is not None and ref_len > 0) else len(io_gantt)
    total_t = max(1, total_t)

    segs = compress_gantt(io_gantt)

    px_per_t = max(10, min(40, inner.w // total_t))

    x0 = inner.x + 10
    h = min(54, max(34, inner.h - 46))
    y0 = inner.y + 12

    for pid, s, e in segs:
        bx = x0 + s * px_per_t
        bw = max(1, (e - s) * px_per_t)
        block = pygame.Rect(bx, y0, bw, h)

        color = CPU_IDLE if pid == "IDLE" else pid_color(pid)
        pygame.draw.rect(screen, color, block, border_radius=8)
        pygame.draw.rect(screen, OUTLINE, block, 2, border_radius=8)

        if bw >= 40:
            label = small.render(pid, True, (10, 10, 10))
            screen.blit(label, (bx + 6, y0 + (h // 2 - label.get_height() // 2)))

        if hover_items is not None:
            if pid == "IDLE":
                hover_items.append((block, ["I/O: IDLE", f"Segment: {s} → {e}"]))
            else:
                p = proc_map.get(pid) if proc_map else None
                if p is not None:
                    hover_items.append((
                        block,
                        [
                            f"PID: {p.pid}",
                            f"AT: {p.arrival_time}   BT: {p.burst_time}   PR: {p.priority}",
                            f"Queue: {p.queue}   IO Rem: {p.io_remaining}",
                            f"Segment: {s} → {e}",
                        ],
                    ))
                else:
                    hover_items.append((block, [f"PID: {pid}", f"Segment: {s} → {e}"]))

    # Markers (aligned to the same scale)
    max_markers = 16
    step = max(1, total_t // max_markers)
    marker_y1 = y0 + h + 6
    marker_y2 = y0 + h + 18
    label_y = y0 + h + 20

    for t in range(0, total_t + 1, step):
        mx = x0 + t * px_per_t
        pygame.draw.line(screen, GRID, (mx, marker_y1), (mx, marker_y2), 2)
        tt = small.render(str(t), True, MUTED)
        screen.blit(tt, (mx - 6, label_y))

# ------------------------------
# Info & Legend panel (Step 1)
# ------------------------------
def draw_info_legend_panel(screen, rect, scheduler: CPUScheduler, font, small):
    draw_shadow_rect(screen, rect)
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    draw_inner_highlight(screen, rect)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    title = font.render("Info & Legend", True, TEXT)
    screen.blit(title, (rect.x + 12, rect.y + 10))

    lx = rect.x + 12
    ly = rect.y + 42

    if scheduler.running:
        p = scheduler.running
        line1 = f"Running: {p.pid} | AT:{p.arrival_time} BT:{p.burst_time} PR:{p.priority} Q:{p.queue} | Rem:{p.remaining_time}"
        if scheduler.algorithm == "RR":
            line2 = f"RR: quantum={scheduler.quantum}  slice_left={scheduler.slice_left}"
        elif scheduler.algorithm == "MLQ":
            if p.queue.upper() == "SYS":
                line2 = (
                    f"MLQ: SYS q={scheduler.quantum_sys} slice_left={scheduler.slice_left} | "
                    f"USER q={scheduler.quantum_user} (SYS dominates USER)"
                )
            else:
                line2 = (
                    f"MLQ: SYS q={scheduler.quantum_sys} (dominates USER) | "
                    f"USER q={scheduler.quantum_user} slice_left={scheduler.slice_left}"
                )
        else:
            line2 = f"Algo: {scheduler.algorithm}"
    else:
        line1 = "Running: IDLE"
        if scheduler.algorithm == "RR":
            line2 = f"RR: quantum={scheduler.quantum}"
        elif scheduler.algorithm == "MLQ":
            line2 = f"MLQ: SYS quantum={scheduler.quantum_sys} (SYS dominates USER)"
        else:
            line2 = f"Algo: {scheduler.algorithm}"

    screen.blit(small.render(line1, True, MUTED), (lx, ly))
    screen.blit(small.render(line2, True, MUTED), (lx, ly + 24))

    pids = sorted({p.pid for p in scheduler.processes})
    pids = ["IDLE"] + pids

    rx = rect.x + rect.w - 420
    ry = rect.y + 42
    box = 18
    gapx = 110
    max_per_row = 3

    for i, pid in enumerate(pids[:9]):
        cx = rx + (i % max_per_row) * gapx
        cy = ry + (i // max_per_row) * 26
        pygame.draw.rect(screen, pid_color(pid), pygame.Rect(cx, cy, box, box), border_radius=4)
        pygame.draw.rect(screen, OUTLINE, pygame.Rect(cx, cy, box, box), 2, border_radius=4)
        screen.blit(small.render(pid, True, MUTED), (cx + box + 8, cy - 2))


# ------------------------------
# Metrics computation and panel
# ------------------------------
def compute_metrics(processes: List[Process]):
    """Return per-process metric rows.

    - Always returns a row for every process (so the table can list all tasks).
    - Averages are computed only across completed processes.
    """
    rows = []
    completed_rows = []

    # Stable ordering for display
    for p in sorted(processes, key=lambda x: (x.arrival_time, x.pid)):
        st = p.start_time
        ct = p.completion_time

        if st is not None and ct is not None:
            tat = ct - p.arrival_time
            wt = tat - p.burst_time
            rt = st - p.arrival_time
            row = {
                "PID": p.pid,
                "AT": p.arrival_time,
                "BT": p.burst_time,
                "PR": p.priority,
                "Q": p.queue,
                "ST": st,
                "CT": ct,
                "TAT": tat,
                "WT": wt,
                "RT": rt,
                "_done": True,
            }
            completed_rows.append(row)
        else:
            # Not finished yet (or not started). Keep placeholders.
            row = {
                "PID": p.pid,
                "AT": p.arrival_time,
                "BT": p.burst_time,
                "PR": p.priority,
                "Q": p.queue,
                "ST": "-" if st is None else st,
                "CT": "-" if ct is None else ct,
                "TAT": "-",
                "WT": "-",
                "RT": "-" if st is None else (st - p.arrival_time),
                "_done": False,
            }

        rows.append(row)

    if completed_rows:
        avg_wt = sum(r["WT"] for r in completed_rows) / len(completed_rows)
        avg_tat = sum(r["TAT"] for r in completed_rows) / len(completed_rows)
        avg_rt = sum(r["RT"] for r in completed_rows) / len(completed_rows)
    else:
        avg_wt = avg_tat = avg_rt = 0.0

    return rows, avg_wt, avg_tat, avg_rt


def draw_metrics_panel(screen, rect, scheduler: CPUScheduler, font, small, tiny, scroll_rows: int = 0):
    draw_shadow_rect(screen, rect)
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    draw_inner_highlight(screen, rect)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    title = font.render("Metrics", True, TEXT)
    screen.blit(title, (rect.x + 12, rect.y + 10))

    rows, avg_wt, avg_tat, avg_rt = compute_metrics(scheduler.processes)

    total = len(scheduler.gantt_chart)
    busy = sum(1 for x in scheduler.gantt_chart if x != "IDLE")
    util = (busy / total * 100.0) if total else 0.0

    summary = f"Avg WT: {avg_wt:.2f}   Avg TAT: {avg_tat:.2f}   Avg RT: {avg_rt:.2f}   CPU Util: {util:.1f}%"
    screen.blit(small.render(summary, True, MUTED), (rect.x + 12, rect.y + 46))

    # Table (use smaller font + tighter line height so many rows fit)
    cols = ["PID", "AT", "BT", "PR", "Q", "ST", "CT", "TAT", "WT", "RT"]
    x = rect.x + 12
    y = rect.y + 76

    col_w = [60, 42, 42, 42, 78, 42, 42, 54, 48, 48]
    for c, w in zip(cols, col_w):
        screen.blit(tiny.render(c, True, TEXT), (x, y))
        x += w

    y += 22

    if not rows:
        screen.blit(tiny.render("(no processes)", True, MUTED), (rect.x + 12, y))
        return 0

    row_h = 20
    max_rows = max(1, (rect.y + rect.h - y - 12) // row_h)

    # Clamp scroll to available rows
    max_scroll = max(0, len(rows) - max_rows)
    scroll_rows = max(0, min(max_scroll, int(scroll_rows)))

    # Render visible window
    visible = rows[scroll_rows: scroll_rows + max_rows]

    for r in visible:
        x = rect.x + 12
        for c, w in zip(cols, col_w):
            screen.blit(tiny.render(str(r[c]), True, MUTED), (x, y))
            x += w
        y += row_h

    # Subtle scroll indicator (only when needed)
    if max_scroll > 0:
        info = tiny.render(f"Rows {scroll_rows + 1}-{min(scroll_rows + max_rows, len(rows))} / {len(rows)}", True, MUTED)
        screen.blit(info, (rect.right - 12 - info.get_width(), rect.y + 46))

    return scroll_rows

# ------------------------------
# Algorithm comparison (end screen)
# ------------------------------

def run_algorithm_once(
    processes: List[Process],
    algorithm: str,
    quantum: int = 2,
    preemptive_priority: bool = True,
    mlq_sys_quantum: int = 2,
    mlq_user_quantum: int = 4,
):
    """Run a full simulation for a given algorithm on a fresh clone of `processes` and return summary metrics."""
    sched = CPUScheduler(clone_processes(processes), algorithm=algorithm, quantum=quantum)
    if algorithm == "PRIORITY":
        sched.preemptive_priority = preemptive_priority
    if algorithm == "MLQ":
        sched.quantum_sys = int(mlq_sys_quantum)
        sched.quantum_user = int(mlq_user_quantum)

    guard = 0
    max_steps = 200000  # safety
    while (not sched.done()) and guard < max_steps:
        sched.tick()
        guard += 1

    rows, avg_wt, avg_tat, avg_rt = compute_metrics(sched.processes)

    total = len(sched.gantt_chart)
    busy = sum(1 for x in sched.gantt_chart if x != "IDLE")
    util = (busy / total * 100.0) if total else 0.0

    makespan = sched.time
    throughput = (len(sched.completed) / makespan) if makespan > 0 else 0.0

    return {
        "algorithm": algorithm,
        "avg_wt": float(avg_wt),
        "avg_tat": float(avg_tat),
        "avg_rt": float(avg_rt),
        "cpu_util": float(util),
        "makespan": int(makespan),
        "throughput": float(throughput),
        "_rows": rows,
    }


def compare_all_algorithms(
    processes: List[Process],
    rr_quantum: int,
    preemptive_priority: bool,
    mlq_sys_quantum: int = 2,
    mlq_user_quantum: int = 4,
):
    """Return a list of result dicts for all supported algorithms."""
    algos = ["FCFS", "SJF", "PRIORITY", "RR", "MLQ"]
    out = []
    for a in algos:
        q = rr_quantum if a == "RR" else 2
        out.append(
            run_algorithm_once(
                processes,
                a,
                quantum=q,
                preemptive_priority=preemptive_priority,
                mlq_sys_quantum=mlq_sys_quantum,
                mlq_user_quantum=mlq_user_quantum,
            )
        )
    return out


def draw_comparison_screen(screen, title_font, font, small, tiny, results, metric_key: str, selected_idx: int):
    """Compare algorithms with a simple bar chart + table, and show per-process metrics for selected algorithm."""
    screen.fill(BG)

    hdr = title_font.render("Algorithm Comparison", True, TEXT)
    screen.blit(hdr, (W // 2 - hdr.get_width() // 2, 26))

    help1 = small.render("1 WT   2 TAT   3 RT   4 CPU Util    |    ←/→ select algo    |    ESC: back", True, MUTED)
    screen.blit(help1, (W // 2 - help1.get_width() // 2, 66))

    panel = pygame.Rect(60, 100, W - 120, H - 160)
    draw_shadow_rect(screen, panel, radius=16)
    pygame.draw.rect(screen, PANEL, panel, border_radius=16)
    draw_inner_highlight(screen, panel, radius=16)
    pygame.draw.rect(screen, BORDER, panel, 2, border_radius=16)

    label_map = {
        "avg_wt": "Average Waiting Time (lower is better)",
        "avg_tat": "Average Turnaround Time (lower is better)",
        "avg_rt": "Average Response Time (lower is better)",
        "cpu_util": "CPU Utilization (higher is better)",
    }
    metric_title = font.render(label_map.get(metric_key, metric_key), True, TEXT)
    screen.blit(metric_title, (panel.x + 22, panel.y + 18))

    chart = pygame.Rect(panel.x + 22, panel.y + 66, panel.w - 44, 260)
    pygame.draw.rect(screen, GANTT_BG, chart, border_radius=12)
    pygame.draw.rect(screen, OUTLINE, chart, 2, border_radius=12)

    if not results:
        msg = small.render("(no results)", True, MUTED)
        screen.blit(msg, (chart.x + 14, chart.y + 14))
        return

    vals = [float(r.get(metric_key, 0.0)) for r in results]
    max_v = max(vals) if vals else 1.0
    max_v = max(max_v, 1e-9)

    n = len(results)
    bar_gap = 16
    bar_w = int((chart.w - bar_gap * (n + 1)) / n)
    bar_w = max(90, min(170, bar_w))
    total_w = n * bar_w + (n + 1) * bar_gap
    x0 = chart.x + (chart.w - total_w) // 2
    y_base = chart.bottom - 46
    max_h = 170

    for i, r in enumerate(results):
        a = r["algorithm"]
        v = float(r.get(metric_key, 0.0))
        h = int((v / max_v) * max_h) if max_v > 0 else 0

        bx = x0 + bar_gap + i * (bar_w + bar_gap)
        by = y_base - h
        bar = pygame.Rect(bx, by, bar_w, h)

        color = pid_color(a)
        pygame.draw.rect(screen, color, bar, border_radius=10)
        pygame.draw.rect(screen, OUTLINE, bar, 2, border_radius=10)

        val_s = tiny.render(f"{v:.2f}" if metric_key != "cpu_util" else f"{v:.1f}%", True, MUTED)
        screen.blit(val_s, (bx + bar_w // 2 - val_s.get_width() // 2, by - 18))

        lab = tiny.render(a, True, MUTED)
        screen.blit(lab, (bx + bar_w // 2 - lab.get_width() // 2, y_base + 12))

    table = pygame.Rect(panel.x + 22, chart.bottom + 18, panel.w - 44, panel.h - (chart.bottom - panel.y) - 40)

    # Split: top = algorithm summary, bottom = per-process metrics for selected algorithm
    split_gap = 12
    summary_h = min(170, max(140, table.h // 2 - 10))
    summary_rect = pygame.Rect(table.x, table.y, table.w, summary_h)
    proc_rect = pygame.Rect(table.x, table.y + summary_h + split_gap, table.w, table.h - summary_h - split_gap)

    # --- Summary table (algorithms) ---
    pygame.draw.rect(screen, GANTT_BG, summary_rect, border_radius=12)
    pygame.draw.rect(screen, OUTLINE, summary_rect, 2, border_radius=12)

    cols = ["Algo", "Avg WT", "Avg TAT", "Avg RT", "CPU Util", "Makespan", "Throughput"]
    col_w = [120, 120, 140, 120, 120, 120, 140]

    tx = summary_rect.x + 14
    ty = summary_rect.y + 12
    for c, w in zip(cols, col_w):
        screen.blit(tiny.render(c, True, TEXT), (tx, ty))
        tx += w

    ty += 24
    row_h = 20
    for i, r in enumerate(results):
        # Highlight selected algorithm
        if i == selected_idx:
            hi = pygame.Rect(summary_rect.x + 8, ty - 2, summary_rect.w - 16, row_h)
            pygame.draw.rect(screen, (255, 255, 255, 18), hi, border_radius=8)

        tx = summary_rect.x + 14
        row = [
            r["algorithm"],
            f"{r['avg_wt']:.2f}",
            f"{r['avg_tat']:.2f}",
            f"{r['avg_rt']:.2f}",
            f"{r['cpu_util']:.1f}%",
            str(r["makespan"]),
            f"{r['throughput']:.3f}",
        ]
        for v, w in zip(row, col_w):
            screen.blit(tiny.render(v, True, MUTED), (tx, ty))
            tx += w
        ty += row_h

    # --- Per-process metrics (selected algorithm) ---
    pygame.draw.rect(screen, GANTT_BG, proc_rect, border_radius=12)
    pygame.draw.rect(screen, OUTLINE, proc_rect, 2, border_radius=12)

    if results:
        selected_idx = max(0, min(selected_idx, len(results) - 1))
        sel = results[selected_idx]
        title2 = small.render(f"Per-Process Metrics: {sel['algorithm']}", True, TEXT)
        screen.blit(title2, (proc_rect.x + 14, proc_rect.y + 10))

        rows = sel.get("_rows", [])
        cols2 = ["PID", "AT", "BT", "PR", "Q", "ST", "CT", "TAT", "WT", "RT"]
        col_w2 = [60, 42, 42, 42, 78, 42, 42, 54, 48, 48]

        tx = proc_rect.x + 14
        ty = proc_rect.y + 40
        for c, w in zip(cols2, col_w2):
            screen.blit(tiny.render(c, True, TEXT), (tx, ty))
            tx += w

        ty += 22
        # Fit rows
        row_h2 = 20
        max_rows = max(1, (proc_rect.bottom - ty - 10) // row_h2)
        shown = rows[:max_rows]

        for r in shown:
            tx = proc_rect.x + 14
            for c, w in zip(cols2, col_w2):
                screen.blit(tiny.render(str(r.get(c, "-")), True, MUTED), (tx, ty))
                tx += w
            ty += row_h2

        if len(rows) > max_rows:
            more = tiny.render(f"(+{len(rows) - max_rows} more rows)", True, MUTED)
            screen.blit(more, (proc_rect.right - 14 - more.get_width(), proc_rect.y + 10))
    else:
        msg = small.render("(no results)", True, MUTED)
        screen.blit(msg, (proc_rect.x + 14, proc_rect.y + 10))


# ------------------------------
# Start screen
# ------------------------------
def draw_start_screen(screen, title_font, font, small, tiny, state):
    if state.get("background") is not None:
        screen.blit(state["background"], (0, 0))
        draw_header_strip(screen, 150)
    else:
        screen.fill(BG)
        draw_header_strip(screen, 150)

    title = title_font.render("CPU Scheduling Visualizer", True, TEXT)
    screen.blit(title, (W // 2 - title.get_width() // 2, 80))

    sub = small.render("Start setup (choose defaults, then press ENTER)", True, MUTED)
    screen.blit(sub, (W // 2 - sub.get_width() // 2, 120))

    panel = pygame.Rect(260, 190, 580, 420)
    draw_shadow_rect(screen, panel, radius=16)
    pygame.draw.rect(screen, PANEL, panel, border_radius=16)
    draw_inner_highlight(screen, panel, radius=16)
    pygame.draw.rect(screen, BORDER, panel, 2, border_radius=16)

    # Rows
    y = panel.y + 40
    row_gap = 62

    algo = state["algorithms"][state["algo_idx"]]
    algo_line = small.render(f"Algorithm:  {algo}", True, TEXT)
    screen.blit(algo_line, (panel.x + 30, y))
    hint = tiny.render("Use ↑/↓ to change", True, MUTED)
    screen.blit(hint, (panel.right - 30 - hint.get_width(), y + 2))
    y += row_gap

    tick_line = small.render(f"Default Tick Time:  {state['tick_ms']} ms", True, TEXT)
    screen.blit(tick_line, (panel.x + 30, y))
    hint = tiny.render("Use ←/→ to adjust", True, MUTED)
    screen.blit(hint, (panel.right - 30 - hint.get_width(), y + 2))
    y += row_gap

    if algo == "RR":
        q_line = small.render(f"Quantum (RR):  {state['quantum']}", True, TEXT)
        screen.blit(q_line, (panel.x + 30, y))
        hint = tiny.render("Use A/D to adjust", True, MUTED)
        screen.blit(hint, (panel.right - 30 - hint.get_width(), y + 2))
        y += row_gap
    else:
        q_line = small.render("Quantum (RR):  (not applicable)", True, MUTED)
        screen.blit(q_line, (panel.x + 30, y))
        y += row_gap

    # Footer layout (fixed positions so nothing overlaps)
    btn = pygame.Rect(panel.x + 190, panel.bottom - 72, 200, 50)

    note1 = "Note: After starting, you can still change tick speed (UP/DOWN)"
    note2 = "and quantum (←/→) while running."

    note_y = btn.y - 54
    keys_y = note_y - 70

    # Key hints
    screen.blit(small.render("ENTER: Start simulation", True, MUTED), (panel.x + 30, keys_y))
    screen.blit(small.render("ESC: Quit", True, MUTED), (panel.x + 30, keys_y + 30))

    # Note (two lines)
    screen.blit(tiny.render(note1, True, MUTED), (panel.x + 30, note_y))
    screen.blit(tiny.render(note2, True, MUTED), (panel.x + 30, note_y + 20))

    # Start button (clickable)
    draw_shadow_rect(screen, btn, radius=12, alpha=SHADOW_ALPHA - 15, offset=(0, 4))
    pygame.draw.rect(screen, (60, 130, 220), btn, border_radius=12)
    pygame.draw.rect(screen, OUTLINE, btn, 2, border_radius=12)
    btxt = font.render("START", True, (10, 10, 10))
    screen.blit(btxt, (btn.x + btn.w // 2 - btxt.get_width() // 2, btn.y + 14))

    state["start_button"] = btn

# ------------------------------
# Add-process modal
# ------------------------------
def draw_add_modal(screen, font, small, fields, active_idx, status):
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    box = pygame.Rect(160, 180, 780, 460)
    draw_shadow_rect(screen, box, radius=16, alpha=SHADOW_ALPHA + 20)
    pygame.draw.rect(screen, PANEL, box, border_radius=16)
    draw_inner_highlight(screen, box, radius=16)
    pygame.draw.rect(screen, BORDER, box, 3, border_radius=16)

    title = font.render("Add Process (Enter to save, Esc to cancel)", True, TEXT)
    screen.blit(title, (box.x + 24, box.y + 18))

    instructions = [
        "Tab/Shift+Tab to change fields | Queue: USER or SYS",
        "Arrival time defaults to current clock if you enter something older.",
    ]
    for i, ln in enumerate(instructions):
        screen.blit(small.render(ln, True, MUTED), (box.x + 24, box.y + 60 + i * 24))

    start_y = box.y + 120
    for idx, f in enumerate(fields):
        # Split long Queue label onto two lines to avoid truncation
        label_y = start_y + idx * 60
        if idx == 4 or f["label"].startswith("Queue"):
            screen.blit(small.render("Queue", True, TEXT), (box.x + 24, label_y))
            screen.blit(small.render("(USER/SYS)", True, TEXT), (box.x + 24, label_y + 22))
        else:
            label = small.render(f["label"], True, TEXT)
            screen.blit(label, (box.x + 24, label_y))

        field_rect = pygame.Rect(box.x + 180, start_y + idx * 60 - 8, 500, 42)
        pygame.draw.rect(screen, (50, 50, 55), field_rect, border_radius=10)
        pygame.draw.rect(
            screen,
            (120, 190, 255) if idx == active_idx else BORDER,
            field_rect,
            2,
            border_radius=10,
        )
        val = small.render(f["value"], True, TEXT)
        screen.blit(val, (field_rect.x + 12, field_rect.y + 10))

    screen.blit(small.render(status, True, MUTED), (box.x + 24, box.y + box.h - 40))


# ------------------------------
# Pygame UI
# ------------------------------
def main():
    pygame.init()

    # Robust fullscreen: draw to a fixed logical surface (W,H) and scale to the window.
    fullscreen = False
    base_size = (W, H)

    # `window` is the actual display surface; `screen` is the logical render surface.
    window = None
    screen = pygame.Surface(base_size)

    # Scaling / letterbox
    scale = 1.0
    scaled_size = base_size
    offset = (0, 0)

    def _recompute_scale():
        nonlocal scale, scaled_size, offset
        ww, wh = window.get_size()
        sx = ww / base_size[0]
        sy = wh / base_size[1]
        scale = min(sx, sy)
        sw = max(1, int(base_size[0] * scale))
        sh = max(1, int(base_size[1] * scale))
        scaled_size = (sw, sh)
        offset = ((ww - sw) // 2, (wh - sh) // 2)

    def to_logical(pos):
        """Map window mouse coordinates -> logical (W,H) coordinates."""
        mx, my = pos
        ox, oy = offset
        # remove letterbox offset
        mx -= ox
        my -= oy
        if scaled_size[0] <= 0 or scaled_size[1] <= 0:
            return (0, 0)
        lx = int(mx * base_size[0] / scaled_size[0])
        ly = int(my * base_size[1] / scaled_size[1])
        # clamp
        lx = max(0, min(base_size[0] - 1, lx))
        ly = max(0, min(base_size[1] - 1, ly))
        return (lx, ly)

    def set_display(full: bool):
        nonlocal window, fullscreen
        fullscreen = full
        if fullscreen:
            # (0,0) picks the desktop resolution.
            window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            window = pygame.display.set_mode(base_size)
        pygame.display.set_caption("CPU Scheduling Visualizer - Phase 3")
        _recompute_scale()

    def present():
        """Scale logical surface to window and present."""
        window.fill(BG)
        if scaled_size == base_size and offset == (0, 0):
            window.blit(screen, (0, 0))
        else:
            frame = pygame.transform.smoothscale(screen, scaled_size)
            window.blit(frame, offset)
        pygame.display.flip()

    # Start windowed
    set_display(False)
    # Now that a video mode exists, convert the logical surface for faster blits.
    screen = screen.convert()

    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("Arial", 32, bold=True)
    font = pygame.font.SysFont("Arial", 24, bold=True)
    small = pygame.font.SysFont("Arial", 18)
    tiny = pygame.font.SysFont("Arial", 15)

    background = build_background_surface()

    base_dataset = build_default_processes()  # last loaded preset/JSON; does not include live additions

    # App mode: "menu" (start screen) or "sim" (running simulator)
    app_mode = "menu"

    start_state = {
        "algorithms": ["FCFS", "SJF", "PRIORITY", "RR", "MLQ"],
        "algo_idx": 0,            # default FCFS
        "tick_ms": TICK_MS_DEFAULT,
        "quantum": 2,
        "start_button": None,
        "background": background,
    }

    # Scheduler will be created when we press START
    scheduler = None
    status_msg = "Ready"
    live_counter = 1

    paused = False
    tick_ms = TICK_MS_DEFAULT
    last_tick = pygame.time.get_ticks()

    adding_process = False
    add_fields = []
    active_field = 0
    add_status = ""
    live_added: List[Process] = []  # additions for current run only

    # Comparison screen state
    compare_results = None
    compare_metric = "avg_tat"
    compare_shown_once = False
    compare_algo_idx = 0

    def reset_scheduler():
        nonlocal scheduler, status_msg, live_added, compare_results, compare_metric, compare_shown_once
        if scheduler is None:
            return
        prev_preempt = scheduler.preemptive_priority
        scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
        scheduler.preemptive_priority = prev_preempt
        live_added = []
        compare_results = None
        compare_metric = "avg_tat"
        compare_shown_once = False
        status_msg = "Reset to base dataset (cleared live additions)"

    def add_live_process(pid, arrival_val, burst_val, pr_val, queue_val):
        nonlocal live_counter, status_msg, live_added
        base_proc = Process(
            pid,
            arrival_time=arrival_val,
            burst_time=burst_val,
            priority=pr_val,
            queue=queue_val,
        )

        live_proc = Process(
            pid,
            arrival_time=arrival_val,
            burst_time=burst_val,
            priority=pr_val,
            queue=queue_val,
        )

        # NOTE: scheduler._all is usually the same list object as scheduler.processes
        # (see CPUScheduler.reset(): self._all = self.processes). Appending to both
        # would duplicate the process and make it show up twice in Metrics.
        scheduler.processes.append(live_proc)
        if scheduler._all is not scheduler.processes:
            scheduler._all.append(live_proc)

        if arrival_val <= scheduler.time:
            live_proc.arrived = True
            if scheduler.algorithm == "MLQ":
                if queue_val == "SYS":
                    scheduler.sys_queue.append(live_proc)
                else:
                    scheduler.user_queue.append(live_proc)
            else:
                scheduler.ready_queue.append(live_proc)

        live_added.append(base_proc)
        live_counter += 1
        status_msg = f"Added {pid} (AT={arrival_val}, BT={burst_val}, PR={pr_val}, Q={queue_val})"

    running = True
    while running:
        clock.tick(FPS)
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE and (not fullscreen):
                # If windowed resize ever occurs, recompute scaling.
                _recompute_scale()

            # Start screen controls
            if app_mode == "menu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        set_display(not fullscreen)
                    elif event.key == pygame.K_ESCAPE:
                        # If in fullscreen, ESC exits fullscreen first; otherwise quit.
                        if fullscreen:
                            set_display(False)
                        else:
                            running = False
                    elif event.key == pygame.K_UP:
                        start_state["algo_idx"] = (start_state["algo_idx"] - 1) % len(start_state["algorithms"])
                    elif event.key == pygame.K_DOWN:
                        start_state["algo_idx"] = (start_state["algo_idx"] + 1) % len(start_state["algorithms"])
                    elif event.key == pygame.K_LEFT:
                        start_state["tick_ms"] = max(100, start_state["tick_ms"] - 100)
                    elif event.key == pygame.K_RIGHT:
                        start_state["tick_ms"] = min(1500, start_state["tick_ms"] + 100)
                    elif event.key == pygame.K_a:
                        # adjust quantum only if RR is selected
                        if start_state["algorithms"][start_state["algo_idx"]] == "RR":
                            start_state["quantum"] = max(1, start_state["quantum"] - 1)
                    elif event.key == pygame.K_d:
                        if start_state["algorithms"][start_state["algo_idx"]] == "RR":
                            start_state["quantum"] = min(10, start_state["quantum"] + 1)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        # Create scheduler and enter sim mode
                        algo = start_state["algorithms"][start_state["algo_idx"]]
                        scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=algo, quantum=start_state["quantum"])
                        tick_ms = start_state["tick_ms"]
                        paused = False
                        last_tick = pygame.time.get_ticks()
                        status_msg = "Started"
                        app_mode = "sim"

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    lx, ly = to_logical(event.pos)
                    if start_state.get("start_button") and start_state["start_button"].collidepoint((lx, ly)):
                        algo = start_state["algorithms"][start_state["algo_idx"]]
                        scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=algo, quantum=start_state["quantum"])
                        tick_ms = start_state["tick_ms"]
                        paused = False
                        last_tick = pygame.time.get_ticks()
                        status_msg = "Started"
                        app_mode = "sim"
                continue

            # Comparison screen controls
            if app_mode == "compare":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        set_display(not fullscreen)
                    elif event.key == pygame.K_ESCAPE:
                        app_mode = "sim"
                        status_msg = "Back to simulation"
                    elif event.key == pygame.K_1:
                        compare_metric = "avg_wt"
                    elif event.key == pygame.K_2:
                        compare_metric = "avg_tat"
                    elif event.key == pygame.K_3:
                        compare_metric = "avg_rt"
                    elif event.key == pygame.K_4:
                        compare_metric = "cpu_util"
                    elif event.key == pygame.K_LEFT:
                        if compare_results:
                            compare_algo_idx = (compare_algo_idx - 1) % len(compare_results)
                    elif event.key == pygame.K_RIGHT:
                        if compare_results:
                            compare_algo_idx = (compare_algo_idx + 1) % len(compare_results)
                continue

            if event.type == pygame.KEYDOWN and adding_process and app_mode == "sim":
                if event.key == pygame.K_ESCAPE:
                    adding_process = False
                    status_msg = "Add canceled"
                elif event.key in (pygame.K_TAB,):
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        active_field = (active_field - 1) % len(add_fields)
                    else:
                        active_field = (active_field + 1) % len(add_fields)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    try:
                        pid_val = add_fields[0]["value"].strip() or f"X{live_counter}"
                        arrival_val = int(add_fields[1]["value"])
                        burst_val = max(1, int(add_fields[2]["value"]))
                        pr_val = int(add_fields[3]["value"])
                        queue_val = add_fields[4]["value"].strip().upper() or "USER"
                        if queue_val not in ("SYS", "USER"):
                            queue_val = "USER"

                        if arrival_val < scheduler.time:
                            arrival_val = scheduler.time

                        add_live_process(pid_val, arrival_val, burst_val, pr_val, queue_val)
                        adding_process = False
                    except ValueError:
                        add_status = "Invalid number in one of the fields"
                elif event.key == pygame.K_BACKSPACE:
                    val = add_fields[active_field]["value"]
                    add_fields[active_field]["value"] = val[:-1]
                else:
                    ch = event.unicode
                    if ch.isprintable():
                        add_fields[active_field]["value"] += ch

                continue


            if event.type == pygame.KEYDOWN and app_mode == "sim":
                if event.key == pygame.K_f:
                    set_display(not fullscreen)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_scheduler()
                elif event.key == pygame.K_UP:
                    tick_ms = min(1500, tick_ms + 100)
                elif event.key == pygame.K_DOWN:
                    tick_ms = max(100, tick_ms - 100)
                elif event.key == pygame.K_1:
                    scheduler.set_algorithm("FCFS")
                    compare_shown_once = False
                elif event.key == pygame.K_2:
                    scheduler.set_algorithm("SJF")
                    compare_shown_once = False
                elif event.key == pygame.K_3:
                    scheduler.set_algorithm("PRIORITY")
                    compare_shown_once = False
                elif event.key == pygame.K_4:
                    scheduler.set_algorithm("RR")
                    compare_shown_once = False
                elif event.key == pygame.K_5:
                    scheduler.set_algorithm("MLQ")
                    compare_shown_once = False
                elif event.key == pygame.K_LEFT:
                    if scheduler.algorithm == "MLQ":
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            scheduler.quantum_user = max(1, scheduler.quantum_user - 1)
                            status_msg = f"MLQ USER quantum={scheduler.quantum_user}"
                        else:
                            scheduler.quantum_sys = max(1, scheduler.quantum_sys - 1)
                            status_msg = f"MLQ SYS quantum={scheduler.quantum_sys}"
                    else:
                        scheduler.quantum = max(1, scheduler.quantum - 1)
                        status_msg = f"RR quantum={scheduler.quantum}"
                elif event.key == pygame.K_RIGHT:
                    if scheduler.algorithm == "MLQ":
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            scheduler.quantum_user = min(10, scheduler.quantum_user + 1)
                            status_msg = f"MLQ USER quantum={scheduler.quantum_user}"
                        else:
                            scheduler.quantum_sys = min(10, scheduler.quantum_sys + 1)
                            status_msg = f"MLQ SYS quantum={scheduler.quantum_sys}"
                    else:
                        scheduler.quantum = min(10, scheduler.quantum + 1)
                        status_msg = f"RR quantum={scheduler.quantum}"
                elif event.key == pygame.K_p:
                    if scheduler.algorithm == "PRIORITY":
                        scheduler.preemptive_priority = not scheduler.preemptive_priority
                        state = "ON" if scheduler.preemptive_priority else "OFF"
                        status_msg = f"Priority preemption {state}"
                    else:
                        status_msg = "Preemption toggle only valid in PRIORITY mode"
                elif event.key == pygame.K_F1:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(1)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F1 (live additions cleared)"
                    live_counter = 1
                    compare_shown_once = False
                elif event.key == pygame.K_F2:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(2)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F2 (idle gaps, live additions cleared)"
                    live_counter = 1
                    compare_shown_once = False
                elif event.key == pygame.K_F3:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(3)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F3 (priority, live additions cleared)"
                    live_counter = 1
                    compare_shown_once = False
                elif event.key == pygame.K_F4:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(4)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F4 (RR, live additions cleared)"
                    live_counter = 1
                    compare_shown_once = False
                elif event.key == pygame.K_F5:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(5)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F5 (MLQ, live additions cleared)"
                    live_counter = 1
                    compare_shown_once = False
                elif event.key == pygame.K_l:
                    try:
                        prev_preempt = scheduler.preemptive_priority
                        base_dataset = load_processes_json("processes.json")
                        scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                        scheduler.preemptive_priority = prev_preempt
                        live_added = []
                        status_msg = "Loaded processes.json (live additions cleared)"
                        live_counter = 1
                        compare_shown_once = False
                    except Exception:
                        status_msg = "Load failed"
                elif event.key == pygame.K_c:
                    paused = True
                    dataset_for_compare = clone_processes(base_dataset + live_added)
                    compare_results = compare_all_algorithms(
                        dataset_for_compare,
                        rr_quantum=scheduler.quantum,
                        preemptive_priority=scheduler.preemptive_priority,
                        mlq_sys_quantum=scheduler.quantum_sys,
                        mlq_user_quantum=scheduler.quantum_user,
                    )
                    compare_metric = "avg_tat"
                    compare_algo_idx = 0
                    app_mode = "compare"
                    status_msg = "Showing comparison (ESC to return)"
                elif event.key == pygame.K_a:
                    adding_process = True
                    add_status = ""
                    add_fields = [
                        {"label": "PID", "value": f"X{live_counter}"},
                        {"label": "Arrival Time", "value": str(scheduler.time)},
                        {"label": "Burst Time", "value": "3"},
                        {"label": "Priority", "value": "0"},
                        {"label": "Queue (USER/SYS)", "value": "USER" if not (pygame.key.get_mods() & pygame.KMOD_SHIFT) else "SYS"},
                    ]
                    active_field = 0
                    status_msg = "Adding new process…"

        if app_mode == "sim" and scheduler is not None:
            if (not paused) and (not scheduler.done()) and (now - last_tick >= tick_ms):
                scheduler.tick()
                last_tick = now

            # Auto-open comparison once at completion
            if scheduler.done() and (not compare_shown_once):
                paused = True
                dataset_for_compare = clone_processes(base_dataset + live_added)
                compare_results = compare_all_algorithms(
                    dataset_for_compare,
                    rr_quantum=scheduler.quantum,
                    preemptive_priority=scheduler.preemptive_priority,
                    mlq_sys_quantum=scheduler.quantum_sys,
                    mlq_user_quantum=scheduler.quantum_user,
                )
                compare_metric = "avg_tat"
                compare_algo_idx = 0
                app_mode = "compare"
                compare_shown_once = True
                status_msg = "Completed — showing comparison (ESC to return)"

        if app_mode == "menu":
            draw_start_screen(screen, title_font, font, small, tiny, start_state)
            present()
            continue

        if app_mode == "compare":
            if start_state.get("background") is not None:
                screen.blit(start_state["background"], (0, 0))
                draw_header_strip(screen, 150)
            else:
                screen.fill(BG)
                draw_header_strip(screen, 150)

            draw_comparison_screen(
                screen,
                title_font,
                font,
                small,
                tiny,
                compare_results or [],
                compare_metric,
                compare_algo_idx,
            )
            present()
            continue

        hover_items = []
        proc_map = {p.pid: p for p in scheduler.processes}
        screen.fill(BG)

        algo_label = scheduler.algorithm
        if scheduler.algorithm == "PRIORITY":
            algo_label = "PRIORITY (Preemptive)" if scheduler.preemptive_priority else "PRIORITY (Non-preemptive)"

        header = [
            f"CPU Scheduling Visualizer ({algo_label}) - Phase 3",
            f"Algo: {algo_label} | Time: {scheduler.time} | Completed: {len(scheduler.completed)}/{len(scheduler.processes)} | Tick: {tick_ms}ms",
            "Controls: SPACE Pause/Resume | R Reset (clears live adds) | UP Slow | DOWN Fast | C Compare | F Fullscreen",
            "Add: A (popup, Shift+A preselect SYS) | P toggle preempt (PRIORITY) | Quantum: ←/→ (RR) | MLQ: SYS ←/→, USER Shift+←/→",
            "Algorithms: 1 FCFS | 2 SJF | 3 PRIORITY | 4 RR | 5 MLQ",
            f"Datasets: F1-F5 presets | L load processes.json | Status: {status_msg}",
        ]
        y = 16
        title_surf = title_font.render(header[0], True, TEXT)
        screen.blit(title_surf, (18, y))
        y += title_surf.get_height() + 6
        for ln in header[1:]:
            surf = small.render(ln, True, TEXT)
            screen.blit(surf, (18, y))
            y += surf.get_height() + 6

        # Layout: place panels below the header dynamically (prevents overlap)
        content_top = y + 14  # gap below header

        cpu_h = 140
        top_h_gap = 10
        info_h = 105

        # Dynamic vertical allocation (prevents overflow): Gantt + I/O + Metrics share remaining space.
        # Gantt and I/O get reasonable defaults; Metrics gets the rest.
        gantt_h = 190
        io_h = 160  # slightly taller so the tick labels fit cleanly
        metrics_h = 260  # will be recalculated after we know gantt_panel_y

        cpu_panel = pygame.Rect(40, content_top, 360, cpu_h)
        rq_panel  = pygame.Rect(430, content_top, 630, cpu_h)

        draw_panel(screen, cpu_panel, "CPU", font, small)
        if scheduler.running:
            chip = pygame.Rect(cpu_panel.x + 20, cpu_panel.y + 70, 220, 56)
            if scheduler.algorithm == "PRIORITY":
                label = f"{scheduler.running.pid} pr:{scheduler.running.priority} rem:{scheduler.running.remaining_time}"
            elif scheduler.algorithm == "MLQ":
                label = f"{scheduler.running.pid} {scheduler.running.queue} rem:{scheduler.running.remaining_time}"
            else:
                label = f"{scheduler.running.pid} rem:{scheduler.running.remaining_time}"
            pulse = int(60 + 90 * (0.5 + 0.5 * math.sin(now / 220.0)))
            draw_process_chip(screen, chip, label, CPU_RUN, small, glow_alpha=pulse)
            p = scheduler.running
            hover_items.append((
                chip,
                [
                    f"PID: {p.pid}",
                    f"AT: {p.arrival_time}   BT: {p.burst_time}   PR: {p.priority}",
                    f"Queue: {p.queue}   Rem: {p.remaining_time}",
                ],
            ))
        else:
            chip = pygame.Rect(cpu_panel.x + 20, cpu_panel.y + 70, 220, 56)
            draw_process_chip(screen, chip, "IDLE", CPU_IDLE, small)
            hover_items.append((chip, ["CPU: IDLE"]))

        if scheduler.algorithm == "MLQ":
            draw_panel(screen, rq_panel, "MLQ Queues (SYS then USER)", font, small)

            # Fit both SYS + USER rows inside rq_panel height (no overlap into the next panel)
            label_x = rq_panel.x + 20
            chip_x0 = rq_panel.x + 90
            chip_w, chip_h = 170, 40
            chip_dx = 180
            max_show = 3

            # SYS row
            sys_label_y = rq_panel.y + 34
            sys_chip_y = rq_panel.y + 50
            screen.blit(small.render("SYS:", True, MUTED), (label_x, sys_label_y))

            for i, p in enumerate(scheduler.sys_queue[:max_show]):
                chip = pygame.Rect(chip_x0 + i * chip_dx, sys_chip_y, chip_w, chip_h)
                label = f"{p.pid} rem:{p.remaining_time}"
                draw_process_chip(screen, chip, label, READY_BOX, small)
                hover_items.append((
                    chip,
                    [
                        f"PID: {p.pid}",
                        f"AT: {p.arrival_time}   BT: {p.burst_time}   PR: {p.priority}",
                        f"Queue: {p.queue}   Rem: {p.remaining_time}",
                    ],
                ))

            if len(scheduler.sys_queue) > max_show:
                more = small.render(f"(+{len(scheduler.sys_queue) - max_show} more)", True, MUTED)
                screen.blit(more, (chip_x0 + max_show * chip_dx, sys_chip_y + 10))

            # USER row
            user_label_y = rq_panel.y + 78
            user_chip_y = rq_panel.y + 94
            screen.blit(small.render("USER:", True, MUTED), (label_x, user_label_y))

            for i, p in enumerate(scheduler.user_queue[:max_show]):
                chip = pygame.Rect(chip_x0 + i * chip_dx, user_chip_y, chip_w, chip_h)
                label = f"{p.pid} rem:{p.remaining_time}"
                draw_process_chip(screen, chip, label, READY_BOX, small)
                hover_items.append((
                    chip,
                    [
                        f"PID: {p.pid}",
                        f"AT: {p.arrival_time}   BT: {p.burst_time}   PR: {p.priority}",
                        f"Queue: {p.queue}   Rem: {p.remaining_time}",
                    ],
                ))

            if len(scheduler.user_queue) > max_show:
                more = small.render(f"(+{len(scheduler.user_queue) - max_show} more)", True, MUTED)
                screen.blit(more, (chip_x0 + max_show * chip_dx, user_chip_y + 10))
        else:
            draw_panel(screen, rq_panel, "Ready Queue (front → back)", font, small)
            rx, ry = rq_panel.x + 20, rq_panel.y + 70
            for i, p in enumerate(scheduler.ready_queue[:6]):
                chip = pygame.Rect(rx + (i % 3) * 200, ry + (i // 3) * 70, 180, 56)
                if scheduler.algorithm == "PRIORITY":
                    label = f"{p.pid} pr:{p.priority} rem:{p.remaining_time}"
                else:
                    label = f"{p.pid} rem:{p.remaining_time}"
                draw_process_chip(screen, chip, label, READY_BOX, small)
                hover_items.append((
                    chip,
                    [
                        f"PID: {p.pid}",
                        f"AT: {p.arrival_time}   BT: {p.burst_time}   PR: {p.priority}",
                        f"Queue: {p.queue}   Rem: {p.remaining_time}",
                    ],
                ))

        info_panel_y = cpu_panel.y + cpu_panel.h + top_h_gap
        info_panel = pygame.Rect(40, info_panel_y, 1020, info_h)
        draw_info_legend_panel(screen, info_panel, scheduler, font, small)

        gantt_panel_y = info_panel.y + info_panel.h + top_h_gap

        gantt_panel = pygame.Rect(40, gantt_panel_y, 1020, gantt_h)
        draw_gantt(screen, gantt_panel, scheduler.gantt_chart, font, small, hover_items=hover_items, proc_map=proc_map)

        io_panel_y = gantt_panel.y + gantt_panel.h + top_h_gap
        io_panel = pygame.Rect(40, io_panel_y, 1020, io_h)
        draw_io_timeline(
            screen,
            io_panel,
            scheduler.io_gantt_chart,
            font,
            small,
            hover_items=hover_items,
            proc_map=proc_map,
            ref_len=len(scheduler.gantt_chart),
        )


        if paused:
            screen.blit(title_font.render("PAUSED", True, TEXT), (900, 14))

        if adding_process:
            draw_add_modal(screen, font, small, add_fields, active_field, add_status or "Fill details and press Enter")

        # Tooltip drawing (after all panels)
        if not adding_process:
            mx, my = to_logical(pygame.mouse.get_pos())
            tip_lines = None
            for r, lines in reversed(hover_items):
                if r.collidepoint((mx, my)):
                    tip_lines = lines
                    break
            if tip_lines:
                draw_tooltip(screen, (mx, my), tip_lines, tiny)

        present()

    pygame.quit()


if __name__ == "__main__":
    main()
