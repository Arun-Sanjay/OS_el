# ==============================
# CPU Scheduling Visualizer
# Phase 3: Visual UI + Multi-Algorithm Scheduler
# ==============================

from dataclasses import dataclass
from typing import List, Optional
import json

import pygame

# ------------------------------
# CONFIG
# ------------------------------
W, H = 1100, 900
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
    priority: int = 0          # lower number = higher priority
    queue: str = "USER"        # for MLQ: "SYS" or "USER"
    arrived: bool = False      # internal: has this process been enqueued yet?

    remaining_time: int = 0
    start_time: Optional[int] = None
    completion_time: Optional[int] = None

    def __post_init__(self):
        self.remaining_time = self.burst_time


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
          SYS queue = Round Robin (quantum_sys)
          USER queue = FCFS
          SYS has strict priority over USER (preempts USER at tick boundary)
    """

    def __init__(self, processes: List[Process], algorithm: str = "FCFS", quantum: int = 2):
        self.processes = processes
        self.algorithm = algorithm
        self.quantum = quantum          # RR quantum (time units)
        self.quantum_sys = 2            # MLQ SYS quantum
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

        # RR/MLQ time-slice tracking
        self.slice_left: int = 0

        # Stable list for arrival checks
        self._all = self.processes

        # Reset runtime fields on processes
        for p in self._all:
            p.remaining_time = p.burst_time
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
                self.slice_left = 0  # USER is FCFS

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

    def execute(self):
        if self.running:
            self.running.remaining_time -= 1
            self.gantt_chart.append(self.running.pid)

            if self.algorithm == "RR":
                self.slice_left -= 1
            elif self.algorithm == "MLQ" and self.running.queue.upper() == "SYS":
                self.slice_left -= 1

            if self.running.remaining_time == 0:
                self.running.completion_time = self.time + 1
                self.completed.append(self.running)
                self.running = None
                self.slice_left = 0
                return

            if self.algorithm == "RR" and self.slice_left == 0:
                self.ready_queue.append(self.running)
                self.running = None
            elif (
                self.algorithm == "MLQ"
                and self.running
                and self.running.queue.upper() == "SYS"
                and self.slice_left == 0
            ):
                self.sys_queue.append(self.running)
                self.running = None
        else:
            self.gantt_chart.append("IDLE")

    def tick(self):
        self.add_arrived_processes()
        self.schedule()
        self.execute()
        self.time += 1


def build_default_processes() -> List[Process]:
    # Default dataset now comes from processes.json if present.
    try:
        return load_processes_json("processes.json")
    except Exception:
        # Fallback to built-in defaults if the file is missing or malformed.
        return [
            Process("P1", arrival_time=0, burst_time=5, priority=2, queue="USER"),
            Process("P2", arrival_time=1, burst_time=3, priority=1, queue="SYS"),
            Process("P3", arrival_time=2, burst_time=6, priority=3, queue="USER"),
            Process("P4", arrival_time=4, burst_time=2, queue="SYS", priority=0),
        ]

# Helper: clone process list (no runtime fields)
def clone_processes(procs: List[Process]) -> List[Process]:
    # Recreate processes fresh (no runtime fields carried over)
    return [
        Process(p.pid, p.arrival_time, p.burst_time, priority=p.priority, queue=p.queue)
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
        processes.append(
            Process(
                pid=str(item["pid"]),
                arrival_time=int(item["arrival_time"]),
                burst_time=int(item["burst_time"]),
                priority=int(item.get("priority", 0)),
                queue=str(item.get("queue", "USER")),
            )
        )
    return processes


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


def draw_gantt(screen, rect, gantt: List[str], font, small):
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

    px_per_t = max(10, min(40, inner.w // max(1, total_t)))

    x0 = inner.x + 10
    y0 = inner.y + 18
    h = 54

    for pid, s, e in segs:
        bx = x0 + s * px_per_t
        bw = max(1, (e - s) * px_per_t)
        block = pygame.Rect(bx, y0, bw, h)
        pygame.draw.rect(screen, pid_color(pid), block, border_radius=8)
        pygame.draw.rect(screen, (20, 20, 20), block, 2, border_radius=8)

        if bw >= 40:
            label = small.render(pid, True, (10, 10, 10))
            screen.blit(label, (bx + 6, y0 + 16))

    max_markers = 16
    step = max(1, total_t // max_markers)
    for t in range(0, total_t + 1, step):
        mx = x0 + t * px_per_t
        pygame.draw.line(screen, GRID, (mx, y0 + h + 8), (mx, y0 + h + 22), 2)
        tt = small.render(str(t), True, MUTED)
        screen.blit(tt, (mx - 6, y0 + h + 26))


# ------------------------------
# Info & Legend panel (Step 1)
# ------------------------------
def draw_info_legend_panel(screen, rect, scheduler: CPUScheduler, font, small):
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
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
                line2 = f"MLQ: SYS quantum={scheduler.quantum_sys}  slice_left={scheduler.slice_left} (SYS dominates USER)"
            else:
                line2 = f"MLQ: SYS quantum={scheduler.quantum_sys}  (USER runs FCFS when SYS empty)"
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
        pygame.draw.rect(screen, (20, 20, 20), pygame.Rect(cx, cy, box, box), 2, border_radius=4)
        screen.blit(small.render(pid, True, MUTED), (cx + box + 8, cy - 2))


# ------------------------------
# Metrics computation and panel
# ------------------------------
def compute_metrics(processes: List[Process]):
    rows = []
    for p in processes:
        if p.completion_time is None or p.start_time is None:
            continue
        ct = p.completion_time
        tat = ct - p.arrival_time
        wt = tat - p.burst_time
        rt = p.start_time - p.arrival_time
        rows.append({
            "PID": p.pid,
            "AT": p.arrival_time,
            "BT": p.burst_time,
            "PR": p.priority,
            "Q": p.queue,
            "ST": p.start_time,
            "CT": ct,
            "TAT": tat,
            "WT": wt,
            "RT": rt,
        })

    if rows:
        avg_wt = sum(r["WT"] for r in rows) / len(rows)
        avg_tat = sum(r["TAT"] for r in rows) / len(rows)
        avg_rt = sum(r["RT"] for r in rows) / len(rows)
    else:
        avg_wt = avg_tat = avg_rt = 0.0

    return rows, avg_wt, avg_tat, avg_rt


def draw_metrics_panel(screen, rect, scheduler: CPUScheduler, font, small):
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    title = font.render("Metrics", True, TEXT)
    screen.blit(title, (rect.x + 12, rect.y + 10))

    rows, avg_wt, avg_tat, avg_rt = compute_metrics(scheduler.completed)

    total = len(scheduler.gantt_chart)
    busy = sum(1 for x in scheduler.gantt_chart if x != "IDLE")
    util = (busy / total * 100.0) if total else 0.0

    summary = f"Avg WT: {avg_wt:.2f}   Avg TAT: {avg_tat:.2f}   Avg RT: {avg_rt:.2f}   CPU Util: {util:.1f}%"
    screen.blit(small.render(summary, True, MUTED), (rect.x + 12, rect.y + 46))

    cols = ["PID", "AT", "BT", "PR", "Q", "ST", "CT", "TAT", "WT", "RT"]
    x = rect.x + 12
    y = rect.y + 78
    col_w = [50, 40, 40, 40, 75, 40, 40, 50, 45, 45]
    for c, w in zip(cols, col_w):
        screen.blit(small.render(c, True, TEXT), (x, y))
        x += w
    y += 28

    if not rows:
        screen.blit(small.render("(metrics will appear after completion)", True, MUTED), (rect.x + 12, y))
        return

    max_rows = max(1, (rect.y + rect.h - y - 10) // 24)
    for r in rows[:max_rows]:
        x = rect.x + 12
        for c, w in zip(cols, col_w):
            screen.blit(small.render(str(r[c]), True, MUTED), (x, y))
            x += w
        y += 24


# ------------------------------
# Start screen
# ------------------------------
def draw_start_screen(screen, font, small, state):
    screen.fill(BG)

    title = font.render("CPU Scheduling Visualizer", True, TEXT)
    screen.blit(title, (W // 2 - title.get_width() // 2, 80))

    sub = small.render("Start setup (choose defaults, then press ENTER)", True, MUTED)
    screen.blit(sub, (W // 2 - sub.get_width() // 2, 120))

    panel = pygame.Rect(260, 180, 580, 460)
    pygame.draw.rect(screen, PANEL, panel, border_radius=16)
    pygame.draw.rect(screen, BORDER, panel, 2, border_radius=16)

    # Rows
    y = panel.y + 40

    algo = state["algorithms"][state["algo_idx"]]
    algo_line = small.render(f"Algorithm:  {algo}", True, TEXT)
    screen.blit(algo_line, (panel.x + 30, y))
    hint = small.render("Use ↑/↓ to change", True, MUTED)
    screen.blit(hint, (panel.x + 360, y))
    y += 70

    tick_line = small.render(f"Default Tick Time:  {state['tick_ms']} ms", True, TEXT)
    screen.blit(tick_line, (panel.x + 30, y))
    hint = small.render("Use ←/→ to adjust", True, MUTED)
    screen.blit(hint, (panel.x + 360, y))
    y += 70

    if algo == "RR":
        q_line = small.render(f"Quantum (RR):  {state['quantum']}", True, TEXT)
        screen.blit(q_line, (panel.x + 30, y))
        hint = small.render("Use A/D to adjust", True, MUTED)
        screen.blit(hint, (panel.x + 360, y))
        y += 70
    else:
        q_line = small.render("Quantum (RR):  (not applicable)", True, MUTED)
        screen.blit(q_line, (panel.x + 30, y))
        y += 70

    # Short key hints (avoid clipping)
    keys = [
        "ENTER: Start simulation",
        "ESC: Quit",
    ]
    for ln in keys:
        screen.blit(small.render(ln, True, MUTED), (panel.x + 30, y))
        y += 28

    # Note (wrapped) placed above the button so it never clips
    note_y = panel.y + panel.h - 150
    note1 = "Note: After starting, you can still change tick speed"
    note2 = "(UP/DOWN) and quantum (←/→) while running."
    screen.blit(small.render(note1, True, MUTED), (panel.x + 30, note_y))
    screen.blit(small.render(note2, True, MUTED), (panel.x + 30, note_y + 24))

    # Start button (clickable)
    btn = pygame.Rect(panel.x + 190, panel.y + panel.h - 78, 200, 50)
    pygame.draw.rect(screen, (60, 130, 220), btn, border_radius=12)
    pygame.draw.rect(screen, (20, 20, 20), btn, 2, border_radius=12)
    btxt = small.render("START", True, (10, 10, 10))
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
    pygame.draw.rect(screen, PANEL, box, border_radius=16)
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
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CPU Scheduling Visualizer - Phase 3")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 26)
    small = pygame.font.SysFont("Arial", 20)

    base_dataset = build_default_processes()  # last loaded preset/JSON; does not include live additions

    # App mode: "menu" (start screen) or "sim" (running simulator)
    app_mode = "menu"

    start_state = {
        "algorithms": ["FCFS", "SJF", "PRIORITY", "RR", "MLQ"],
        "algo_idx": 0,            # default FCFS
        "tick_ms": TICK_MS_DEFAULT,
        "quantum": 2,
        "start_button": None,
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

    def reset_scheduler():
        nonlocal scheduler, status_msg, live_added
        if scheduler is None:
            return
        prev_preempt = scheduler.preemptive_priority
        scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
        scheduler.preemptive_priority = prev_preempt
        live_added = []
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

        scheduler.processes.append(live_proc)
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

            # Start screen controls
            if app_mode == "menu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
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
                    if start_state.get("start_button") and start_state["start_button"].collidepoint(event.pos):
                        algo = start_state["algorithms"][start_state["algo_idx"]]
                        scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=algo, quantum=start_state["quantum"])
                        tick_ms = start_state["tick_ms"]
                        paused = False
                        last_tick = pygame.time.get_ticks()
                        status_msg = "Started"
                        app_mode = "sim"
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
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_scheduler()
                elif event.key == pygame.K_UP:
                    tick_ms = min(1500, tick_ms + 100)
                elif event.key == pygame.K_DOWN:
                    tick_ms = max(100, tick_ms - 100)
                elif event.key == pygame.K_1:
                    scheduler.set_algorithm("FCFS")
                elif event.key == pygame.K_2:
                    scheduler.set_algorithm("SJF")
                elif event.key == pygame.K_3:
                    scheduler.set_algorithm("PRIORITY")
                elif event.key == pygame.K_4:
                    scheduler.set_algorithm("RR")
                elif event.key == pygame.K_5:
                    scheduler.set_algorithm("MLQ")
                elif event.key == pygame.K_LEFT:
                    scheduler.quantum = max(1, scheduler.quantum - 1)
                elif event.key == pygame.K_RIGHT:
                    scheduler.quantum = min(10, scheduler.quantum + 1)
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
                elif event.key == pygame.K_F2:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(2)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F2 (idle gaps, live additions cleared)"
                    live_counter = 1
                elif event.key == pygame.K_F3:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(3)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F3 (priority, live additions cleared)"
                    live_counter = 1
                elif event.key == pygame.K_F4:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(4)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F4 (RR, live additions cleared)"
                    live_counter = 1
                elif event.key == pygame.K_F5:
                    prev_preempt = scheduler.preemptive_priority
                    base_dataset = load_preset(5)
                    scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                    scheduler.preemptive_priority = prev_preempt
                    live_added = []
                    status_msg = "Loaded preset F5 (MLQ, live additions cleared)"
                    live_counter = 1
                elif event.key == pygame.K_l:
                    try:
                        prev_preempt = scheduler.preemptive_priority
                        base_dataset = load_processes_json("processes.json")
                        scheduler = CPUScheduler(clone_processes(base_dataset), algorithm=scheduler.algorithm, quantum=scheduler.quantum)
                        scheduler.preemptive_priority = prev_preempt
                        live_added = []
                        status_msg = "Loaded processes.json (live additions cleared)"
                        live_counter = 1
                    except Exception:
                        status_msg = "Load failed"
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

        if app_mode == "menu":
            draw_start_screen(screen, font, small, start_state)
            pygame.display.flip()
            continue

        screen.fill(BG)

        algo_label = scheduler.algorithm
        if scheduler.algorithm == "PRIORITY":
            algo_label = "PRIORITY (Preemptive)" if scheduler.preemptive_priority else "PRIORITY (Non-preemptive)"

        header = [
            f"CPU Scheduling Visualizer ({algo_label}) - Phase 3",
            f"Algo: {algo_label} | Time: {scheduler.time} | Completed: {len(scheduler.completed)}/{len(scheduler.processes)} | Tick: {tick_ms}ms",
            "Controls: SPACE Pause/Resume | R Reset (clears live adds) | UP Slow | DOWN Fast",
            "Add: A (popup, Shift+A preselect SYS) | P toggle preempt (PRIORITY) | Quantum: ←/→",
            "Algorithms: 1 FCFS | 2 SJF | 3 PRIORITY | 4 RR | 5 MLQ",
            f"Datasets: F1-F5 presets | L load processes.json | Status: {status_msg}",
        ]
        y = 18
        for ln in header:
            surf = small.render(ln, True, TEXT)
            screen.blit(surf, (18, y))
            y += 26

        # Layout: place panels below the header dynamically (prevents overlap)
        content_top = y + 14  # gap below header

        cpu_h = 170
        top_h_gap = 10
        info_h = 105
        gantt_h = 220
        metrics_h = 170

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
            draw_process_chip(screen, chip, label, CPU_RUN, small)
        else:
            chip = pygame.Rect(cpu_panel.x + 20, cpu_panel.y + 70, 220, 56)
            draw_process_chip(screen, chip, "IDLE", CPU_IDLE, small)

        if scheduler.algorithm == "MLQ":
            draw_panel(screen, rq_panel, "MLQ Queues (SYS then USER)", font, small)

            # Fit both SYS + USER rows inside rq_panel height (no overlap into the next panel)
            label_x = rq_panel.x + 20
            chip_x0 = rq_panel.x + 90
            chip_w, chip_h = 170, 40
            chip_dx = 180
            max_show = 3

            # SYS row
            sys_label_y = rq_panel.y + 40
            sys_chip_y = rq_panel.y + 56
            screen.blit(small.render("SYS:", True, MUTED), (label_x, sys_label_y))

            for i, p in enumerate(scheduler.sys_queue[:max_show]):
                chip = pygame.Rect(chip_x0 + i * chip_dx, sys_chip_y, chip_w, chip_h)
                label = f"{p.pid} rem:{p.remaining_time}"
                draw_process_chip(screen, chip, label, READY_BOX, small)

            if len(scheduler.sys_queue) > max_show:
                more = small.render(f"(+{len(scheduler.sys_queue) - max_show} more)", True, MUTED)
                screen.blit(more, (chip_x0 + max_show * chip_dx, sys_chip_y + 10))

            # USER row
            user_label_y = rq_panel.y + 86
            user_chip_y = rq_panel.y + 102
            screen.blit(small.render("USER:", True, MUTED), (label_x, user_label_y))

            for i, p in enumerate(scheduler.user_queue[:max_show]):
                chip = pygame.Rect(chip_x0 + i * chip_dx, user_chip_y, chip_w, chip_h)
                label = f"{p.pid} rem:{p.remaining_time}"
                draw_process_chip(screen, chip, label, READY_BOX, small)

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

        info_panel_y = cpu_panel.y + cpu_panel.h + top_h_gap
        info_panel = pygame.Rect(40, info_panel_y, 1020, info_h)
        draw_info_legend_panel(screen, info_panel, scheduler, font, small)

        gantt_panel_y = info_panel.y + info_panel.h + top_h_gap
        gantt_panel = pygame.Rect(40, gantt_panel_y, 1020, gantt_h)
        draw_gantt(screen, gantt_panel, scheduler.gantt_chart, font, small)

        metrics_panel_y = gantt_panel.y + gantt_panel.h + top_h_gap
        metrics_panel = pygame.Rect(40, metrics_panel_y, 1020, metrics_h)
        draw_metrics_panel(screen, metrics_panel, scheduler, font, small)

        if paused:
            screen.blit(font.render("PAUSED", True, (255, 255, 255)), (950, 18))

        if adding_process:
            draw_add_modal(screen, font, small, add_fields, active_field, add_status or "Fill details and press Enter")

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
