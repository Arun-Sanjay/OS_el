import json
import sys

from main import Process, CPUScheduler, compute_metrics


def build_processes(dataset):
    return [
        Process(
            str(item["pid"]),
            int(item["arrival_time"]),
            int(item["burst_time"]),
            priority=int(item.get("priority", 0)),
            queue=str(item.get("queue", "USER")),
        )
        for item in dataset
    ]


def run_test(test):
    dataset = test["dataset"]
    settings = test.get("settings", {})
    algo = settings.get("algorithm", "FCFS")
    quantum = int(settings.get("quantum", 2))
    preemptive = bool(settings.get("preemptive_priority", True))

    sched = CPUScheduler(build_processes(dataset), algorithm=algo, quantum=quantum)
    if algo == "PRIORITY":
        sched.preemptive_priority = preemptive

    guard = 0
    while (not sched.done()) and guard < 200000:
        sched.tick()
        guard += 1

    rows, avg_wt, avg_tat, avg_rt = compute_metrics(sched.processes)
    row_map = {r["PID"]: r for r in rows}

    expected = test.get("expected", {})
    if "per_pid" in expected:
        for pid, exp in expected["per_pid"].items():
            if pid not in row_map:
                return False, f"missing PID {pid}"
            row = row_map[pid]
            for key in ["CT", "TAT", "WT", "RT"]:
                got = row.get(key)
                if got == "-":
                    return False, f"PID {pid} {key} not computed"
                if int(got) != int(exp[key]):
                    return False, f"PID {pid} {key} expected {exp[key]} got {got}"
        return True, "ok"

    if "averages" in expected:
        tol = float(expected.get("tolerance", 0.01))
        avg = {"avg_wt": avg_wt, "avg_tat": avg_tat, "avg_rt": avg_rt}
        for k, v in expected["averages"].items():
            if abs(avg[k] - float(v)) > tol:
                return False, f"{k} expected {v} got {avg[k]:.4f}"
        return True, "ok"

    return False, "no expected outputs"


def main():
    with open("tests/expected_outputs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    failed = 0
    for t in tests:
        ok, msg = run_test(t)
        name = t.get("name", "(unnamed)")
        if ok:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name}: {msg}")

    if failed:
        print(f"{failed} test(s) failed")
        sys.exit(1)

    print("All tests passed")


if __name__ == "__main__":
    main()
