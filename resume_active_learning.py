from __future__ import annotations

import argparse
import json
import time

import psutil


def wait_for_process(pid: int | None) -> None:
    if not pid:
        return
    while psutil.pid_exists(pid):
        time.sleep(5.0)


def wait_for_orphaned_solver() -> None:
    """Do not overlap a resumed run with a solver left by a dead parent."""
    while True:
        active = False
        for process in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (process.info.get("name") or "").lower()
                command = " ".join(process.info.get("cmdline") or []).lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if (
                name in {"cfx5solve.exe", "solver-mpi.exe", "mpiexec.exe", "cfxtg.exe"}
                and "opt_new" in command
                and "al_iter" in command
            ):
                active = True
                break
        if not active:
            return
        time.sleep(10.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--wait-pid", type=int, default=None)
    args = parser.parse_args()

    wait_for_process(args.wait_pid)
    wait_for_orphaned_solver()

    from impeller_app.config import AppConfig
    from impeller_app.core.active_learning import ActiveLearningService

    service = ActiveLearningService(AppConfig.load())
    legacy = service.legacy
    checkpoint = int(legacy.get_resume_iter())
    print(
        f"[守护续跑] checkpoint={checkpoint} target={args.target}",
        flush=True,
    )
    if checkpoint >= int(args.target):
        print("[守护续跑] 目标已经完成，无需再次启动。", flush=True)
        return 0
    result = legacy.main_multiobjective_active_learning(
        max_al_iters=int(args.target)
    )
    print(
        "[守护续跑结果] " + json.dumps(result, ensure_ascii=False),
        flush=True,
    )
    return 0 if int(result.get("completed_iters", 0)) >= int(args.target) else 2


if __name__ == "__main__":
    raise SystemExit(main())
