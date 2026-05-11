#!/usr/bin/env python3
"""Sequential init-stability trial runner for the three MPC workspaces.

Default mode preserves the existing C++ CSV path by running long enough for the
20 s eval writer to flush. For a faster init-only study, first patch the C++
eval writer threshold from 20.0 to 2.0 in each workspace, rebuild generation,
then run this script with --duration 6.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime
from typing import Optional


CONTROLLERS = {
    "hpipm": {
        "root": Path("/home/markwcl/Bipedal_MPC"),
        "csv": "hpipm_eval_result.csv",
    },
    "pmp": {
        "root": Path("/home/markwcl/My_MPC"),
        "csv": "pmp_eval_result.csv",
    },
    "pmp_mlp": {
        "root": Path("/home/markwcl/MPC_ML"),
        "csv": "pmp_ml_eval_result.csv",
    },
}


def existing_csvs(root: Path, csv_name: str) -> set[Path]:
    return set(root.glob(f"eval/*/{csv_name}"))


def newest_new_csv(root: Path, csv_name: str, before: set[Path]) -> Optional[Path]:
    candidates = [path for path in root.glob(f"eval/*/{csv_name}") if path not in before]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def first_window_max(csv_path: Path, window_s: float) -> float:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        first_time = None
        values = []
        for row in reader:
            timestamp = float(row["timestamp"])
            if first_time is None:
                first_time = timestamp
            if timestamp - first_time <= window_s:
                values.append(float(row["max_tracking_error"]))
            else:
                break
    return max(values) if values else float("nan")


def launch(workspace: Path, package_launch: str, env: dict[str, str], log_path: Path) -> subprocess.Popen:
    command = ["bash", "-lc", f"source install/setup.bash && ros2 launch {package_launch}"]
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd=str(workspace),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    process._init_stability_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        for sig, timeout_s in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 1)):
            try:
                os.killpg(os.getpgid(process.pid), sig)
                process.wait(timeout=timeout_s)
                break
            except Exception:
                continue
    log_handle = getattr(process, "_init_stability_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def run_trial(controller: str, trial: int, args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    cfg = CONTROLLERS[controller]
    root = cfg["root"]
    csv_name = cfg["csv"]
    before = existing_csvs(root, csv_name)

    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_log = out_dir / f"{controller}_{trial:04d}_{stamp}_sim.log"
    management_log = out_dir / f"{controller}_{trial:04d}_{stamp}_management.log"

    sim = launch(root, "sim sim_launch.py", env, sim_log)
    time.sleep(args.sim_warmup)
    management = launch(root, "management management_launch.py", env, management_log)
    time.sleep(args.duration)
    stop(management)
    stop(sim)
    time.sleep(args.cleanup_wait)

    csv_path = newest_new_csv(root, csv_name, before)
    if csv_path is None:
        return {
            "controller": controller,
            "trial": trial,
            "status": "no_csv",
            "first_window_max": "nan",
            "pass": False,
            "csv": "",
        }

    max_error = first_window_max(csv_path, args.window)
    passed = max_error < args.threshold
    return {
        "controller": controller,
        "trial": trial,
        "status": "ok",
        "first_window_max": f"{max_error:.6f}",
        "pass": passed,
        "csv": str(csv_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+", default=["hpipm", "pmp", "pmp_mlp"], choices=CONTROLLERS.keys())
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--sim-warmup", type=float, default=2.0)
    parser.add_argument("--cleanup-wait", type=float, default=1.0)
    parser.add_argument("--ros-domain-id", type=int, default=0)
    parser.add_argument("--out-dir", default="/home/markwcl/MPC_ML/eval/init_stability")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with summary_path.open("w", newline="") as handle:
        fieldnames = ["controller", "trial", "status", "first_window_max", "pass", "csv"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for controller in args.controllers:
            for trial in range(1, args.trials + 1):
                row = run_trial(controller, trial, args, out_dir)
                writer.writerow(row)
                handle.flush()
                print(row)

    print(f"[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
