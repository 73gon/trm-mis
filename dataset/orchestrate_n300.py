#!/usr/bin/env python3
"""End-to-end Python orchestrator for the n=300 v8 pipeline.

What it does:
  1. Launches 18 single-threaded Gurobi workers (15 train + 3 test) to build
     the n=300 multi-MIS dataset into per-worker chunk dirs.
  2. Polls every POLL_SECS using time.sleep, reporting builder progress and
     SLURM job statuses.
  3. When all builders exit AND we have at least MIN_TRAIN_SHARDS + MIN_TEST_SHARDS
     shards, runs the chunk merger and submits slurm/smallmis_n300_v8.slurm.
  4. Continues polling until the n=300 training job is no longer in squeue.
  5. Regenerates visualizations/learning_curve_v8_all.png with all four runs.

Designed to be run with `nohup ... &` so it survives terminal disconnects.
All progress is appended to logs/orchestrator_n300.log.
"""

from __future__ import annotations

import glob
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/mmardan/trm")
os.chdir(ROOT)

LOG = ROOT / "logs" / "orchestrator_n300.log"
LOG.parent.mkdir(exist_ok=True)

DATA_DIR = ROOT / "data" / "smallmis_n300_multi"
SLURM_SCRIPT = ROOT / "slurm" / "smallmis_n300_v8.slurm"
MERGE_SCRIPT = ROOT / "dataset" / "merge_n300_chunks.py"
PLOT_SCRIPT = ROOT / "visualizations" / "plot_curves.py"
BUILDER_MODULE = "dataset.build_mis_dataset_multilabel"
VENV_PY = ROOT / ".venv" / "bin" / "python"

# 18 single-threaded Gurobi workers (login node has 24 CPUs).
TRAIN_WORKERS = 15
TRAIN_PER_WORKER = 200  # 15 * 200 = 3000 train graphs total
TEST_WORKERS = 3
TEST_PER_WORKER = 100  # 3 * 100 = 300 test graphs total
SHARD_SIZE = 50  # first shard at ~1.4h (50 * ~100s/graph)
POOL_TIME_LIMIT = 120.0
TRAIN_SEED_BASE = 7_000_000
TEST_SEED_BASE = 8_000_000

POLL_SECS = 300

JOB_N50 = 3991
JOB_N100 = 3992
JOB_N200 = 4060
PRIOR_LOGS = [
    ROOT / "logs" / f"smallmis_n50_v8_{JOB_N50}.log",
    ROOT / "logs" / f"smallmis_n100_v8_{JOB_N100}.log",
    ROOT / "logs" / f"smallmis_n200_v8_{JOB_N200}.log",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def squeue_state(job_id: int) -> str:
    if job_id is None:
        return ""
    out = sh(f"squeue -j {job_id} -h -o '%T'").stdout.strip()
    return out


def builder_count() -> int:
    out = sh(f"pgrep -fc {BUILDER_MODULE}").stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


def shard_counts() -> tuple[int, int]:
    train = len(glob.glob(str(DATA_DIR / "train_chunk_*" / "mis_shard_*.pt")))
    test = len(glob.glob(str(DATA_DIR / "test_chunk_*" / "mis_shard_*.pt")))
    return train, test


def merged_shard_counts() -> tuple[int, int]:
    train = len(glob.glob(str(DATA_DIR / "train" / "mis_shard_*.pt")))
    test = len(glob.glob(str(DATA_DIR / "test" / "mis_shard_*.pt")))
    return train, test


def launch_builders() -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    for i in range(TRAIN_WORKERS):
        seed = TRAIN_SEED_BASE + i * 100_000
        out_dir = DATA_DIR / f"train_chunk_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = ROOT / "logs" / f"build_n300_train_chunk_{i}.log"
        cmd = [
            str(VENV_PY),
            "-m",
            BUILDER_MODULE,
            "--output-dir",
            str(out_dir),
            "--num-instances",
            str(TRAIN_PER_WORKER),
            "--shard-size",
            str(SHARD_SIZE),
            "--seed-start",
            str(seed),
            "--n-min",
            "300",
            "--n-max",
            "300",
            "--d-min",
            "7.35",
            "--d-max",
            "7.35",
            "--pool-size",
            "16",
            "--pool-gap",
            "0.0",
            "--pool-time-limit",
            str(POOL_TIME_LIMIT),
            "--threads",
            "1",
        ]
        f = log_path.open("w")
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT))
        log(f"launched train chunk {i} -> {out_dir} (log {log_path.name})")

    for i in range(TEST_WORKERS):
        seed = TEST_SEED_BASE + i * 100_000
        out_dir = DATA_DIR / f"test_chunk_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = ROOT / "logs" / f"build_n300_test_chunk_{i}.log"
        cmd = [
            str(VENV_PY),
            "-m",
            BUILDER_MODULE,
            "--output-dir",
            str(out_dir),
            "--num-instances",
            str(TEST_PER_WORKER),
            "--shard-size",
            str(SHARD_SIZE),
            "--seed-start",
            str(seed),
            "--n-min",
            "300",
            "--n-max",
            "300",
            "--d-min",
            "7.35",
            "--d-max",
            "7.35",
            "--pool-size",
            "16",
            "--pool-gap",
            "0.0",
            "--pool-time-limit",
            str(POOL_TIME_LIMIT),
            "--threads",
            "1",
        ]
        f = log_path.open("w")
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT))
        log(f"launched test chunk {i} -> {out_dir} (log {log_path.name})")

    return procs


def run_merge() -> tuple[int, int]:
    res = sh(f"{shlex.quote(str(VENV_PY))} {shlex.quote(str(MERGE_SCRIPT))}")
    log("merge stdout:\n" + res.stdout.strip())
    if res.stderr.strip():
        log("merge stderr:\n" + res.stderr.strip())
    return merged_shard_counts()


def submit_n300() -> int | None:
    res = sh(f"sbatch {shlex.quote(str(SLURM_SCRIPT))}")
    if res.returncode != 0:
        log(f"sbatch FAILED: {res.stderr}")
        return None
    m = re.search(r"(\d+)", res.stdout)
    if not m:
        log(f"could not parse job id from: {res.stdout!r}")
        return None
    return int(m.group(1))


def find_log(job_id: int) -> Path | None:
    matches = sorted(glob.glob(str(ROOT / f"logs/smallmis_n300_v8_{job_id}.log")))
    return Path(matches[-1]) if matches else None


def parse_eval(log_path: Path | None) -> tuple[int, float | None, float | None, int | None]:
    """Return (count, best_bce, best_pp_AR, last_iter)."""
    if not log_path or not log_path.exists():
        return 0, None, None, None
    bces, pps, its = [], [], []
    bce_re = re.compile(r"bce=([0-9.]+)")
    pp_re = re.compile(r"pp_AR=([0-9.]+)")
    it_re = re.compile(r"eval-test it=(\d+)")
    with log_path.open() as f:
        for line in f:
            if "eval-test it=" in line:
                m_b = bce_re.search(line)
                m_p = pp_re.search(line)
                m_i = it_re.search(line)
                if m_b:
                    bces.append(float(m_b.group(1)))
                if m_p:
                    pps.append(float(m_p.group(1)))
                if m_i:
                    its.append(int(m_i.group(1)))
    if not bces:
        return 0, None, None, None
    return len(bces), min(bces), max(pps), its[-1] if its else None


def generate_final_plot(j300: int | None) -> None:
    logs = [str(p) for p in PRIOR_LOGS]
    if j300 is not None:
        p = find_log(j300)
        if p:
            logs.append(str(p))
    out_png = ROOT / "visualizations" / "learning_curve_v8_all.png"
    cmd = f"{shlex.quote(str(VENV_PY))} {shlex.quote(str(PLOT_SCRIPT))} {' '.join(shlex.quote(l) for l in logs)} -o {shlex.quote(str(out_png))}"
    res = sh(cmd)
    log("plot stdout: " + res.stdout.strip())
    if res.stderr.strip():
        log("plot stderr: " + res.stderr.strip())


def main() -> int:
    log("== orchestrator start ==")
    log(
        f"n=300 plan: train={TRAIN_WORKERS}x{TRAIN_PER_WORKER}={TRAIN_WORKERS * TRAIN_PER_WORKER}, "
        f"test={TEST_WORKERS}x{TEST_PER_WORKER}={TEST_WORKERS * TEST_PER_WORKER}, "
        f"shard_size={SHARD_SIZE}, pool_time_limit={POOL_TIME_LIMIT}s"
    )

    procs = launch_builders()
    log(f"launched {len(procs)} workers")

    j300: int | None = None
    submitted = False

    iters = 0
    while True:
        iters += 1
        time.sleep(POLL_SECS)

        wk = builder_count()
        tr, te = shard_counts()
        mtr, mte = merged_shard_counts()
        st300 = squeue_state(j300) if j300 else ""
        ev, best_bce, best_pp, last_it = parse_eval(find_log(j300)) if j300 else (0, None, None, None)

        log(
            f"poll#{iters} | builders={wk} chunkshards tr={tr} te={te} merged tr={mtr} te={mte} "
            f"| n300_job={j300}({st300}) ev={ev} bestBCE={best_bce} bestPP={best_pp} last_it={last_it}"
        )

        # Trigger merge + submit
        if not submitted and wk == 0 and tr >= 1 and te >= 1:
            log("all builders done -> running merge")
            mtr, mte = run_merge()
            log(f"merged train={mtr} test={mte}")
            if mtr >= 1 and mte >= 1:
                j300 = submit_n300()
                if j300:
                    log(f"submitted n=300 v8 job={j300}")
                    submitted = True
                else:
                    log("submission failed, will retry next loop")
            else:
                log("merge produced 0 shards; cannot submit")

        # Exit condition: submitted, job no longer in queue
        if submitted and j300 and not st300:
            log("n=300 training job no longer in squeue -> finalize")
            break

        # Hard cap (~12h of polling = 144 iters * 300s)
        if iters > 200:
            log("hit iteration cap; bailing out")
            break

    # Final sacct + plot
    if j300:
        res = sh(f"sacct -j {j300} --format=JobID,State,ExitCode,Elapsed -P")
        log("final sacct:\n" + res.stdout.strip())
    generate_final_plot(j300)
    log("== orchestrator done ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
