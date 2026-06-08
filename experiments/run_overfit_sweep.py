#!/usr/bin/env python3
"""Driver for the overfit sweep: stages N=1, 250, 2500, 6250, 12500, 25000 on
both SATLIB and ER-700-800. Submits SLURM jobs, waits for completion, parses
outcome, retries once with 2x iterations on failure. Writes a summary JSON
at the end so the separate documenter step can render the final report.

Usage
-----
    python -m experiments.run_overfit_sweep            # run
    python -m experiments.run_overfit_sweep --dry_run  # just print plan

Safe to interrupt: it re-checks SLURM queue before submitting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/mmardan/trm")
TEMPLATE = ROOT / "slurm" / "overfit" / "overfit_sl_template.slurm"
SWEEP_LOG = ROOT / "logs" / "overfit_sweep_summary.json"
LOCK = ROOT / "logs" / "overfit_sweep_driver.lock"
SWEEP_PREFIX = "sweep1"

DATASETS = ["satlib", "er_700_800"]

# (stage_tag, num_graphs, iterations, budget_tag, extra_args)
# Extra args: batch size and hyperparameters that scale with N.
STAGES = [
    ("stage1_N1",     1,     3000,  "3k",  "--batch_size 1  --lr 1e-3 --feasibility_weight 1.0"),
    ("stage2_N250",   250,   15000, "15k", "--batch_size 16 --lr 5e-4 --feasibility_weight 2.0"),
    ("stage3_N1500",  1500,  25000, "25k", "--batch_size 16 --lr 5e-4 --feasibility_weight 2.0"),
    ("stage4_N9000",  9000,  40000, "40k", "--batch_size 16 --lr 3e-4 --feasibility_weight 2.0"),
    ("stage5_N25000", 25000, 60000, "60k", "--batch_size 16 --lr 3e-4 --feasibility_weight 2.0"),
]


# ------------------------------ SLURM helpers -------------------------------

def sbatch_submit(*, dataset: str, stage_tag: str, num_graphs: int,
                  iterations: int, budget_tag: str, retry: bool,
                  extra_args: str = "") -> int:
    """Submit one SLURM job; return job id."""
    ds_short = "sat" if dataset == "satlib" else "er"
    run_name = f"{SWEEP_PREFIX}_{ds_short}_N{num_graphs}"
    if retry:
        run_name += "_retry"
    chkpt_dir = f"checkpoints/{run_name}"
    job_name = run_name  # identical so squeue is readable

    exports = [
        f"DATASET={dataset}",
        f"NUM_GRAPHS={num_graphs}",
        f"ITERATIONS={iterations}",
        f"STAGE_TAG={stage_tag}",
        f"RUN_NAME={run_name}",
        f"CHKPT_DIR={chkpt_dir}",
        f"ITER_BUDGET_TAG={budget_tag}",
        f"EXTRA_ARGS={extra_args}",
    ]
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--export=ALL," + ",".join(exports),
        str(TEMPLATE),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"sbatch output unexpected: {out}")
    jid = int(m.group(1))
    print(f"    -> submitted {dataset} N={num_graphs} iters={iterations} "
          f"job={jid} (retry={retry})  run_name={run_name}")
    return jid


def squeue_active(job_ids: list[int]) -> set[int]:
    """Return subset of job_ids that are still in the queue (pending or running)."""
    if not job_ids:
        return set()
    out = subprocess.check_output(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%A"],
        text=True,
    ).split()
    active = {int(x) for x in out if x.isdigit()}
    return {j for j in job_ids if j in active}


def wait_for_jobs(job_ids: list[int], poll_sec: int = 60) -> None:
    print(f"  waiting for jobs {job_ids} ...", flush=True)
    while True:
        still = squeue_active(job_ids)
        if not still:
            print(f"  all jobs finished: {job_ids}", flush=True)
            return
        print(f"    still running: {sorted(still)} "
              f"({len(job_ids) - len(still)}/{len(job_ids)} done)", flush=True)
        time.sleep(poll_sec)


# ---------------------------- outcome parsing -------------------------------

def find_log(job_id: int) -> Path | None:
    # logs/<jobname>_<jid>.log; jobname varies, so glob
    hits = list((ROOT / "logs").glob(f"*_{job_id}.log"))
    return hits[0] if hits else None


def parse_result(job_id: int) -> dict:
    log = find_log(job_id)
    if log is None:
        return {"job_id": job_id, "status": "no_log"}
    text = log.read_text(errors="replace")

    def grab(pattern, group=1, cast=float, default=None):
        m = re.search(pattern, text)
        if not m:
            return default
        try:
            return cast(m.group(group))
        except Exception:
            return default

    best_pp_ar = grab(r"best pp_AR=([0-9.]+)", cast=float)
    solved_at = grab(r"solved_at=(-?\d+)", cast=int)
    early = grab(r"early_stopped_at=(-?\d+)", cast=int)
    # Find reason word after 'reason='
    mreason = re.search(r"early_stopped_at=-?\d+ reason=(\S*)", text)
    reason = mreason.group(1) if mreason else ""
    # Last metrics line
    last_line = None
    for line in reversed(text.splitlines()):
        if line.startswith("[it "):
            last_line = line
            break
    rc_match = re.search(r"overfit_sl exited rc=(\d+)", text)
    rc = int(rc_match.group(1)) if rc_match else None
    gpu_match = re.search(r"GPU avg=([0-9.]+)%", text)
    gpu_avg = float(gpu_match.group(1)) if gpu_match else None

    # Criterion: pass if either criterion fired (early_stop), or best_pp_ar >= 0.999
    passed = False
    if early and early > 0:
        passed = True
    elif best_pp_ar is not None and best_pp_ar >= 0.999:
        passed = True

    return {
        "job_id": job_id,
        "log_path": str(log),
        "rc": rc,
        "best_pp_ar": best_pp_ar,
        "solved_at": solved_at,
        "early_stopped_at": early,
        "stop_reason": reason,
        "last_metrics_line": last_line,
        "gpu_avg_pct": gpu_avg,
        "passed": passed,
    }


# -------------------------- cache readiness check ---------------------------

def cache_ready(dataset: str, num_graphs: int) -> bool:
    """True if enough mis_shard_*.cache.pt files exist to cover num_graphs."""
    d = ROOT / f"data/difusco_benchmark/datasets/{dataset}/train"
    shards = sorted(d.glob("mis_shard_*.pt"))
    shards = [s for s in shards if not s.name.endswith(".cache.pt")]
    need = (num_graphs + 249) // 250
    covered = 0
    for s in shards[:need]:
        cache = s.with_suffix(".cache.pt")
        if cache.exists() and cache.stat().st_mtime >= s.stat().st_mtime:
            covered += 1
        else:
            break
    return covered >= need


def wait_for_cache(dataset: str, num_graphs: int, poll_sec: int = 60) -> None:
    if cache_ready(dataset, num_graphs):
        return
    print(f"  waiting for {dataset} cache to cover {num_graphs} graphs ...",
          flush=True)
    while not cache_ready(dataset, num_graphs):
        time.sleep(poll_sec)
    print(f"  {dataset} cache ready for {num_graphs} graphs", flush=True)


# --------------------------------- main -------------------------------------

def run_stage(stage_tag: str, num_graphs: int, iterations: int,
              budget_tag: str, extra_args: str = "") -> list[dict]:
    print(f"\n### {stage_tag}: N={num_graphs} iters={iterations} "
          f"budget={budget_tag} extra='{extra_args}'", flush=True)
    # Ensure caches cover this N for each dataset (stage 1 is fine without cache).
    if num_graphs > 1:
        for ds in DATASETS:
            wait_for_cache(ds, num_graphs)

    # Submit both datasets in parallel
    jobs: dict[int, str] = {}  # job_id -> dataset
    for ds in DATASETS:
        jid = sbatch_submit(dataset=ds, stage_tag=stage_tag,
                            num_graphs=num_graphs, iterations=iterations,
                            budget_tag=budget_tag, retry=False,
                            extra_args=extra_args)
        jobs[jid] = ds
    wait_for_jobs(list(jobs.keys()))

    results = []
    for jid, ds in jobs.items():
        res = parse_result(jid)
        res["dataset"] = ds
        res["stage_tag"] = stage_tag
        res["num_graphs"] = num_graphs
        res["iterations"] = iterations
        res["attempt"] = 1
        results.append(res)
        print(f"  [{ds}] passed={res['passed']} best_pp_AR={res['best_pp_ar']} "
              f"early={res['early_stopped_at']} reason={res['stop_reason']} "
              f"rc={res['rc']} gpu={res['gpu_avg_pct']}")

    # Retry failures with 2x iterations (one retry max)
    retry_jobs: dict[int, str] = {}
    for res in results:
        if not res["passed"]:
            new_iters = iterations * 2
            print(f"  !! {res['dataset']} failed, retrying with {new_iters} iters")
            jid = sbatch_submit(dataset=res["dataset"], stage_tag=stage_tag,
                                num_graphs=num_graphs, iterations=new_iters,
                                budget_tag=f"{budget_tag}_2x", retry=True,
                                extra_args=extra_args)
            retry_jobs[jid] = res["dataset"]
    if retry_jobs:
        wait_for_jobs(list(retry_jobs.keys()))
        for jid, ds in retry_jobs.items():
            res = parse_result(jid)
            res["dataset"] = ds
            res["stage_tag"] = stage_tag
            res["num_graphs"] = num_graphs
            res["iterations"] = iterations * 2
            res["attempt"] = 2
            results.append(res)
            print(f"  [{ds} retry] passed={res['passed']} "
                  f"best_pp_AR={res['best_pp_ar']}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--start_at", type=int, default=1,
                    help="Skip to stage N (1-indexed).")
    args = ap.parse_args()

    # --- single-instance lock ---
    if not args.dry_run:
        if LOCK.exists():
            try:
                pid = int(LOCK.read_text().strip())
                if pid > 0 and Path(f"/proc/{pid}").exists():
                    print(f"!! Another driver is already running (pid={pid}). "
                          f"Delete {LOCK} if this is wrong.", file=sys.stderr)
                    return 1
            except Exception:
                pass
        LOCK.write_text(str(os.getpid()))

    plan = STAGES[args.start_at - 1:]
    print("Plan:")
    for st in plan:
        print(f"  {st}")
    if args.dry_run:
        return 0

    all_results: list[dict] = []
    # Resume partially-complete summary if present
    if SWEEP_LOG.exists():
        try:
            all_results = json.loads(SWEEP_LOG.read_text()).get("results", [])
        except Exception:
            all_results = []

    for tag, N, it, bud, extra in plan:
        stage_results = run_stage(tag, N, it, bud, extra)
        all_results.extend(stage_results)
        SWEEP_LOG.write_text(json.dumps(
            {"results": all_results, "updated": time.time()}, indent=2))
        print(f"  wrote partial summary to {SWEEP_LOG}")

        # Gate: both datasets must pass (either first attempt or retry).
        # We look only at attempts for THIS stage.
        by_ds: dict[str, bool] = {}
        for r in stage_results:
            if r["dataset"] not in by_ds or r["passed"]:
                by_ds[r["dataset"]] = r["passed"] or by_ds.get(r["dataset"], False)
        if not all(by_ds.get(ds, False) for ds in DATASETS):
            print(f"\n!! Stage {tag} did not pass on both datasets: {by_ds}")
            print("!! Aborting sweep. Investigate logs before rerunning next stage.")
            return 2

    print("\n### Sweep complete. Summary:")
    for r in all_results:
        print(f"  {r['stage_tag']:<16} {r['dataset']:<10} attempt={r['attempt']} "
              f"passed={r['passed']} best_pp_AR={r['best_pp_ar']} "
              f"early={r['early_stopped_at']} gpu={r['gpu_avg_pct']}")
    try:
        LOCK.unlink(missing_ok=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
