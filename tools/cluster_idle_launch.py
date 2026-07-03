#!/usr/bin/env python3
# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")


@dataclass
class HostScanResult:
    host: str
    launch_host: str
    idle_gpu_indices: list[int]
    busy_gpu_indices: list[int]
    gpu_rows: list[dict[str, int]]

    @property
    def idle_gpus(self) -> int:
        return len(self.idle_gpu_indices)

    @property
    def busy_gpus(self) -> int:
        return len(self.busy_gpu_indices)


def _resolve_hosts() -> list[str]:
    raw = os.getenv("PADDLE_TRAINERS") or os.getenv("TRAINER_INSTANCES") or ""
    hosts = [item.strip() for item in raw.split(",") if item.strip()]
    if not hosts:
        raise RuntimeError("Unable to resolve hosts from PADDLE_TRAINERS / TRAINER_INSTANCES")
    deduped: list[str] = []
    seen: set[str] = set()
    for host in hosts:
        if host not in seen:
            seen.add(host)
            deduped.append(host)
    return deduped


def _resolve_nvidia_smi(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("nvidia-smi")
    if found:
        return found
    return DEFAULT_NVIDIA_SMI


def _run_local_probe(*, nvidia_smi: str, util_max: int, mem_max_mib: int) -> dict[str, object]:
    cmd = [nvidia_smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    host = os.environ.get("HOSTNAME", "unknown").split(".")[0]
    launch_host = socket.gethostbyname(socket.gethostname())
    rows: list[dict[str, int | bool]] = []
    for raw in subprocess.check_output(cmd, text=True).strip().splitlines():
        if not raw.strip():
            continue
        idx, util, mem_used, mem_total = [part.strip() for part in raw.split(",")]
        row = {
            "index": int(idx),
            "utilization_gpu": int(util),
            "memory_used_mib": int(mem_used),
            "memory_total_mib": int(mem_total),
        }
        row["is_idle"] = row["utilization_gpu"] <= util_max and row["memory_used_mib"] <= mem_max_mib
        rows.append(row)
    return {"host": host, "launch_host": launch_host, "rows": rows}


def scan_hosts(
    hosts: list[str],
    *,
    util_max: int,
    mem_max_mib: int,
    nvidia_smi: str,
) -> list[HostScanResult]:
    script_path = str(Path(__file__).resolve())
    cmd = [
        "mpirun",
        "-np",
        str(len(hosts)),
        "--host",
        ",".join(hosts),
        "--map-by",
        "ppr:1:node",
        sys.executable,
        script_path,
        "_probe",
        "--nvidia-smi",
        nvidia_smi,
        "--util-max",
        str(util_max),
        "--mem-max-mib",
        str(mem_max_mib),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    results: list[HostScanResult] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "<stdout>:" in line:
            line = line.split("<stdout>:", 1)[1].strip()
        payload = json.loads(line)
        idle_gpu_indices = [row["index"] for row in payload["rows"] if row["is_idle"]]
        busy_gpu_indices = [row["index"] for row in payload["rows"] if not row["is_idle"]]
        results.append(
            HostScanResult(
                host=payload["host"],
                launch_host=payload["launch_host"],
                idle_gpu_indices=idle_gpu_indices,
                busy_gpu_indices=busy_gpu_indices,
                gpu_rows=payload["rows"],
            )
        )
    results.sort(key=lambda item: (-item.idle_gpus, item.host))
    return results


def _format_scan_table(results: list[HostScanResult]) -> str:
    lines = ["host\tlaunch_host\tidle_gpus\tbusy_gpus\tidle_gpu_indices"]
    for item in results:
        idle = ",".join(str(idx) for idx in item.idle_gpu_indices) or "-"
        lines.append(f"{item.host}\t{item.launch_host}\t{item.idle_gpus}\t{item.busy_gpus}\t{idle}")
    return "\n".join(lines)


def _choose_host(results: list[HostScanResult], min_idle_gpus: int) -> HostScanResult:
    for item in results:
        if item.idle_gpus >= min_idle_gpus:
            return item
    raise RuntimeError(f"No host has at least {min_idle_gpus} idle GPUs")


def launch_command(
    selected: HostScanResult,
    *,
    gpu_count: int,
    workdir: Path,
    command: str,
    dry_run: bool,
) -> int:
    visible = ",".join(str(idx) for idx in selected.idle_gpu_indices[:gpu_count])
    remote_command = f"cd {shlex.quote(str(workdir))} && export CUDA_VISIBLE_DEVICES={shlex.quote(visible)} && {command}"
    mpirun_cmd = [
        "mpirun",
        "-np",
        "1",
        "--host",
        selected.launch_host,
        "bash",
        "-lc",
        remote_command,
    ]
    print(f"selected_host={selected.host}")
    print(f"selected_launch_host={selected.launch_host}")
    print(f"selected_gpus={visible}")
    print("launch_command=" + shlex.join(mpirun_cmd))
    if dry_run:
        return 0
    completed = subprocess.run(mpirun_cmd)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan the 16-node queue for idle GPUs and launch an experiment")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    probe_parser = subparsers.add_parser("_probe", help=argparse.SUPPRESS)
    probe_parser.add_argument("--nvidia-smi", type=str, required=True)
    probe_parser.add_argument("--util-max", type=int, required=True)
    probe_parser.add_argument("--mem-max-mib", type=int, required=True)

    scan_parser = subparsers.add_parser("scan", help="scan queue hosts and report idle GPU counts")
    scan_parser.add_argument("--hosts", type=str, default=None, help="comma-separated host/IP list; defaults to PADDLE_TRAINERS")
    scan_parser.add_argument("--limit-hosts", type=int, default=None, help="only scan the first N resolved hosts")
    scan_parser.add_argument("--util-max", type=int, default=10, help="GPU utilization threshold for idle classification")
    scan_parser.add_argument("--mem-max-mib", type=int, default=5000, help="GPU memory-used threshold for idle classification")
    scan_parser.add_argument("--nvidia-smi", type=str, default=None, help="override nvidia-smi path")
    scan_parser.add_argument("--json", action="store_true", help="print raw JSON rows instead of a table")

    launch_parser = subparsers.add_parser("launch", help="scan queue hosts and launch on a node with enough idle GPUs")
    launch_parser.add_argument("--hosts", type=str, default=None, help="comma-separated host/IP list; defaults to PADDLE_TRAINERS")
    launch_parser.add_argument("--limit-hosts", type=int, default=None, help="only scan the first N resolved hosts")
    launch_parser.add_argument("--util-max", type=int, default=10, help="GPU utilization threshold for idle classification")
    launch_parser.add_argument("--mem-max-mib", type=int, default=5000, help="GPU memory-used threshold for idle classification")
    launch_parser.add_argument("--gpu-count", type=int, default=1, help="number of idle GPUs required and exported")
    launch_parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="working directory for the remote command")
    launch_parser.add_argument("--command", type=str, required=True, help="command to run on the selected host")
    launch_parser.add_argument("--nvidia-smi", type=str, default=None, help="override nvidia-smi path")
    launch_parser.add_argument("--dry-run", action="store_true", help="print the chosen host and mpirun command without executing it")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    nvidia_smi = _resolve_nvidia_smi(args.nvidia_smi)
    if args.subcommand == "_probe":
        print(json.dumps(_run_local_probe(nvidia_smi=nvidia_smi, util_max=args.util_max, mem_max_mib=args.mem_max_mib)))
        return 0

    hosts = [item.strip() for item in (args.hosts.split(",") if args.hosts else _resolve_hosts()) if item.strip()]
    if args.limit_hosts is not None:
        hosts = hosts[: args.limit_hosts]
    if not hosts:
        raise RuntimeError("No hosts available for scanning")

    results = scan_hosts(hosts, util_max=args.util_max, mem_max_mib=args.mem_max_mib, nvidia_smi=nvidia_smi)

    if args.subcommand == "scan":
        if args.json:
            print(json.dumps([item.__dict__ for item in results], indent=2))
        else:
            print(_format_scan_table(results))
        return 0

    selected = _choose_host(results, args.gpu_count)
    return launch_command(
        selected,
        gpu_count=args.gpu_count,
        workdir=args.workdir,
        command=args.command,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
