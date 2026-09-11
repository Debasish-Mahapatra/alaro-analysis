#!/usr/bin/env python3
"""Wait for extraction parents, audit all archive markers, and retry gaps."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def audit(root):
    rows = []
    for src in sorted(root.glob("iop[12]/*/*.tar.gz")):
        exp = src.parent.name
        dest = src.parent / "untar-output" / src.name[:-7]
        marker = dest / ".untar_done"
        status = "missing"
        if marker.is_file():
            try:
                data = json.loads(marker.read_text())
                status = "complete" if data.get("source") == str(src) and data.get("status") in ("complete", "validated_existing") else "bad_provenance"
            except Exception as exc:
                status = "bad_marker:" + repr(exc)
        kind = "DDH" if src.name.startswith("DDH") else ("sfx" if src.name.startswith("sfx") else "pf")
        rows.append({"source": str(src), "destination": str(dest), "kind": kind, "experiment": exp, "status": status})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/iop-runs"))
    ap.add_argument("--log-root", type=Path, default=None)
    ap.add_argument("--pids", nargs="*", type=int, default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()
    log = args.log_root or args.root / "analysis" / "logs"
    log.mkdir(parents=True, exist_ok=True)
    extractor = Path(__file__).with_name("extract_iop_archives.py")
    while any(alive(pid) for pid in args.pids):
        time.sleep(60)
    for attempt in range(args.max_retries + 1):
        rows = audit(args.root)
        missing = [r for r in rows if r["status"] != "complete"]
        report = {"attempt": attempt, "expected": len(rows), "complete": len(rows) - len(missing), "missing": missing, "time": time.time()}
        (log / "extraction_audit_latest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: report[k] for k in ("attempt", "expected", "complete")}, sort_keys=True), flush=True)
        if not missing or attempt == args.max_retries:
            (log / "extraction_audit_final.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return 0 if not missing else 1
        kinds = sorted(set(r["kind"] for r in missing))
        cmd = ["python3", "-u", str(extractor), "--workers", str(args.workers), "--kinds"] + kinds
        print("retry", attempt + 1, kinds, flush=True)
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
