#!/usr/bin/env python3
"""Safely, resumably extract ALARO IOP tar.gz archives.

Each archive is extracted into ALARO/<experiment>/untar-output/<basename>.
Extraction happens in a per-archive partial directory and completion is
committed with .untar_done only after the complete stream and manifest have
been written. Source archives are never modified.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath

DATE_RE = re.compile(r"(?:pf|sfx|DDH)(\d{8})$")
KINDS = {"pf": 0, "DDH": 1, "sfx": 2}


def archive_key(path: Path):
    stem = path.name[:-7] if path.name.endswith(".tar.gz") else path.stem
    m = DATE_RE.match(stem)
    date = m.group(1) if m else "99999999"
    kind = next((k for k in KINDS if stem.startswith(k)), "sfx")
    # Post-spinup windows are useful first; preserve kind priority within them.
    mmdd = date[4:]
    in_window = ("0115" <= mmdd <= "0228") or ("0729" <= mmdd <= "0831")
    return (0 if in_window else 1, KINDS[kind], date, str(path))


def safe_member_name(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return p


def extract_one(job):
    src_s, output_root_s, log_root_s = job
    src = Path(src_s)
    output_root = Path(output_root_s)
    log_root = Path(log_root_s)
    iop = src.parent.parent.name
    experiment = src.parent.name
    basename = src.name[:-7]
    # Keep each IOP run isolated; never merge same-date archives across iop1/iop2.
    destination = src.parent / "untar-output" / basename
    marker = destination / ".untar_done"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{iop}_{experiment}_{basename}.log"
    started = time.time()
    result = {"source": str(src), "destination": str(destination), "iop": iop, "experiment": experiment,
              "archive": src.name, "source_size": src.stat().st_size, "started": started}
    if marker.is_file():
        result.update(status="skipped_done", elapsed=0.0)
        with log_path.open("w") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        return result
    if destination.exists():
        # Validate an old extraction in place. No existing bytes are changed;
        # the marker is created only after every archive member is accounted for.
        try:
            members = files = member_bytes = 0
            with tarfile.open(src, mode="r|gz") as tf:
                for member in tf:
                    members += 1
                    member_bytes += max(0, member.size)
                    rel = safe_member_name(member.name)
                    target = destination.joinpath(*rel.parts)
                    if member.isdir():
                        if not target.is_dir():
                            raise FileNotFoundError(str(target))
                    elif member.isfile():
                        if not target.is_file() or target.stat().st_size != member.size:
                            raise FileNotFoundError(f"size mismatch: {target}")
                        files += 1
                    else:
                        raise ValueError(f"unsupported archive member type: {member.name!r}")
            result.update(status="validated_existing", members=members, files=files,
                          member_bytes=member_bytes, extracted_bytes=member_bytes,
                          finished=time.time())
            (destination / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            marker.write_text(json.dumps(result, sort_keys=True) + "\n")
        except Exception as exc:
            result.update(status="conflict_existing_without_marker", error=repr(exc), finished=time.time())
        with log_path.open("w") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{basename}.partial"
    partial.mkdir(exist_ok=True)
    members = files = 0
    member_bytes = extracted_bytes = 0
    unsafe = []
    try:
        with tarfile.open(src, mode="r|gz") as tf:
            for member in tf:
                members += 1
                member_bytes += max(0, member.size)
                try:
                    rel = safe_member_name(member.name)
                except ValueError as exc:
                    unsafe.append(str(exc))
                    raise
                target = partial.joinpath(*rel.parts)
                # Resolve check protects against unexpected platform behavior.
                if os.path.commonpath((str(partial.resolve()), str(target.parent.resolve()))) != str(partial.resolve()):
                    raise ValueError(f"member escapes destination: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member type: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size == member.size:
                        files += 1
                        extracted_bytes += member.size
                        continue
                    # This file is inside our own .partial directory and may
                    # be a truncated write from an interrupted run. Preserve
                    # it, then write a fresh member atomically.
                    stale_root = log_root.parent / "recovered-partial-extraction" / iop / experiment / basename
                    stale_root.mkdir(parents=True, exist_ok=True)
                    stale = stale_root / (target.name + f".stale.{os.getpid()}")
                    os.rename(target, stale)
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise OSError(f"could not read archive member: {member.name!r}")
                tmp = target.with_name(target.name + f".part.{os.getpid()}")
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(fd, "wb") as out:
                    shutil.copyfileobj(extracted, out, length=1024 * 1024)
                os.rename(tmp, target)
                files += 1
                extracted_bytes += member.size
        manifest = dict(result, status="complete", members=members, files=files,
                        member_bytes=member_bytes, extracted_bytes=extracted_bytes,
                        finished=time.time())
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Rename is atomic on the same filesystem. Never replace an existing path.
        os.rename(partial, destination)
        marker = destination / ".untar_done"
        marker.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        with log_path.open("w") as f:
            f.write(json.dumps(manifest, sort_keys=True) + "\n")
        return manifest
    except Exception as exc:
        result.update(status="failed", error=repr(exc), members=members, files=files,
                      member_bytes=member_bytes, extracted_bytes=extracted_bytes,
                      finished=time.time())
        with log_path.open("w") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/iop-runs"))
    ap.add_argument("--output-root", type=Path, default=Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO"))
    ap.add_argument("--log-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kinds", nargs="+", choices=sorted(KINDS), default=None,
                    help="restrict archive prefixes (default: all)")
    args = ap.parse_args()
    log_root = args.log_root or args.input_root / "analysis" / "logs"
    archives = sorted((p for p in args.input_root.glob("iop[12]/*/*.tar.gz")
                       if p.is_file() and (args.kinds is None or any(p.name.startswith(k) for k in args.kinds))),
                      key=archive_key)
    expected = 963 if args.kinds is None else sum(321 for _ in args.kinds)
    if len(archives) != expected:
        print(f"WARNING: discovered {len(archives)} archives; expected {expected}", flush=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = log_root / "extraction_manifest.jsonl"
    jobs = [(str(p), str(args.output_root), str(log_root)) for p in archives]
    with manifest.open("a") as mf:
        with mp.Pool(processes=max(1, args.workers)) as pool:
            for result in pool.imap_unordered(extract_one, jobs):
                mf.write(json.dumps(result, sort_keys=True) + "\n")
                mf.flush()
                print(f"{result['status']} {result['archive']} {result.get('extracted_bytes', 0)}", flush=True)


if __name__ == "__main__":
    main()
