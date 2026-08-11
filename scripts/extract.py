"""Resumable extraction: audit and extract in a single pass over the archive.

For each entry in the ZIP central directory, stat the target on disk. If it
already exists at exactly the archive's recorded size, skip it; otherwise extract
it. That makes the script idempotent -- re-running after an interruption costs one
stat walk plus only the remaining bytes, never a full re-extraction of a 570 GB
tree. It also supersedes a separate verify pass, which would otherwise repeat the
same expensive stat walk over 819,640 files.

Size equality is the completeness test rather than mere existence, because an
interrupted extraction leaves a truncated final file that an existence check would
happily accept. Each file is written to a `.part` sibling and atomically renamed
into place only once fully written, so an interruption of *this* script can never
leave a half-written file for a later run to mistake for good.

Usage:
    python scripts/extract.py [--zip ZIP] [--dest DEST] [--audit-only] [--out DIR]

Exit code 0 = destination matches the archive, 1 = work remains.
"""
import argparse
import json
import os
import shutil
import sys
import time
import zipfile

DEFAULT_ZIP = r"F:\rsna\rsna-knee-abnormality-detection.zip"
DEFAULT_DEST = r"F:\rsna\rsna-knee-abnormality-detection"

CHUNK = 8 * 1024 * 1024
PROGRESS_EVERY = 5000


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", dest="zip_path", default=DEFAULT_ZIP)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--out", default=".", help="where to write report files")
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="report what is missing without writing anything",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.dest, exist_ok=True)

    print(f"archive : {args.zip_path}")
    print(f"dest    : {args.dest}")
    print("reading central directory...", flush=True)

    with zipfile.ZipFile(args.zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        total_entries = len(infos)
        total_bytes = sum(i.file_size for i in infos)
        print(
            f"entries : {total_entries}  ({human(total_bytes)} uncompressed)",
            flush=True,
        )

        # Pass 1: audit. Cheap relative to extraction and lets us report a real
        # ETA and check free space before writing the first byte.
        todo = []
        present_bytes = 0
        print("auditing what is already on disk...", flush=True)
        audit_start = time.time()
        for n, info in enumerate(infos, 1):
            if n % 100000 == 0:
                print(f"  audited {n}/{total_entries}  todo={len(todo)}", flush=True)
            target = os.path.join(args.dest, info.filename.replace("/", os.sep))
            try:
                if os.stat(target).st_size == info.file_size:
                    present_bytes += info.file_size
                    continue
            except OSError:
                pass
            todo.append(info)

        todo_bytes = sum(i.file_size for i in todo)
        report = {
            "zip_path": args.zip_path,
            "dest": args.dest,
            "entries_total": total_entries,
            "entries_present": total_entries - len(todo),
            "entries_todo": len(todo),
            "bytes_total": total_bytes,
            "bytes_present": present_bytes,
            "bytes_todo": todo_bytes,
            "complete": not todo,
            "audit_seconds": round(time.time() - audit_start, 1),
        }
        print(json.dumps(report, indent=2), flush=True)
        with open(os.path.join(args.out, "extract_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        if not todo:
            print("COMPLETE: destination matches the archive entry for entry.")
            return 0

        with open(os.path.join(args.out, "todo_entries.txt"), "w", encoding="utf-8") as f:
            for info in todo:
                f.write(info.filename + "\n")

        if args.audit_only:
            print(f"audit only -- {len(todo)} entries ({human(todo_bytes)}) outstanding")
            return 1

        avail = shutil.disk_usage(args.dest).free
        print(f"free space: {human(avail)}, need {human(todo_bytes)}", flush=True)
        if avail < todo_bytes:
            print("ABORT: insufficient free space", file=sys.stderr)
            return 2

        # Pass 2: extract only what is outstanding.
        print(f"extracting {len(todo)} entries...", flush=True)
        done_bytes = 0
        started = time.time()
        for n, info in enumerate(todo, 1):
            target = os.path.join(args.dest, info.filename.replace("/", os.sep))
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = target + ".part"
            with zf.open(info) as src, open(tmp, "wb") as dst:
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    dst.write(buf)
            os.replace(tmp, target)

            done_bytes += info.file_size
            if n % PROGRESS_EVERY == 0 or n == len(todo):
                elapsed = time.time() - started
                rate = done_bytes / elapsed if elapsed else 0
                eta = (todo_bytes - done_bytes) / rate if rate else 0
                print(
                    f"  {n}/{len(todo)}  {human(done_bytes)}/{human(todo_bytes)}  "
                    f"{human(rate)}/s  eta {eta/60:.1f} min",
                    flush=True,
                )

    print("extraction pass finished -- re-run to confirm a clean audit", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
