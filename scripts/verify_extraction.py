"""Audit an extracted dataset tree against its source ZIP central directory.

Matching directory counts do not prove a complete extraction: a study directory
exists as soon as its first file lands, so an interrupted run leaves full-looking
directories with missing slices inside. This walks every entry in the archive and
checks that the file exists on disk *and* that its size matches the size recorded
in the archive, which also catches the truncated final file an interrupted
extraction leaves behind.

Usage:
    python scripts/verify_extraction.py [ZIP] [DEST] [--out DIR]

Writes verify_report.json and todo_entries.txt (entries needing re-extraction,
consumable by scripts/resume_extract.py) to --out, default the current directory.

Exit code 0 = extraction complete, 1 = incomplete.
"""
import argparse
import json
import os
import sys
import zipfile

DEFAULT_ZIP = r"F:\rsna\rsna-knee-abnormality-detection.zip"
DEFAULT_DEST = r"F:\rsna\rsna-knee-abnormality-detection"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_path", nargs="?", default=DEFAULT_ZIP)
    ap.add_argument("dest", nargs="?", default=DEFAULT_DEST)
    ap.add_argument("--out", default=".", help="where to write the report files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    missing = []         # entry absent from disk entirely
    size_mismatch = []   # present but wrong size (truncated / partial write)
    ok = 0
    dirs = 0
    total_uncompressed = 0

    print("reading central directory...", flush=True)
    with zipfile.ZipFile(args.zip_path) as zf:
        infos = zf.infolist()
        print(f"central directory entries: {len(infos)}", flush=True)

        for i, info in enumerate(infos):
            if i and i % 100000 == 0:
                print(
                    f"  checked {i}/{len(infos)}  "
                    f"missing={len(missing)} mismatch={len(size_mismatch)}",
                    flush=True,
                )
            if info.is_dir():
                dirs += 1
                continue
            total_uncompressed += info.file_size
            target = os.path.join(args.dest, info.filename.replace("/", os.sep))
            try:
                st = os.stat(target)
            except OSError:
                missing.append((info.filename, info.file_size))
                continue
            if st.st_size != info.file_size:
                size_mismatch.append((info.filename, info.file_size, st.st_size))
                continue
            ok += 1

    report = {
        "zip_path": args.zip_path,
        "dest": args.dest,
        "zip_entries_total": len(infos),
        "zip_dir_entries": dirs,
        "zip_file_entries": len(infos) - dirs,
        "extracted_ok": ok,
        "missing_count": len(missing),
        "size_mismatch_count": len(size_mismatch),
        "zip_total_uncompressed_bytes": total_uncompressed,
        "complete": not missing and not size_mismatch,
    }
    print(json.dumps(report, indent=2), flush=True)

    with open(os.path.join(args.out, "verify_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(args.out, "todo_entries.txt"), "w", encoding="utf-8") as f:
        for name, _ in missing:
            f.write(name + "\n")
        for name, _, _ in size_mismatch:
            f.write(name + "\n")

    with open(os.path.join(args.out, "missing_sample.txt"), "w", encoding="utf-8") as f:
        for name, sz in missing[:200]:
            f.write(f"MISSING  {sz:>12}  {name}\n")
        for name, want, got in size_mismatch[:200]:
            f.write(f"MISMATCH want={want} got={got}  {name}\n")

    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
