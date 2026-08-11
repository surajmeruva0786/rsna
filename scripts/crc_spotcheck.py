"""CRC-32 spot-check of extracted files against the archive's stored checksums.

The full audit proves every file exists at the right size. Size equality cannot
detect content corruption, so sample entries at random and compare the on-disk
CRC-32 to the value recorded in the ZIP central directory. A full CRC pass would
mean reading 570 GB and decompressing 265 GB; a random sample catches systematic
corruption at a tiny fraction of the cost.
"""
import os
import random
import sys
import zlib
import zipfile

ZIP = r"F:\rsna\rsna-knee-abnormality-detection.zip"
DEST = r"F:\rsna\rsna-knee-abnormality-detection"
SAMPLE = 400

random.seed(20260811)

with zipfile.ZipFile(ZIP) as zf:
    infos = [i for i in zf.infolist() if not i.is_dir()]

# Always check the five CSVs, plus a random sample of DICOMs.
csvs = [i for i in infos if i.filename.lower().endswith(".csv")]
rest = [i for i in infos if not i.filename.lower().endswith(".csv")]
picks = csvs + random.sample(rest, min(SAMPLE, len(rest)))

bad = []
checked_bytes = 0
for n, info in enumerate(picks, 1):
    target = os.path.join(DEST, info.filename.replace("/", os.sep))
    crc = 0
    with open(target, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            crc = zlib.crc32(buf, crc)
    checked_bytes += info.file_size
    if crc != info.CRC:
        bad.append((info.filename, hex(info.CRC), hex(crc)))
    if n % 100 == 0:
        print(f"  {n}/{len(picks)} checked, {len(bad)} bad", flush=True)

print(f"files checked : {len(picks)} ({checked_bytes/2**30:.2f} GiB)")
print(f"csvs checked  : {len(csvs)}")
print(f"crc mismatches: {len(bad)}")
for name, want, got in bad[:20]:
    print(f"  BAD want={want} got={got} {name}")
sys.exit(1 if bad else 0)
