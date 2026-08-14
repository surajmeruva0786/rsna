#!/usr/bin/env bash
# Wait for the train preprocessing run to finish, verify it, then preprocess the test
# split and start training. Chained rather than launched by hand so training begins the
# moment the ~2.3h preprocessing pass completes instead of at the next check-in.
#
# Guarded deliberately: a truncated or failed cache must not silently become a training
# run whose results look plausible but mean nothing.
set -uo pipefail

CACHE="${RSNA_CACHE:-C:/rsna_cache}"
LOG="$CACHE/prep_train.log"
EXPECTED_STUDIES=4407

# Wait on a line preprocess.py itself prints, not on an exit code appended by whatever
# wrapper launched it. An earlier version waited for an "EXIT=" sentinel written by the
# calling shell; when that shell was killed the sentinel never arrived, and the chain sat
# waiting forever even though preprocessing had completed successfully 2 hours earlier.
echo "[chain] waiting for preprocessing to finish..."
while ! grep -q "^total .* min" "$LOG" 2>/dev/null; do
  sleep 30
done
echo "[chain] preprocessing reported completion"

if grep -qE "Traceback|Error" "$LOG"; then
  echo "[chain] ABORT: preprocessing log contains errors"
  exit 1
fi

rows=$(python -c "import pandas as pd,os;print(len(pd.read_csv(os.environ['CACHE_DIR']+'/train_index.csv')))" 2>/dev/null)
echo "[chain] index rows: $rows (expected $EXPECTED_STUDIES)"
if [ "$rows" != "$EXPECTED_STUDIES" ]; then
  echo "[chain] ABORT: index row count mismatch"
  exit 1
fi

echo "[chain] preprocessing test split..."
python scripts/preprocess.py --split test --workers 6 > "$CACHE/prep_test.log" 2>&1
echo "[chain] test preprocessing exit=$?"

# Fold 0 only, to measure real epoch time on a shared GPU before committing to a full
# 5-fold budget. The checkpoint it writes is already usable for a submission.
echo "[chain] starting training (fold 0)..."
python scripts/train.py \
  --folds 5 --only-fold 0 --epochs 10 \
  --batch 3 --accum 5 --workers 3 \
  --out run1 > "$CACHE/train_run1.log" 2>&1
echo "[chain] training exit=$?"
