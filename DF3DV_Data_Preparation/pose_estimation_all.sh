#!/usr/bin/env bash
set -euo pipefail

ok_scenes=()
bad_scenes=()

for d in */ ; do
  [[ -d "$d" ]] || continue
  BASE="${d%/}"
  echo "=== Processing $BASE ==="

  # Run the per-scene script; do NOT let a failure stop the loop
  if ./pose_estimation.sh "$BASE"; then
    ok_scenes+=("$BASE")
  else
    echo "[WARN] $BASE failed (exit $?) — continuing..."
    bad_scenes+=("$BASE")
  fi
done

echo
echo "================ SUMMARY ================"
echo "Succeeded: ${#ok_scenes[@]}"
for s in "${ok_scenes[@]}"; do echo "  ✔ $s"; done
echo "Failed:    ${#bad_scenes[@]}"
for s in "${bad_scenes[@]}"; do echo "  ✖ $s"; done

# Return non-zero if any failed (optional)
(( ${#bad_scenes[@]} == 0 )) || exit 1

