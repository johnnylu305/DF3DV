#!/usr/bin/env bash
set -euo pipefail

# Usage: ./check_jpg_sizes.sh [PARENT_DIR]
PARENT="${1:-.}"

command -v exiftool >/dev/null 2>&1 || {
  echo "Error: 'exiftool' not found. Install it (apt/brew) and retry." >&2
  exit 1
}

for top in "$PARENT"/*/ ; do
  [[ -d "$top" ]] || continue
  echo "== Checking folder: $top =="

  # Get sizes as "WxH" (one row per file) using CSV to keep rows intact
  mapfile -t sizes < <(
    exiftool -csv -ImageWidth -ImageHeight -ext jpg -ext JPG -ext jpeg -r "$top" \
      | awk -F, 'NR>1 && $2 != "" && $3 != "" {print $2 "x" $3}'
  )

  if [[ ${#sizes[@]} -eq 0 ]]; then
    echo "  ⚠️  No JPGs found"
    continue
  fi

  # Unique size count
  uniq_sizes=$(printf "%s\n" "${sizes[@]}" | sort -u)
  uniq_count=$(printf "%s\n" "${uniq_sizes}" | grep -c . || true)

  if [[ $uniq_count -eq 1 ]]; then
    only_size=$(printf "%s\n" "${uniq_sizes}")
    echo "  ✅ Uniform size: ${only_size}  (files: ${#sizes[@]})"
  else
    echo "  ❌ Mismatched sizes (count : WxH):"
    printf "%s\n" "${sizes[@]}" | sort | uniq -c | awk '{printf "    %6d : %s\n", $1, $2}'
  fi
done

