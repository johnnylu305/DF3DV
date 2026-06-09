import os
import sys
import argparse
from pathlib import Path

# ---------- config ----------
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
RECURSIVE = True  # scan subfolders

# ---------- helpers ----------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS

def iter_images(root: Path):
    if RECURSIVE:
        yield from (p for p in root.rglob("*") if is_image(p))
    else:
        yield from (p for p in root.iterdir() if is_image(p))

def ensure_unique(dst: Path) -> Path:
    """If dst exists, append _dupN before extension to avoid overwrite."""
    if not dst.exists():
        return dst
    stem, suf = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}_dup{i}{suf}")
        if not cand.exists():
            return cand
        i += 1

def make_hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return True, None
    except OSError as e:
        return False, e

def link_folder(src_root: Path, human_label: str, prefix: str, c_root: Path):
    """
    Create hard links in C for all images under src_root,
    naming them with the given prefix + original basename.
    """
    created = skipped = renamed = 0
    for src in iter_images(src_root):
        dst_name = prefix + src.name  # only basename, prefixed
        dst = c_root / dst_name

        if dst.exists():
            dst = ensure_unique(dst)
            renamed += 1

        ok, err = make_hardlink(src, dst)
        if ok:
            created += 1
        else:
            skipped += 1
            print(f"[SKIP {human_label}] {src} -> {dst}  ({err})")

    print(f"[{human_label}] created: {created}, renamed(_dupN): {renamed}, skipped: {skipped}")

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Link Clean and Clutter images into a combined All folder."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory (e.g. /home/.../230825-MascotDrawing)"
    )
    args = parser.parse_args()

    base = args.base_dir.resolve()
    dir_a = base / f"{base.name}-Clean" / "images"
    dir_b = base / f"{base.name}-Clutter" / "images"
    dir_c = base / f"{base.name}-All" / "images"

    missing = [p for p in (dir_a, dir_b) if not p.exists()]
    if missing:
        for m in missing:
            print(f"[Error] Path not found: {m}", file=sys.stderr)
        sys.exit(1)

    dir_c.mkdir(parents=True, exist_ok=True)

    print(f"Linking from:\n  Clean(A)={dir_a}\n  Clutter(B)={dir_b}\ninto C={dir_c}\n")
    link_folder(dir_a, human_label="Clean",   prefix="extra_",   c_root=dir_c)
    link_folder(dir_b, human_label="Clutter", prefix="clutter_", c_root=dir_c)
    print(f"\n[Done] Output folder: {dir_c}")

if __name__ == "__main__":
    main()

