# This script finds scenes without metadata and interactively generates
# corresponding scene-level JSON files. It scans scene folders under the given
# root, creates Meta/<root>/ if needed, skips existing metadata files, and asks
# the user to input environment, theme, distractors, scenario labels,
# view-dependent status, and low-quality status.
# ex: python label_meta.py --root 0000 --meta Meta
from pathlib import Path
import argparse
import json


SCENARIOS = [
    "color-similar distractors",
    "fluid distractors",
    "frontal occlusion distractors",
    "highly reflective distractors",
    "large-scale distractors",
    "local air distractors",
    "local appearance distractors",
    "semantically similar distractors",
    "semi-transparent distractors",
    "semi-transient distractors",
    "shadow distractors",
    "slow-motion distractors",
    "various distractors",
    "common distractors as static parts",
    "daily scenes",
    "nighttime scenes",
    "other distractors/scenes",
]



def parse_scene_name(scene_id):
    # Example: 040625-LundoBin -> LundoBin
    if "-" in scene_id:
        return scene_id.split("-", 1)[1]
    return scene_id


def ask_environment():
    while True:
        text = input("environment (i=indoor, o=outdoor): ").strip().lower()

        if text == "i":
            return "indoor"
        if text == "o":
            return "outdoor"

        print("Invalid input. Please enter i or o.")


def ask_list(prompt):
    text = input(f"{prompt} (comma-separated): ").strip()
    if not text:
        return [""]
    return [x.strip() for x in text.split(",") if x.strip()]


def ask_bool(prompt):
    while True:
        text = input(f"{prompt} (0=False, 1=True): ").strip()

        if text == "0":
            return ["False"]
        if text == "1":
            return ["True"]

        print("Invalid input. Please enter 0 or 1.")


def ask_scenario(prompt, allow_empty=False):
    print(f"\n{prompt}")

    if allow_empty:
        print("  -1: empty")

    for i, scenario in enumerate(SCENARIOS):
        print(f"  {i}: {scenario}")

    while True:
        text = input("Select scenario index: ").strip()

        if allow_empty and text == "-1":
            return [""]

        if text.isdigit():
            idx = int(text)
            if 0 <= idx < len(SCENARIOS):
                return [SCENARIOS[idx]]

        print("Invalid input. Please select a valid index.")


def create_meta_for_scene(scene_dir, meta_path):
    scene_id = scene_dir.name
    scene_name = parse_scene_name(scene_id)

    print("\n" + "=" * 80)
    print(f"Creating metadata for scene: {scene_id}")
    print(f"Scene name: {scene_name}")
    print("=" * 80)

    environment = ask_environment()
    theme = ask_list("theme")
    distractors = ask_list("distractors")
    scenario = ask_scenario("scenario", allow_empty=False)
    secondary_scenario = ask_scenario("secondary_scenario", allow_empty=True)
    view_dependent = ask_bool("view_dependent")
    low_quality = ask_bool("low_quality")

    metadata = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "environment": environment,
        "theme": theme,
        "distractors": distractors,
        "scenario": scenario,
        "secondary_scenario": secondary_scenario,
        "view_dependent": view_dependent,
        "low_quality": low_quality,
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root folder, e.g., 0000",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="Meta",
        help="Meta root folder, e.g., Meta",
    )

    args = parser.parse_args()

    root = Path(args.root)
    meta_root = Path(args.meta)

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    target_meta_root = meta_root / root.name
    target_meta_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {root}")
    print(f"Meta root: {target_meta_root}")

    created = 0
    skipped = 0

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_id = scene_dir.name
        meta_path = target_meta_root / f"{scene_id}_meta.json"

        if meta_path.exists():
            print(f"Skip existing meta: {meta_path}")
            skipped += 1
            continue

        create_meta_for_scene(scene_dir, meta_path)
        created += 1

    print("\nDone.")
    print(f"Created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Metadata saved under: {target_meta_root}")


if __name__ == "__main__":
    main()
