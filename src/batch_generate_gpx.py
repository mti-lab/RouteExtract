import subprocess
from pathlib import Path
import os


MANIFEST_PATH = Path("../dataset/maps_manifest.yaml")
SCRIPT_PATH = Path("generate_gpx.py")
GT_MASK_DIR = Path("../dataset/masks_testset")
PRED_MASK_DIR = Path("../outputs/predicted_masks_npz")

REPO_ROOT = Path(__file__).resolve().parent.parent
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")

# All combinations of the three flags (excluding invalid ones)
flag_combinations = [
    {"e2e": True, "gp": False, "ir": False},
    {"e2e": False, "gp": True, "ir": False},
    {"e2e": True, "gp": True, "ir": False},
    {"e2e": True, "gp": False, "ir": True},
    {"e2e": False, "gp": True, "ir": True},
    {"e2e": True, "gp": True, "ir": True},
]

# Sources: "gt" or "pred"
mask_sources = {
    "gt": GT_MASK_DIR,
    "pred": PRED_MASK_DIR,
}

for source_name, mask_dir in mask_sources.items():
    for flags in flag_combinations:
        if not (flags["e2e"] or flags["gp"]):
            continue  # Skip invalid config

        # Compose folder name
        config_name = "_".join([k for k, v in flags.items() if v])
        output_dir = Path(f"../outputs/gpx_{config_name}_{source_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for mask_path in sorted(mask_dir.glob("*.npz")):
            base = mask_path.stem  # e.g., map005_trail01
            map_id, trail_id_str = base.split("_")
            trail_id = int(trail_id_str[5:])  # strip 'trail'

            gpx_filename = output_dir / f"{map_id}_trail{trail_id:02d}_extracted.gpx"

            args = [
                "python", str(SCRIPT_PATH),
                "--manifest_path", str(MANIFEST_PATH),
                "--map_id", map_id,
                "--trail_id", str(trail_id),
                "--mask_path", str(mask_path),
                "--gpx_filepath", str(gpx_filename),
                "--no_iterations", "100",
                "--top_n", "20"
            ]

            if flags["e2e"]:
                args.append("--end_to_end")
            if flags["gp"]:
                args.append("--graph_processing")
            if flags["ir"]:
                args.append("--iterative_refinement")

            print(f"[RUN] {config_name}_{source_name} -> {base}")
            subprocess.run(args, check=True, env=env)

print("All GPX files generated for all configurations.")