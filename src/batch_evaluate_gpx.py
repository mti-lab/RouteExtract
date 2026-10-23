import csv
from pathlib import Path
from evaluate_gpx import load_gpx_to_utm, load_mask_to_utm, chamfer_distance, CRS_TARGET, visualize_debug, mean_distance_A_from_B
from dataset.download_maps import load_config_file
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt

# Paths (adjust as needed)
PLOT_PATH = Path("../outputs/chamfer_histogram.png")
OUTPUT_ROOT = Path("../outputs")
MASK_DIR = Path("../dataset/masks")  # Ground truth masks
MANIFEST_PATH = Path("../dataset/maps_manifest.yaml")
OUTPUT_CSV = Path("../outputs/evaluation_all_configs.csv")
PIPELINE_CONFIG = "gpx_e2e_gp_ir_pred"
SKIP_APPROXIMATE_GCPS = True  # Skip maps marked "gcp_quality: approximate" (not drawn to scale, GCPs only roughly fit)

# Transformer
transformer = Transformer.from_crs("EPSG:4326", CRS_TARGET, always_xy=True)
manifest = load_config_file(MANIFEST_PATH, return_dict=True)

def merge_results_to_csv(results, output_csv):
    data = {}  # key = mask_file, value = dict of fields

    # Load existing CSV if it exists
    try:
        with open(output_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["mask_file"]] = row
    except FileNotFoundError:
        pass

    # Merge in new results
    for mask_file, label_suffix, chamfer, d_g2m, d_m2g in results:
        if mask_file not in data:
            data[mask_file] = {"mask_file": mask_file}
        data[mask_file][f"chamfer_{label_suffix}"]     = f"{chamfer:.2f}"
        data[mask_file][f"gpx_to_mask_{label_suffix}"] = f"{d_g2m:.2f}"
        data[mask_file][f"mask_to_gpx_{label_suffix}"] = f"{d_m2g:.2f}"

    # Define field order
    ordered_fields = ["mask_file"]
    example_row = next(iter(data.values()), {})

    # Collect unique suffixes from the keys
    suffixes = set()
    for key in example_row:
        if key == "mask_file":
            continue
        if key.startswith("chamfer_"):
            suffixes.add(key[len("chamfer_"):])
        elif key.startswith("gpx_to_mask_"):
            suffixes.add(key[len("gpx_to_mask_"):])
        elif key.startswith("mask_to_gpx_"):
            suffixes.add(key[len("mask_to_gpx_"):])

    for suffix in sorted(suffixes):
        ordered_fields += [
            f"chamfer_{suffix}",
            f"gpx_to_mask_{suffix}",
            f"mask_to_gpx_{suffix}",
        ]

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(data.values())

    print(f"[✓] Merged results written to {output_csv}")

results = []

for gpx_dir in sorted(OUTPUT_ROOT.glob(PIPELINE_CONFIG)):
    label_suffix = gpx_dir.name[4:]  # Strip "gpx_" prefix
    print(f"\n[==] Evaluating config: {label_suffix}")

    for gpx_file in sorted(gpx_dir.glob("*.gpx")):
        try:
            base = gpx_file.stem  # e.g., map005_trail01_extracted
            parts = base.split("_")
            map_id = parts[0]
            trail_id = parts[1]
            mask_file = MASK_DIR / f"{map_id}_{trail_id}.npz"

            if not mask_file.exists():
                print(f"[SKIP] No GT mask for {gpx_file.name}")
                continue

            if SKIP_APPROXIMATE_GCPS and manifest[map_id].get("gcp_quality") == "approximate":
                print(f"[SKIP] Approximate georeferencing for {gpx_file.name}")
                continue

            gpx_pts = load_gpx_to_utm(gpx_file, transformer)
            mask_pts = load_mask_to_utm(mask_file, MANIFEST_PATH, map_id)

            distance_chamfer = chamfer_distance(gpx_pts, mask_pts)
            distance_gpx_to_mask = mean_distance_A_from_B(gpx_pts, mask_pts)
            distance_mask_to_gpx = mean_distance_A_from_B(mask_pts, gpx_pts)
            visualize_debug(gpx_pts, mask_pts, distance_chamfer)

            results.append((mask_file.name, label_suffix, distance_chamfer, distance_gpx_to_mask, distance_mask_to_gpx))
            print(f"[OK] {gpx_file.name} → {distance_chamfer:.2f} m")

        except Exception as e:
            print(f"[ERROR] {gpx_file.name}: {e}")

merge_results_to_csv(results, OUTPUT_CSV)

print(f"Results written to {OUTPUT_CSV}")


# Statistics
distances = np.array([r[2] for r in results])
mean_d = np.mean(distances)
std_d = np.std(distances)
min_d = np.min(distances)
max_d = np.max(distances)
median_d = np.median(distances)

print("\n📊 Summary Statistics:")
print(f"Mean Chamfer Distance   : {mean_d:.2f} m")
print(f"Median Chamfer Distance : {median_d:.2f} m")
print(f"Standard Deviation      : {std_d:.2f} m")
print(f"Min Distance            : {min_d:.2f} m")
print(f"Max Distance            : {max_d:.2f} m")

# Histogram
plt.figure(figsize=(10, 5))
plt.hist(distances, bins=20, color="skyblue", edgecolor="black")
plt.axvline(mean_d, color="red", linestyle="--", label=f"Mean: {mean_d:.2f} m")
plt.axvline(median_d, color="green", linestyle="--", label=f"Median: {median_d:.2f} m")
plt.xlabel("Chamfer Distance (m)")
plt.ylabel("Frequency")
plt.title("Chamfer Distance Histogram")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

print(f"\n📉 Histogram saved to: {PLOT_PATH}")