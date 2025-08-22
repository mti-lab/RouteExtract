import numpy as np
from rasterio.transform import xy
from pyproj import Transformer, CRS
from scipy.spatial import cKDTree
import gpxpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataset.download_maps import load_config_file
from geo_utils import compute_affine_from_gcps, affine_matrix_to_rasterio_affine

# ------------------------------------------------------------------
# CONFIGURATION
CRS_TARGET = CRS.from_epsg(32654)   # WGS84 / UTM 54N, metres
GPX_PATH   = "../outputs/map004b_trail01_pred_extracted.gpx"
MASK_PATH  = "../dataset/masks/map004b_trail01.npz"
MANIFEST_PATH = "../dataset/maps_manifest.yaml"
MAP_ID = "map004b"
MAX_MASK_POINTS = 150_000
SHOW_PLOT = False
# ------------------------------------------------------------------

# 1) ----------------------------------------------------------------
# Load GPX and convert to UTM 54N
def load_gpx_to_utm(path, transformer):
    with open(path, "r") as f:
        gpx = gpxpy.parse(f)
    lonlat = [(p.longitude, p.latitude)
              for trk in gpx.tracks
              for seg in trk.segments
              for p   in seg.points]
    lons, lats = np.array(lonlat).T
    east, north = transformer.transform(lons, lats, direction="forward")
    return np.column_stack([east, north])

# 2) ----------------------------------------------------------------
# Load mask, extract trail-pixel world coords, convert to UTM 54N
def load_mask_to_utm(mask_path, manifest_path, map_id):
    # Load affine transform from GCPs
    manifest = load_config_file(manifest_path, return_dict=True)
    gcps = manifest[map_id]["gcps"]
    A = compute_affine_from_gcps(gcps)  # 2x3 matrix
    affine = affine_matrix_to_rasterio_affine(A)  # affine for rasterio.xy

    trail_mask = np.load(mask_path)["mask"]
    rows, cols = np.nonzero(trail_mask)

    # Convert pixel coordinates to world coordinates
    xs, ys = xy(affine, rows, cols, offset="center")

    # Transform lon/lat → UTM
    transformer = Transformer.from_crs("EPSG:4326", CRS_TARGET, always_xy=True)
    xs_utm, ys_utm = transformer.transform(xs, ys)

    coords = np.column_stack([xs_utm, ys_utm])


    if len(coords) > MAX_MASK_POINTS:
        idx = np.random.choice(len(coords), MAX_MASK_POINTS, replace=False)
        coords = coords[idx]

    return coords


# 3) ----------------------------------------------------------------
# Nearest Neighbor-based distances
def chamfer_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf
    return 0.5 *( mean_distance_A_from_B(A, B) + mean_distance_A_from_B(B, A))

def mean_distance_A_from_B(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf
    tree_B = cKDTree(B)
    dists_A = tree_B.query(A, workers=-1)[0]
    return dists_A.mean()

# ------------------------------------------------------------------

def visualize_debug(gpx_pts, mask_pts, chamfer_dist_m):
    plt.figure(figsize=(10, 10))
    plt.scatter(mask_pts[:, 0], mask_pts[:, 1], s=1, c="red", label="Trail mask")
    plt.plot(gpx_pts[:, 0], gpx_pts[:, 1], c="blue", linewidth=1.5, label="GPX route")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(f"Chamfer Distance = {chamfer_dist_m:.2f} m")
    plt.grid(True)
    plt.tight_layout()

    # Custom legend handles
    legend_elements = [
        Line2D([0], [0], color='red', label='Trail mask', linewidth=1.5),
        Line2D([0], [0], color='blue', label='GPX route', linewidth=1.5)
    ]
    plt.legend(handles=legend_elements)
    plt.show()

# MAIN
transformer_ll_to_utm = Transformer.from_crs("EPSG:4326", CRS_TARGET, always_xy=True)

gpx_pts   = load_gpx_to_utm(GPX_PATH, transformer_ll_to_utm)
mask_pts  = load_mask_to_utm(MASK_PATH, MANIFEST_PATH, MAP_ID)

chamfer_m = chamfer_distance(gpx_pts, mask_pts)
print(f"Chamfer distance: {chamfer_m:.2f} m")

if SHOW_PLOT:
    visualize_debug(gpx_pts, mask_pts, chamfer_m)

