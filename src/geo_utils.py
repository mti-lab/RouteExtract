import numpy as np
from rasterio.transform import Affine, xy

def compute_affine_from_gcps(gcps):
    """
    Compute an affine transform (2x3) from a list of GCPs in the format:
    { "pixel": [x, y], "coord": [lon, lat] }
    """
    src = np.array([g["pixel"] + [1] for g in gcps])  # Nx3
    dst = np.array([g["coord"] for g in gcps])        # Nx2

    if src.shape[0] < 3:
        raise ValueError("At least 3 GCPs are required for an affine transform.")

    A, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)  # solve: src @ A.T ≈ dst
    return A.T  # shape (2, 3)

def affine_matrix_to_rasterio_affine(A):
    """
    Converts a 2x3 numpy matrix into a rasterio Affine transform.
    """
    return Affine(A[0,0], A[0,1], A[0,2],
                  A[1,0], A[1,1], A[1,2])

# --- Coordinate conversion functions ---

def pixel_to_geo(x, y, transform):
    """
    Convert image pixel coordinates (x, y) to geographic coordinates (lon, lat).
    """
    lon, lat = xy(transform, y, x)
    return lon, lat

def geo_to_pixel(lon, lat, transform):
    """
    Convert geographic coordinates (lon, lat) to pixel coordinates (x, y).
    """
    inv_transform = ~transform
    x, y = inv_transform * (lon, lat)
    return x, y
