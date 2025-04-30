from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import rasterio.plot
import matplotlib.pyplot as plt
import gpxpy.gpx
import numpy as np
import csv
import argparse

# Parse all arguments
parser = argparse.ArgumentParser(description='Converts image into TIF with georeference. Optionally also overlays a GPX trail on top of image for test purposes.')
parser.add_argument('--input_image_path', required=True, help='Path to the input image.')
parser.add_argument('--ground_control_points_path', required=True, help='Path to the CSV of ground control points.')
parser.add_argument('--generate_gpx_overlay', default=False, action='store_true', help='Flag to generate a GPX overlay plot.')
parser.add_argument('--gpx_file_path', default=None, help='Path to the GPX file (only used with --generate_gpx_overlay).')
args = parser.parse_args()

input_image_path = args.input_image_path
ground_control_points_path = args.ground_control_points_path
generate_gpx_overlay = args.generate_gpx_overlay  # Optional: Generate pyplot with overlaid GPX trail as sanity check
gpx_file_path = args.gpx_file_path  # Only used if generate_gpx_overlay is true

# Load Ground Control Points from CSV file
gcps = []
with open(ground_control_points_path, mode='r') as file:
    reader = csv.reader(file)
    next(reader) # Skip header
    for lat, lon, x, y in reader:
        gcps.append(GroundControlPoint(row=float(y), col=float(x), x=float(lon), y=float(lat)))


# Load input image and generate georeferenced version with the GCPs
with rasterio.open(input_image_path) as src:
    image = src.read()
    meta = src.meta.copy()
    width = src.width
    height = src.height

transform = from_gcps(gcps)

meta.update({
    'driver': 'GTiff',
    'transform': transform,
    'crs': 'EPSG:4326'
})

image_path_no_file_ending = '.'.join(input_image_path.split('.')[:-1])
georeferenced_path = f'{image_path_no_file_ending}_georeferenced.tif'

with rasterio.open(georeferenced_path, 'w', **meta) as dst:
    dst.write(image)

# Optional: Load GPX file and overlay it on top of georeferenced image
# This is supposed to be a sanity check to see if GCPs result in a reasonable projection
if generate_gpx_overlay:
    gpx = gpxpy.parse(open(gpx_file_path, 'r'))

    # Extract track points
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.longitude, point.latitude))
    gpx_coords = np.array(points)

    # Figure out plot size (width and height were read and stored previously)
    my_dpi = 96
    fig, ax = plt.subplots(figsize=(width/float(my_dpi), height/float(my_dpi)))

    with rasterio.open(georeferenced_path) as src:
        bounds = src.bounds
        image = src.read()

        if image.shape[0] == 3:  # RGB
            img = np.moveaxis(image, 0, -1)
            ax.imshow(img, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        else:  # Grayscale
            ax.imshow(image[0], extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], cmap='gray')

    # Overlay the GPX track
    ax.plot(gpx_coords[:, 0], gpx_coords[:, 1], color='cyan', linewidth=5)

    # Overlay the GPX points
    ax.scatter(gpx_coords[:, 0], gpx_coords[:, 1], color='blue', s=20)

    # Turn off the axis and margins
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.show()
