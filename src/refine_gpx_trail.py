import rasterio
import gpxpy
from rasterio.transform import rowcol, xy
import numpy as np
import cv2
import csv
import requests
import matplotlib.pyplot as plt
import random
import math

def load_coordinates_from_csv(file_path):
    coordinates = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            lat, lon = float(row[0]), float(row[1])
            coordinates.append((lat, lon))
    return coordinates

def get_route_as_gpx(stops):
    # URL of GraphHopper instance
    graphhopper_url = "http://localhost:8989/route"

    # Convert the list of stops into the required format for GraphHopper API
    points = [f"{lat},{lon}" for lat, lon in stops]

    # Parameters for the routing request
    params = {
        'point': points,  # List of points as latitude,longitude strings
        'type': 'gpx',  # Request for GPX format
        'profile': 'foot',  # Vehicle type (can be car, bike, foot, etc.)
        'elevation': 'true',  # Use elevation data
    }

    # Print the full URL for debugging
    print(f"Requesting URL: {graphhopper_url}")
    print(f"Request parameters: {params}")

    # Sending the request to GraphHopper
    response = requests.get(graphhopper_url, params=params)

    # Check if the response was successful
    if response.status_code == 200:
        # Save the GPX file to disk
        with open('../route.gpx', 'wb') as f:
            f.write(response.content)
        print("GPX file saved as 'route.gpx'.")
        return response.content
    else:
        # Log the response for debugging
        print(f"Error: Unable to fetch route. Status code: {response.status_code}")
        print(f"Response content: {response.text}")


def evaluate_gpx(gpx_data):
    # Extract all points
    gpx_points = []
    for track in gpx_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                gpx_points.append((point.longitude, point.latitude))

    # Transform GPX points to image coordinates
    # Convert (lon, lat) to (row, col) in the image
    pixel_coords = [rowcol(transform, lon, lat) for lon, lat in gpx_points]

    # Remember: row is y, col is x
    pixel_coords = [(col, row) for row, col in pixel_coords]

    # Compare GPX Mask with Segmented Trail
    gpx_mask = np.zeros_like(trail_mask)

    for x, y in pixel_coords:
        cv2.circle(gpx_mask, (x, y), radius=2, color=1, thickness=-1)

    # Compute Distances
    # Compute distance transform of the trail
    dist_transform = cv2.distanceTransform(1 - trail_mask, distanceType=cv2.DIST_L2, maskSize=5)

    # Sample the distance at each GPX point
    distances = []
    for x, y in pixel_coords:
        if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
            distances.append(dist_transform[y, x])
        else:
            distances.append(-1)

    # Find points with largest distances
    distances = np.array(distances)
    pixel_coords = np.array(pixel_coords)

    # Threshold or select top N mismatches
    top_n = 20
    top_indices = np.argsort(-distances)[:top_n]  # Largest distances first

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.imshow(map_img)
    plt.axis('off')

    # Plot all GPX points (small dots)
    plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c='cyan', s=5, label='GPX Trail')

    # Plot top N farthest points (larger red markers)
    plt.scatter(pixel_coords[top_indices, 0], pixel_coords[top_indices, 1],
                c='red', s=50, marker='x', label=f'Top {top_n} Farthest Points')

    plt.legend()
    plt.show()

    # Get indices of top N largest distances
    distances = np.array(distances)
    valid_indices = np.where(distances >= 0)[0]
    sorted_indices = valid_indices[np.argsort(-distances[valid_indices])]
    top_indices = sorted_indices[:top_n]

    # Choose one at random
    chosen_idx = random.choice(top_indices)
    x_chosen, y_chosen = pixel_coords[chosen_idx]
    print(
        f"Randomly selected from top {top_n} mismatches: index {chosen_idx}, pixel: ({x_chosen}, {y_chosen}), distance: {distances[chosen_idx]:.2f}")

    # Find nearest trail pixel
    trail_yx = np.argwhere(trail_mask == 1)
    trail_coords = np.array([(x, y) for y, x in trail_yx])
    dists = np.linalg.norm(trail_coords - np.array([[x_chosen, y_chosen]]), axis=1)
    nearest_idx = np.argmin(dists)
    x_nearest, y_nearest = trail_coords[nearest_idx]

    # Convert both to geo coordinates
    lon_chosen, lat_chosen = xy(transform, y_chosen, x_chosen)
    lon_nearest, lat_nearest = xy(transform, y_nearest, x_nearest)

    print("\nRandom GPX Point (Geo):", (lat_chosen, lon_chosen))
    print("Nearest Trail Point (Geo):", (lat_nearest, lon_nearest))
    return (float(lat_nearest), float(lon_nearest))


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Compute the perpendicular distance from (px, py) to the segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment is a single point
        return math.hypot(px - x1, py - y1)

    # Project point onto the segment, computing the parameterized position t
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # clamp t to the segment

    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def insert_point_min_distance(points, new_point):
    """Insert new_point into the list `points` at the position minimizing the distance to a segment."""
    if len(points) < 2:
        points.append(new_point)
        return points

    min_distance = float('inf')
    best_index = None

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dist = point_to_segment_distance(new_point[0], new_point[1], x1, y1, x2, y2)
        if dist < min_distance:
            min_distance = dist
            best_index = i + 1

    points.insert(best_index, new_point)
    return points



### One-time setup: Load georeferenced image, regular image, trail mask
# Open the files
map_georeferenced = rasterio.open('../kobo-mask_georeferenced.tif')
map_img = cv2.imread('../test-data/kobo.jpg')
trail_mask = cv2.imread('../test-data/kobo-mask.png', cv2.IMREAD_GRAYSCALE)
trail_mask = (trail_mask == 0).astype(np.uint8)
csv_file_path = '../test-data/kobo-coordinates.csv'
no_iterations = 10  # Number of iterations of refining the map (currently still set manually and not based on metrics)
top_n = 20  # Number of selected farthest points for finding new points of the route

# Get the Affine transform (maps pixel coords <-> world coords)
transform = map_georeferenced.transform
stops = load_coordinates_from_csv(csv_file_path)

for _ in range(no_iterations):
    gpx = gpxpy.parse(get_route_as_gpx(stops))
    new_stop = evaluate_gpx(gpx)
    insert_point_min_distance(stops, new_stop)
