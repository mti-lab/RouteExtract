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
from scipy.spatial import KDTree
from mask_to_graph import generate_path, extract_graph_with_cycles, bfs_no_revisiting
from skimage.morphology import skeletonize

def load_coordinates_from_csv(file_path):
    coordinates = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            lat, lon = float(row[0]), float(row[1])
            coordinates.append((lat, lon))
    return coordinates

def get_route_as_gpx(stops, route_filename='route.gpx'):
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
        with open(f'../outputs/{route_filename}', 'wb') as f:
            f.write(response.content)
        print(f"GPX file saved as '{route_filename}'.")
        return response.content
    else:
        # Log the response for debugging
        print(f"Error: Unable to fetch route. Status code: {response.status_code}")
        print(f"Response content: {response.text}")

def convert_gpx_to_pixel_coordinates(gpx_data):
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
    return pixel_coords

def evaluate_from_mask(gpx_data, sampling_rate=1.0):
    pixel_coords = convert_gpx_to_pixel_coordinates(gpx_data)

    # Build KD-tree for fast nearest neighbor lookup
    tree = KDTree(pixel_coords)

    # Get all trail pixels
    trail_yx = np.argwhere(skeleton == 1)  # (row, col)
    trail_pixels = np.array([(x, y) for y, x in trail_yx])  # shape (M, 2)

    # Query nearest GPX point for each trail pixel
    distances, indices = tree.query(trail_pixels)
    max_idx = np.argmax(distances)

    # Extract worst-matching trail pixel and its nearest GPX point
    x_trail, y_trail = trail_pixels[max_idx]
    x_gpx, y_gpx = pixel_coords[indices[max_idx]]

    # Convert both to geo-coordinates
    lon_trail, lat_trail = xy(transform, y_trail, x_trail)
    lon_gpx, lat_gpx = xy(transform, y_gpx, x_gpx)
    print(f"Trail pixel with largest distance from GPX: pixel ({x_trail}, {y_trail})")
    print(f"Nearest GPX pixel: ({x_gpx}, {y_gpx})")
    print(f"Pixel distance: {distances[max_idx]:.2f}")
    print("\nGround Truth Trail Point (Geo):", (lat_trail, lon_trail))
    print("Nearest GPX Point (Geo):", (lat_gpx, lon_gpx))
    return (float(lat_trail), float(lon_trail), int(x_trail), int(y_trail))


def evaluate_from_gpx(gpx_data, top_n=20):
    pixel_coords = convert_gpx_to_pixel_coordinates(gpx_data)

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
    top_indices = np.argsort(-distances)[:top_n]  # Largest distances first

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.imshow(map_img[..., ::-1])
    plt.axis('off')

    # Plot all GPX points (small dots)
    plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c='green', s=15, label='GPX Trail')

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


def find_nearest_point(point, point_list):
    """Find nearest point in point_list to the given point."""
    tree = KDTree(point_list)
    dist, idx = tree.query(point)
    return point_list[idx]


def find_connection_nodes(graph, start, known_points, max_steps=1000):
    """Walk from 'start' in both directions until we find two known points."""
    visited = set()
    frontier = [(start, [])]
    found_points = []

    while frontier and len(found_points) < 2:
        current, path = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)

        # If we find a known point, save it
        if current in known_points:
            found_points.append(current)

        for neighbor in graph.neighbors(current):
            if neighbor not in visited and len(path) < max_steps:
                frontier.append((neighbor, path + [current]))

    if len(found_points) == 2:
        return found_points
    else:
        return None  # Didn't find both points


def insert_point_by_connectivity(graph, known_image_points, c_image_point):
    # Map known points for quick lookup
    known_set = set(map(tuple, known_image_points))

    # Walk from the projected point in both directions to find two known points
    connection_points = find_connection_nodes(graph, tuple(c_image_point), known_set, max_steps=5000)

    if connection_points:
        hit1, hit2 = connection_points
        idx1 = known_image_points.index(hit1)
        idx2 = known_image_points.index(hit2)

        # Insert c between hit1 and hit2
        insert_idx = min(idx1, idx2) + 1
        return insert_idx
    else:
        return None  # Couldn't find both ends


def linear_sample_with_edges(data, fraction):
    n = len(data)
    if n < 2:
        return data[:]  # Too short to sample meaningfully

    sample_size = max(2, int(n * fraction))

    # Generate linearly spaced indices from 0 to n-1
    indices = np.linspace(0, n - 1, num=sample_size, dtype=int)
    return [data[i] for i in indices]



### One-time setup: Load georeferenced image, regular image, trail mask
# Open the files
map_georeferenced = rasterio.open('../Data/Minami_Senju/scan-georeferenced.tif')
map_img = cv2.imread('../Data/Minami_Senju/scan.jpg')
trail_mask = cv2.imread('../Data/Minami_Senju/scan-mask.png', cv2.IMREAD_GRAYSCALE)
trail_mask = (trail_mask == 0).astype(np.uint8)
skeleton = skeletonize(trail_mask > 0)
csv_file_path = '../Data/Minami_Senju/scan-coordinates.csv'
gpx_filename = 'route.gpx'
no_iterations = 10  # Number of iterations of refining the map (currently still set manually and not based on metrics)
top_n = 20  # Number of selected farthest points for finding new points of the route

G = extract_graph_with_cycles(trail_mask)

# Get the Affine transform (maps pixel coords <-> world coords)
transform = map_georeferenced.transform
start_and_end = load_coordinates_from_csv(csv_file_path)
start_and_end = [~transform * (lon, lat) for (lat, lon) in start_and_end]
start_and_end = [(y, x) for (x, y) in start_and_end]
path_edges = generate_path(trail_mask, start=start_and_end[0], end=start_and_end[1])

visited_nodes = set()
visited_edges = set()
edge_trails = dict()
for edge in path_edges:
    if edge in visited_edges:
        continue

    edge_trail, visited_nodes = bfs_no_revisiting(G, start=edge[0], end=edge[1], visited=visited_nodes)
    if edge_trail:
        edge_trail = linear_sample_with_edges(edge_trail, 0.005)
        edge_trails[edge] = edge_trail
        edge_trails[(edge[1], edge[0])] = list(reversed(edge_trail))
        visited_edges.add(edge)
        visited_edges.add((edge[1], edge[0]))

stops_pixel_coords = []
for edge in path_edges:
    edge_trail = edge_trails.get(edge)
    if edge_trail:
        stops_pixel_coords = stops_pixel_coords + edge_trails[edge]
    else:
        stops_pixel_coords = stops_pixel_coords + [edge[0], edge[1]]
stops_lat_lon = [xy(transform, x, y) for x, y in stops_pixel_coords]
stops_lat_lon = [(lat, lon) for (lon, lat) in stops_lat_lon]

for _ in range(no_iterations):
    gpx = gpxpy.parse(get_route_as_gpx(stops_lat_lon, gpx_filename))
    lat, lon, x, y = evaluate_from_mask(gpx)
    insert_point_min_distance(stops_lat_lon, (lat, lon))
