import gpxpy
import numpy as np
import cv2
import requests
import random
import math
from scipy.spatial import KDTree
from trail_graph import generate_path, extract_graph_with_cycles, astar_no_revisiting
import argparse
from geo_utils import compute_affine_from_gcps, affine_matrix_to_rasterio_affine, pixel_to_geo, geo_to_pixel
from dataset.download_maps import load_config_file
import sys

def get_route_as_gpx(stops, route_filepath='route.gpx'):
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
        with open(route_filepath, 'wb') as f:
            f.write(response.content)
        print(f"GPX file saved under '{route_filepath}'.")
        return response.content
    else:
        # Log the response for debugging
        print(f"Error: Unable to fetch route. Status code: {response.status_code}")
        print(f"Response content: {response.text}")

def convert_gpx_to_pixel_coordinates(gpx_data, transform=None):
    gpx_points = []
    for track in gpx_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                gpx_points.append((point.longitude, point.latitude))

    return [geo_to_pixel(lon, lat, transform) for lon, lat in gpx_points]


def create_gpx_mask(gpx_coords, mask_shape):
    """
    Create a binary mask from GPX coordinates including both points and connecting lines.

    Args:
        gpx_coords: Nx2 array/list of GPX points (x, y) in image/map space
        mask_shape: Tuple of (height, width) for the output mask

    Returns:
        Binary mask with GPX track drawn
    """
    gpx_mask = np.zeros(mask_shape, dtype=np.uint8)

    # Convert coordinates to integer pixels
    pixels = np.array(gpx_coords).astype(np.int32)

    # Draw lines between consecutive points
    for i in range(len(pixels) - 1):
        pt1 = (pixels[i][0], pixels[i][1])
        pt2 = (pixels[i + 1][0], pixels[i + 1][1])
        cv2.line(gpx_mask, pt1, pt2, color=1, thickness=1)

    # Draw points
    for x, y in pixels:
        cv2.circle(gpx_mask, (x, y), radius=1, color=1, thickness=-1)

    return gpx_mask


def evaluate_from_mask(gpx_data, trail_mask, transform=None, sampling_rate=1.0):
    pixel_coords = convert_gpx_to_pixel_coordinates(gpx_data, transform)

    # Build KD-tree for fast nearest neighbor lookup
    tree = KDTree(pixel_coords)

    # Get all trail pixels
    trail_yx = np.argwhere(trail_mask == 1)  # (row, col)
    trail_pixels = np.array([(x, y) for y, x in trail_yx])  # shape (M, 2)

    # Query nearest GPX point for each trail pixel
    distances, indices = tree.query(trail_pixels)
    max_idx = np.argmax(distances)

    # Extract worst-matching trail pixel and its nearest GPX point
    x_trail, y_trail = trail_pixels[max_idx]
    x_gpx, y_gpx = pixel_coords[indices[max_idx]]

    # Convert both to geo-coordinates
    lat_trail, lon_trail = pixel_to_geo(x_trail, y_trail, transform)
    lat_gpx, lon_gpx = pixel_to_geo(x_gpx, y_gpx, transform)
    #print(f"Trail pixel with largest distance from GPX: pixel ({x_trail}, {y_trail})")
    #print(f"Nearest GPX pixel: ({x_gpx}, {y_gpx})")
    #print(f"Pixel distance: {distances[max_idx]:.2f}")
    #print("\nGround Truth Trail Point (Geo):", (lat_trail, lon_trail))
    #print("Nearest GPX Point (Geo):", (lat_gpx, lon_gpx))
    return (float(lat_trail), float(lon_trail))


def evaluate_from_gpx(gpx_data, trail_mask, transform=None, top_n=20):
    pixel_coords = convert_gpx_to_pixel_coordinates(gpx_data, transform)
    pixel_coords = [(int(x), int(y)) for x, y in pixel_coords]


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

    # Get indices of top N largest distances
    distances = np.array(distances)
    valid_indices = np.where(distances >= 0)[0]
    sorted_indices = valid_indices[np.argsort(-distances[valid_indices])]
    top_indices = sorted_indices[:top_n]

    # Choose one at random
    chosen_idx = random.choice(top_indices)
    x_chosen, y_chosen = pixel_coords[chosen_idx]

    # Find nearest trail pixel
    trail_yx = np.argwhere(trail_mask == 1)
    trail_coords = np.array([(x, y) for y, x in trail_yx])
    dists = np.linalg.norm(trail_coords - np.array([[x_chosen, y_chosen]]), axis=1)
    nearest_idx = np.argmin(dists)
    x_nearest, y_nearest = trail_coords[nearest_idx]

    # Convert both to geo coordinates
    lat_chosen, lon_chosen = pixel_to_geo(x_chosen, y_chosen, transform)
    lat_nearest, lon_nearest = pixel_to_geo(x_nearest, y_nearest, transform)

    #print("\nRandom GPX Point (Geo):", (lat_chosen, lon_chosen))
    #print("Nearest Trail Point (Geo):", (lat_nearest, lon_nearest))
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


def compute_chamfer_distance(gpx_coords, trail_mask):
    """
    gpx_coords: Nx2 numpy array of GPX points (x, y) in image/map space
    mask: 2D binary numpy array (1 for trail pixels, 0 for background)
    """
    # Create GPX mask with lines
    gpx_mask = create_gpx_mask(gpx_coords, trail_mask.shape)

    # Get coordinates of all track pixels from both masks
    gpx_pixels = np.column_stack(np.nonzero(gpx_mask))[:, ::-1]  # (x, y) format
    trail_pixels = np.column_stack(np.nonzero(trail_mask))[:, ::-1]  # (x, y) format

    if len(trail_pixels) == 0 or len(gpx_pixels) == 0:
        return np.inf  # One of the sets is empty

        # Create KD-trees for efficient nearest neighbor search
    tree_trail = KDTree(trail_pixels)
    tree_gpx = KDTree(gpx_pixels)

    # GPX → Trail distances
    dists1, _ = tree_trail.query(gpx_pixels)
    chamfer1 = np.mean(dists1)

    # Trail → GPX distances
    dists2, _ = tree_gpx.query(trail_pixels)
    chamfer2 = np.mean(dists2)

    # Average bidirectional Chamfer distance
    return (chamfer1 + chamfer2) / 2


def main(args):
    print("inside main")
    mask_path = args.mask_path
    trail_mask = np.load(mask_path)["mask"]

    manifest_path = args.manifest_path
    manifest = load_config_file(manifest_path, return_dict=True)

    map_id = args.map_id
    trail_id = args.trail_id - 1 # trail IDs start at 0 but file names start at 1
    gpx_filepath = args.gpx_filepath

    # Computing affine transform from GCPs
    entry = manifest[map_id]
    gcps = entry["gcps"]
    A = compute_affine_from_gcps(gcps)  # 2x3 numpy array
    transform = affine_matrix_to_rasterio_affine(A)  # rasterio-style affine

    start_and_end = [entry["trails"][trail_id]["start"], entry["trails"][trail_id]["end"]]
    chamfer_distance_e2e = math.inf
    chamfer_distance_gp = math.inf

    # End-to-End query
    if args.end_to_end or not args.graph_processing:
        gpx = gpxpy.parse(get_route_as_gpx(start_and_end, gpx_filepath))
        gpx_pixel_coords = convert_gpx_to_pixel_coordinates(gpx, transform)

        chamfer_distance_e2e = compute_chamfer_distance(gpx_pixel_coords, trail_mask)
        print('Chamfer distance from End to End Query: ', chamfer_distance_e2e)

    # Graph processing
    ## Preprocessing: Generate graph representation, find path that visits all key nodes on simplified graph, then figure
    ## out a path on the full graph for every edge
    if args.graph_processing:
        start_and_end_pixels = [geo_to_pixel(lon, lat, transform) for lat, lon in start_and_end]  # result: x, y
        start_and_end_pixels = [(y, x) for (x, y) in start_and_end_pixels]  # Graph works with y, x!
        path_edges, node_degrees = generate_path(trail_mask, start=start_and_end_pixels[0], end=start_and_end_pixels[1])  # This method creates its own graph which it simplifies before finding the path
        print(path_edges)

        G = extract_graph_with_cycles(trail_mask)  # Sampling the points requires the full graph G
        visited_nodes = set()
        visited_edges = set()
        edge_trails = dict()
        for edge in path_edges:
            if edge in visited_edges:
                continue

            print(f"finding path for edge: {edge}...")
            node_a, node_b = edge
            edge_trail, visited_nodes = astar_no_revisiting(G, start=node_a, end=node_b, visited=visited_nodes)
            print(f'path found: {edge_trail}')
            if edge_trail:
                edge_trail = linear_sample_with_edges(edge_trail, 0.005)
                edge_trails[edge] = edge_trail
                visited_edges.add(edge)

                # If edge is a path between junction and leaf node, then we must traverse the same path back from the node
                if (node_degrees[node_a] == 1 and node_degrees[node_b] > 2) or (node_degrees[node_b] == 1 and node_degrees[node_a] > 2):
                    edge_trails[(node_b, node_a)] = list(reversed(edge_trail))
                    visited_edges.add((node_b, node_a))

        stops_pixel_coords = []
        for edge in path_edges:
            edge_trail = edge_trails.get(edge)
            if edge_trail:
                stops_pixel_coords = stops_pixel_coords + edge_trails[edge]
            else:
                stops_pixel_coords = stops_pixel_coords + [edge[0], edge[1]]
        stops_lat_lon = [pixel_to_geo(x, y, transform) for y, x in stops_pixel_coords]
        stops_lat_lon = [(lon, lat) for lat, lon in stops_lat_lon]  # switch back as coordinates are expected as lon, lat
        gpx = gpxpy.parse(get_route_as_gpx(stops_lat_lon, gpx_filepath))
        gpx_pixel_coords = convert_gpx_to_pixel_coordinates(gpx, transform)

        chamfer_distance_gp = compute_chamfer_distance(gpx_pixel_coords, trail_mask)
        print('Chamfer distance from Graph Processing: ', chamfer_distance_gp)

    # Choose either the End-to-End or Graph Processing query as initial result
    if chamfer_distance_e2e < chamfer_distance_gp:
        print('Chamfer distance from End to End Query is smaller: ', chamfer_distance_e2e)
        stops_lat_lon = start_and_end
        chamfer_distance = chamfer_distance_e2e
    else:
        print('Chamfer distance from Graph Processing is smaller: ', chamfer_distance_gp)
        chamfer_distance = chamfer_distance_gp
    gpx = gpxpy.parse(get_route_as_gpx(stops_lat_lon, gpx_filepath))

    if args.iterative_refinement:
        print("Starting iterative refinement..")
        for i in range(args.no_iterations):
            # Evaluate from Mask
            lat_mask, lon_mask = evaluate_from_mask(gpx, trail_mask, transform)
            stops_lat_lon_mask = stops_lat_lon.copy()
            insert_point_min_distance(stops_lat_lon_mask, (lon_mask, lat_mask))
            gpx_mask = gpxpy.parse(get_route_as_gpx(stops_lat_lon_mask, gpx_filepath))
            gpx_mask_pixel_coords = convert_gpx_to_pixel_coordinates(gpx_mask, transform)
            chamfer_mask = compute_chamfer_distance(gpx_mask_pixel_coords, trail_mask)
            print('Chamfer distance from Mask: ', chamfer_mask)

            # Evaluate from GPX
            lat_gpx, lon_gpx = evaluate_from_gpx(gpx, trail_mask, transform)
            stops_lat_lon_gpx = stops_lat_lon.copy()
            insert_point_min_distance(stops_lat_lon_gpx, (lon_gpx, lat_gpx))
            gpx_gpx = gpxpy.parse(get_route_as_gpx(stops_lat_lon_gpx, gpx_filepath))
            gpx_gpx_pixel_coords = convert_gpx_to_pixel_coordinates(gpx_gpx, transform)
            chamfer_gpx = compute_chamfer_distance(gpx_gpx_pixel_coords, trail_mask)
            print('Chamfer distance from GPX: ', chamfer_gpx)

            if chamfer_distance <= min(chamfer_mask, chamfer_gpx):
                print(f'Chamfer distance has not decreased. Stopping early after iteration {i}!')
                get_route_as_gpx(stops_lat_lon, f"{gpx_filepath}")
                break

            if chamfer_mask < chamfer_gpx:
                print('Chamfer distance from Mask is smaller: ', chamfer_mask)
                stops_lat_lon = stops_lat_lon_mask
                gpx = gpx_mask
            else:
                print('Chamfer distance from GPX is smaller: ', chamfer_gpx)
                stops_lat_lon = stops_lat_lon_gpx
                gpx = gpx_gpx


            chamfer_distance = min(chamfer_mask, chamfer_gpx)
            get_route_as_gpx(stops_lat_lon, f"{gpx_filepath}")
            print(f'GPX file of iteration {i} saved to ../outputs/{gpx_filepath}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process trail data and extract GPX from scanned map.")
    parser.add_argument('--manifest_path', type=str, default='../dataset/maps_manifest.yaml', help='Path to map manifest')
    parser.add_argument('--map_id', type=str, default='map009')
    parser.add_argument('--trail_id', type=int, default=1)
    parser.add_argument('--mask_path', type=str, default='../dataset/masks/map009_trail01.npz',
                        help='Path to trail mask image (NPZ)')
    parser.add_argument('--gpx_filepath', type=str, default='../outputs/map009_trail01_gt_extracted_PAPER.gpx',)
    parser.add_argument('--no_iterations', type=int, default=100,
                        help='Number of iterations for map refinement')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of farthest points to select for route discovery')
    parser.add_argument('--end_to_end', action='store_true')
    parser.add_argument('--graph_processing', action='store_true')
    parser.add_argument('--iterative_refinement', action='store_true')

    args = parser.parse_args()

    print("Running from:", __file__)
    print("sys.path:", sys.path)
    main(args)

