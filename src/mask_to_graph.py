import cv2
from skimage.morphology import skeletonize
import math
from scipy.spatial import KDTree
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch


def extract_graph_with_cycles(mask):
    skeleton = skeletonize(mask > 0)
    h, w = skeleton.shape
    G = nx.Graph()

    # Step 1: Add all skeleton pixels as nodes and connect 8-connected neighbors
    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                G.add_node((y, x))
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                            G.add_edge((y, x), (ny, nx_))

    return G

def simplify_graph(G_full):
    G_simple = nx.Graph()
    keypoints = [n for n in G_full.nodes if G_full.degree[n] != 2]
    visited = set()

    for kp in keypoints:
        for neighbor in G_full.neighbors(kp):
            if (kp, neighbor) in visited or (neighbor, kp) in visited:
                continue

            path = [kp]
            current = neighbor
            prev = kp

            while G_full.degree[current] == 2 and current not in keypoints:
                path.append(current)
                next_nodes = list(G_full.neighbors(current))
                next_nodes.remove(prev)
                prev, current = current, next_nodes[0]

            path.append(current)
            visited.add((kp, current))
            visited.add((current, kp))

            # Add simplified edge
            G_simple.add_node(kp)
            G_simple.add_node(current)
            G_simple.add_edge(kp, current, pixels=path)

    return G_simple

def collapse_nearby_nodes(G, distance_threshold=2):
    nodes = list(G.nodes)
    coords = np.array(nodes)
    tree = KDTree(coords)

    merged = {}
    visited = set()

    for i, node in enumerate(nodes):
        if node in visited:
            continue
        # Find neighbors within threshold
        idxs = tree.query_ball_point(coords[i], r=distance_threshold)
        group = [nodes[j] for j in idxs if nodes[j] not in visited]
        if not group:
            continue
        # Choose a representative (e.g., average of coords)
        rep_y = int(np.round(np.mean([p[0] for p in group])))
        rep_x = int(np.round(np.mean([p[1] for p in group])))
        rep = (group[0][0], group[0][1])

        # Map old nodes to representative
        for old in group:
            merged[old] = rep
            visited.add(old)

    # Create new graph
    new_G = nx.Graph()
    for u, v in G.edges:
        new_u = merged.get(u, u)
        new_v = merged.get(v, v)
        if new_u != new_v:  # Avoid self-loops from collapsed nodes
            new_G.add_edge(new_u, new_v)

    return new_G

def bfs_path_visiting_all_nodes(G, start, end):
    queue = deque()
    # Each element in the queue is a tuple: (current_node, path_so_far, visited_nodes)
    queue.append((start, [start], set([start])))

    while queue:
        current, path, visited = queue.popleft()

        # If we've visited all nodes and are at the end, we're done
        if len(visited) == len(G.nodes) and current == end:
            return path

        for neighbor in G.neighbors(current):
            # Allow revisits, but track which nodes we've visited at least once
            new_visited = visited | {neighbor}
            new_path = path + [neighbor]
            queue.append((neighbor, new_path, new_visited))

    return None  # No path found

def bfs_no_revisiting(G, start, end, visited=None):
    if not visited:
        visited = set()
    queue = deque()
    queue.append((start, [start], visited))
    visited_initial = visited

    while queue:
        current, path, visited = queue.popleft()

        if current == end:
            return path, visited

        for neighbor in G.neighbors(current):
            if neighbor not in visited or neighbor == end:
                new_visited = visited | {neighbor}
                new_path = path + [neighbor]
                queue.append((neighbor, new_path, new_visited))

    return None, visited_initial  # No path found


def visualize_graph_on_image(image, G, title="Graph Overlay", node_size=30):
    plt.figure(figsize=(7, 5))
    plt.imshow(image, cmap='gray')

    # Draw edges
    for u, v in G.edges:
        y0, x0 = u
        y1, x1 = v
        plt.plot([x0, x1], [y0, y1], 'c-', linewidth=1)

    # Draw nodes
    node_pos = {n: (n[1], n[0]) for n in G.nodes}  # swap (y, x) to (x, y)

    # Assign colors based on degree
    node_colors = []
    for node in G.nodes:
        deg = G.degree[node]
        if deg == 1:
            node_colors.append('#FAC719')  # endpoints
        elif deg == 2:
            node_colors.append('#19FAD0')  # simple path node (optional: exclude these)
        elif deg >= 3:
            node_colors.append('#F318FA')  # junctions
        else:
            node_colors.append('gray')  # fallback

    nx.draw_networkx_nodes(G, pos=node_pos, node_size=node_size, node_color=node_colors)

    color_labels = {"#FAC719": "Endpoint", "#19FAD0": "Path Node", "#F318FA": "Junction"}
    legend_elements = [Patch(facecolor=color, label=label) for color, label in color_labels.items()]
    plt.title(title)
    plt.axis('off')
    plt.legend(handles=legend_elements, title="Node Types", loc="upper left")
    plt.show()

def find_closest_node(G, point):
    y_input, x_input = point
    closest_node = None
    min_distance = float('inf')

    for node in G.nodes():
        y_node = node[0]
        x_node = node[1]

        if x_node is not None and y_node is not None:
            distance = math.hypot(x_node - x_input, y_node - y_input)
            if distance < min_distance:
                min_distance = distance
                closest_node = node

    return closest_node

def generate_path(trail_mask, start, end):
    G = extract_graph_with_cycles(trail_mask)
    G = simplify_graph(G)
    G = collapse_nearby_nodes(G, distance_threshold=10)
    G = simplify_graph(G)
    if len(list(nx.connected_components(G))) > 1:
        G = connect_components(G)
        G = collapse_nearby_nodes(G, distance_threshold=10)
        G = simplify_graph(G)

    # since start and end might not be part of the graph, we match them to the closest points in the graph
    start_node = find_closest_node(G, start)
    end_node = find_closest_node(G, end)
    path = bfs_path_visiting_all_nodes(G, start_node, end_node)
    path_edges = list(zip(path, path[1:]))
    return path_edges

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def connect_components(G):
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G  # Already connected

    candidate_edges = []
    for i, comp1 in enumerate(components):
        for j in range(i + 1, len(components)):
            comp2 = components[j]
            # Find shortest edge between these two components
            min_dist = float('inf')
            best_pair = None
            for u in comp1:
                for v in comp2:
                    d = euclidean_distance(u, v)
                    if d < min_dist:
                        min_dist = d
                        best_pair = (u, v)
            candidate_edges.append((*best_pair, min_dist))

    # Create graph over these inter-component connections
    inter_component_graph = nx.Graph()
    for u, v, d in candidate_edges:
        inter_component_graph.add_edge(u, v, weight=d)

    # Build MST over that graph
    mst = nx.minimum_spanning_tree(inter_component_graph)

    # Add MST edges to the original graph
    for u, v in mst.edges():
        G.add_edge(u, v, weight=euclidean_distance(u, v))

    return G


if __name__ == '__main__':
    # Load the binary trail mask
    trail_mask = cv2.imread('../Data/Minami_Senju/scan-mask.png', cv2.IMREAD_GRAYSCALE)
    trail_mask = (trail_mask == 0).astype(np.uint8)

    G = extract_graph_with_cycles(trail_mask)
    visualize_graph_on_image(trail_mask, G, title="Initial Graph")

    G = simplify_graph(G)
    visualize_graph_on_image(trail_mask, G, title="Graph after Simplifying")

    G = collapse_nearby_nodes(G, distance_threshold=10)
    visualize_graph_on_image(trail_mask, G, title="Graph after Collapsing Nearby Nodes")

    G = simplify_graph(G)
    visualize_graph_on_image(trail_mask, G, title="Graph after Simplifying Again")

    if len(list(nx.connected_components(G))) > 1:
        print('Graph is not connected yet!')

        G = connect_components(G)
        visualize_graph_on_image(trail_mask, G, title="Graph after Connecting Components")

        G = collapse_nearby_nodes(G, distance_threshold=10)
        G = simplify_graph(G)
        visualize_graph_on_image(trail_mask, G, title="Graph after Simplifying Last Time")

    print(G.nodes)
    start = (1268, 900)
    end = (1268, 900)
    start_node = find_closest_node(G, start)
    end_node = find_closest_node(G, end)
    path = bfs_path_visiting_all_nodes(G, start_node, end_node)
    print(path)

    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

