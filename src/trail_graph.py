import cv2
from skimage.morphology import skeletonize
import math
from scipy.spatial import KDTree
from collections import deque
import networkx as nx
import numpy as np
from itertools import product

def remove_small_components(mask, min_area=100):
    """
    Removes small connected components from a binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 1

    return cleaned_mask



def extract_graph_with_cycles(mask):
    #skeleton = skeletonize(mask < 255)
    skeleton = skeletonize(remove_small_components(mask))
    h, w = skeleton.shape[0], skeleton.shape[1]
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
    no_nodes_full = len(G_full.nodes)

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

    no_nodes_simple = len(G_simple.nodes)
    if no_nodes_full != no_nodes_simple:
        return simplify_graph(G_simple)
    else:
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


def preprocess_graph_for_path_finding(G, start, end):
    # Removes leaf nodes from simplified graph and groups them together with their connected junction node. The idea is that
    # if leaf node A is connected to junction B, then our path visiting all nodes will inevitably visit B before and after
    # visiting A (unless A is start or end node, then we visit B only once before or after). So by removing the leaf
    # nodes, we have fewer nodes to include during our search
    G_simplified = G.copy()
    junction_to_leaves = {}

    # Collect all leaf nodes and their junctions
    leaf_nodes = [node for node in G.nodes if G.degree(node) == 1 and node != start and node != end]

    for leaf in leaf_nodes:
        # Get the junction this leaf is connected to
        junction = list(G.neighbors(leaf))[0]
        if junction not in junction_to_leaves:
            junction_to_leaves[junction] = []
        junction_to_leaves[junction].append(leaf)
        # Remove leaf from simplified graph
        G_simplified.remove_node(leaf)

    return G_simplified, junction_to_leaves


def bfs_path_visiting_all_nodes(G, start, end, limit=4):
    # First preprocess the graph
    G_simplified, junction_to_leaves = preprocess_graph_for_path_finding(G, start, end)
    limit = len(G_simplified.nodes) * limit

    queue = deque()
    queue.append((start, [start], set([start])))

    while queue:
        current, path, visited = queue.popleft()

        # If we've visited all nodes in simplified graph and are at the end, expand path with leaf nodes
        if len(visited) == len(G_simplified.nodes) and current == end:
            # Expand path by inserting leaf node visits at their junction points
            final_path = []
            for i, node in enumerate(path):
                final_path.append(node)
                # If this node is a junction with leaves and we're not at the end of path
                if node in junction_to_leaves:
                    # Insert visits to all unvisited leaves at this junction
                    for leaf in junction_to_leaves[node]:
                        if leaf not in visited:
                            final_path.extend([leaf, node])
                            visited.add(leaf)
            return final_path

        for neighbor in G_simplified.neighbors(current):
            new_path = path + [neighbor]
            new_visited = visited | {neighbor}
            if len(new_path) < limit and new_path.count(neighbor) <= G_simplified.degree(neighbor) * 2:
                queue.append((neighbor, new_path, new_visited))

    return None  # No path found


def bfs_no_revisiting(G, start, end, visited=None, limit=0.5):
    if not visited:
        visited = set()
    limit = len(G.nodes) * limit  # To avoid endless exploring
    queue = deque()
    queue.append((start, [start], visited))
    visited_initial = visited

    while queue:
        current, path, visited = queue.popleft()

        if current == end:
            return path, visited

        if len(path) <= limit:
            for neighbor in G.neighbors(current):
                if neighbor not in visited or neighbor == end:
                    new_visited = visited | {neighbor}
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_visited))

    return None, visited_initial  # No path found


def astar_no_revisiting(G, start, end, visited=None, limit=0.5):
    if not visited:
        visited = set()
    limit = len(G.nodes) * limit  # To avoid endless exploring
    visited_initial = visited

    # Priority queue will be simulated with a list that we sort by f_score
    # Each entry is (f_score, current_node, path, visited)
    queue = [(0, start, [start], visited)]
    g_scores = {start: 0}  # Cost from start to node

    def heuristic(node):
        return euclidean_distance(node, end)

    while queue:
        queue.sort(key=lambda x: x[0])  # Sort by f_score
        f_score, current, path, visited = queue.pop(0)

        if current == end:
            return path, visited

        if len(path) <= limit:
            for neighbor in G.neighbors(current):
                if neighbor not in visited or neighbor == end:
                    new_visited = visited | {neighbor}
                    new_path = path + [neighbor]

                    # Calculate g_score for neighbor
                    g_score = g_scores[current] + euclidean_distance(current, neighbor)

                    if neighbor not in g_scores or g_score < g_scores[neighbor]:
                        g_scores[neighbor] = g_score
                        f_score = g_score + heuristic(neighbor)
                        queue.append((f_score, neighbor, new_path, new_visited))

    return None, visited_initial  # No path found




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
    G = collapse_nearby_nodes(G, distance_threshold=5)
    G = simplify_graph(G)
    if len(list(nx.connected_components(G))) > 1:
        G = connect_components_closest_first(G)
        G = collapse_nearby_nodes(G, distance_threshold=10)

    # since start and end might not be part of the graph, we match them to the closest points in the graph
    start_node = find_closest_node(G, start)
    end_node = find_closest_node(G, end)
    path = bfs_path_visiting_all_nodes(G, start_node, end_node)
    path_edges = list(zip(path, path[1:]))

    node_degrees = dict(G.degree)
    return path_edges, node_degrees

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def connect_components_closest_first(G):
    G = G.copy()
    components = list(nx.connected_components(G))

    while len(components) > 1:
        print(f"[INFO] Connecting components: {len(components)} remaining...")

        min_dist = float('inf')
        best_pair = None

        # Find the closest pair of nodes across any two components
        for i, comp1 in enumerate(components):
            for j in range(i + 1, len(components)):
                for u, v in product(comp1, components[j]):
                    d = euclidean_distance(u, v)
                    if d < min_dist:
                        min_dist = d
                        best_pair = (u, v)

        # Add edge between best_pair
        if best_pair is not None:
            u, v = best_pair
            G.add_edge(u, v, weight=min_dist)
        else:
            print("[WARN] No pair found — something went wrong.")
            break

        # Recompute components after merge
        components = list(nx.connected_components(G))

    print("[INFO] All components connected.")
    return G

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

