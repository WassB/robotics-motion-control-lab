import numpy as np
import random as rd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import heapq


def sample_free_points(n, grid):
    """
    Sample n free points in the grid.
    Returns a list of tuples (x, y).
    """
    points = set()

    while len(points) < n:
        x = rd.randint(0, grid.shape[0] - 1)
        y = rd.randint(0, grid.shape[1] - 1)

        if grid[x, y] == 0:
            points.add((x, y))

    return list(points)


def euclidean_distance(p1, p2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return np.linalg.norm(p2 - p1)


def is_collision_free_segment(p1, p2, grid):
    """
    Check whether the segment between p1 and p2 is collision-free.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    dist = euclidean_distance(p1, p2)
    n_steps = int(dist * 2) + 1

    for t in np.linspace(0, 1, n_steps):
        point = p1 + t * (p2 - p1)

        x = int(round(point[0]))
        y = int(round(point[1]))

        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
            return False

        if grid[x, y] == 1:
            return False

    return True


def build_roadmap(points, grid, k):
    """
    Build a PRM graph:
    graph[node] = [(neighbor, cost), ...]
    """
    graph = {point: [] for point in points}

    points_array = np.array(points)
    tree = cKDTree(points_array)

    for i, point in enumerate(points):
        distances, indices = tree.query(points_array[i], k=min(k + 1, len(points)))

        # si k=1 ou peu de points, scipy peut renvoyer des scalaires
        if np.isscalar(indices):
            indices = [indices]
            distances = [distances]

        for j, dist in zip(indices[1:], distances[1:]):  # skip self
            neighbor = points[j]

            if is_collision_free_segment(point, neighbor, grid):
                graph[point].append((neighbor, dist))
                
                # rendre le graphe symétrique
                if point not in [n for n, _ in graph[neighbor]]:
                    graph[neighbor].append((point, dist))

    return graph


def dijkstra_graph(graph, start, goal):

    parent = dict()
    visited = set()
    dist = {node: np.inf for node in graph}
    dist[start] = 0.0

    search_list = []
    heapq.heappush(search_list, (0.0, start))

    while search_list:
        current_dist, node = heapq.heappop(search_list)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            break

        for neighbor, cost in graph.get(node, []):
            new_dist = current_dist + cost

            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = node
                heapq.heappush(search_list, (new_dist, neighbor))

    return parent, visited, dist


def reconstruct_path(parent, start, goal):
    if goal != start and goal not in parent:
        return None

    path = [goal]
    current = goal

    while current != start:
        current = parent[current]
        path.append(current)

    path.reverse()
    return path


def visited_to_grid(visited, grid_shape):
    visited_grid = np.zeros(grid_shape, dtype=bool)

    for x, y in visited:
        visited_grid[x, y] = True

    return visited_grid


def plot_prm_result(grid, points, graph, path, visited, start, goal):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, origin="lower", cmap="Greys")

    # samples
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    plt.scatter(px, py, s=10, alpha=0.5, label="Samples")

    # edges
    for node, neighbors in graph.items():
        for neighbor, _ in neighbors:
            x = [node[0], neighbor[0]]
            y = [node[1], neighbor[1]]
            plt.plot(x, y, alpha=0.15)

    # visited
    if visited is not None:
        vx = [p[0] for p in visited]
        vy = [p[1] for p in visited]
        plt.scatter(vx, vy, s=15, alpha=0.5, label="Visited")

    # final path
    if path is not None:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, linewidth=3, label="Path")

    plt.scatter(start[0], start[1], marker="o", s=100, label="Start")
    plt.scatter(goal[0], goal[1], marker="*", s=150, label="Goal")

    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


def main():
    n = 200
    k = 10

    start = (5, 5)
    goal = (95, 95)

    grid = np.zeros((100, 100))
    grid[20:40, 20:40] = 1
    grid[55:70, 10:60] = 1
    grid[20:45, 60:80] = 1

    points = sample_free_points(n, grid)

    if start not in points:
        points.append(start)
    if goal not in points:
        points.append(goal)

    graph = build_roadmap(points, grid, k)

    parent, visited, dist = dijkstra_graph(graph, start, goal)
    path = reconstruct_path(parent, start, goal)

    if path is None:
        print("No path found.")
    else:
        print(f"Path found with cost {dist[goal]:.3f} and {len(path)} nodes.")

    plot_prm_result(grid, points, graph, path, visited, start, goal)


if __name__ == "__main__":
    main()