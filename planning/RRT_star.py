import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def sample_random_point(grid):
    

    while True:
        x = np.random.randint(0,grid.shape[0])
        y = np.random.randint(0,grid.shape[1])

        if grid[x,y]==0:
            return(x,y)

def nearest_node(tree_nodes,q_rand):
    
    q_rand = np.asarray(q_rand,dtype=float)
    dist = np.inf
    q_near = None

    for point in tree_nodes:
        point_arr = np.asarray(point,dtype=float)
        new_dist = np.linalg.norm(q_rand-point_arr)
        if new_dist < dist:
            dist = new_dist
            q_near = point

    return q_near


def is_collision_free_segment(p1, p2, grid):

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    dist = np.linalg.norm(p2-p1)
    n_steps = int((2 * dist)) + 1 

    for t in np.linspace(0, 1, n_steps):
        p = p1 + t * (p2 - p1)

        x = int(round(p[0]))
        y = int(round(p[1]))

        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
            return False

        if grid[x, y] == 1:
            return False
    
    return True


def steer(q_near, q_rand, step_size):
    q_near = np.asarray(q_near, dtype=float)
    q_rand = np.asarray(q_rand, dtype=float)

    direction = q_rand - q_near
    dist = np.linalg.norm(direction)

    if dist == 0:
        return tuple(q_near.astype(int))

    step = min(dist, step_size)
    q_new = q_near + step * direction / dist

    x = int(round(q_new[0]))
    y = int(round(q_new[1]))

    return (x, y)


def k_neighbors(q_new, nodes, grid, k):

    points_array = np.array(nodes)
    tree = cKDTree(points_array)
    distances, indices = tree.query(q_new, k=min(k + 1, len(nodes)))
    neighbors = list()

    for indice,distance in zip(indices[1:],distances[1:]):

        if is_collision_free_segment(nodes[indice], q_new, grid):
            neighbors.append((nodes[indice],distance))

    return neighbors


def the_father(dist,q_new,nodes,grid,k):

    neighbors = k_neighbors(q_new, nodes, grid, k)
    actual_cost = np.inf
    parent = None

    for neighbor,distance in neighbors:
        new_cost = dist[neighbor] + distance
                
        if new_cost<actual_cost:
            actual_cost = new_cost
            parent = neighbor


    return parent,actual_cost



def build_rrt_star(start, goal, grid, max_iter, step_size, goal_radius, k):
    
    start = tuple(start)

    goal = tuple(goal)

    nodes = [start]
    parent = dict()
    visited = np.zeros(grid.shape,dtype=bool)
    visited[start[0],start[1]] = True
    dist = {start: 0.0}

    for _ in range(max_iter):

            q_rand = sample_random_point(grid)
            q_near = nearest_node(nodes,q_rand)
            q_new = steer(q_near, q_rand, step_size)

            if not (0 <= q_new[0] < grid.shape[0] and 0 <= q_new[1] < grid.shape[1]):
                continue
    
            if visited[q_new[0],q_new[1]]:
                continue

            if not is_collision_free_segment(q_near, q_new, grid):
                continue

            nodes.append(q_new)
            father,cost= the_father(dist,q_new,nodes,grid,k)
            parent[q_new] = father
            dist[q_new] = cost
            visited[q_new[0],q_new[1]] = True
            
            dist_goal = np.linalg.norm(np.asarray(goal,dtype=float)-np.asarray(q_new,dtype=float))

            if dist_goal<goal_radius:

                if is_collision_free_segment(q_new, goal, grid):
                    nodes.append(goal)
                    parent[goal],_ = the_father(dist,goal,nodes,grid,k)
                    visited[goal[0], goal[1]] = True
                    break

    return parent, nodes, visited


def reconstruct_path(parent, start, goal):
    start = tuple(start)
    goal = tuple(goal)

    if goal != start and goal not in parent:
        return None

    path = [goal]
    current = goal

    while current != start:
        current = parent[current]
        path.append(current)

    path.reverse()
    return path


def plot_rrt_tree(grid, nodes, parent, start=None, goal=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, origin="lower", cmap="Greys")

    for node in nodes:
        if node in parent:
            p = parent[node]
            plt.plot([p[0], node[0]], [p[1], node[1]], alpha=0.3)

    if len(nodes) > 0:
        nx = [n[0] for n in nodes]
        ny = [n[1] for n in nodes]
        plt.scatter(nx, ny, s=10, alpha=0.6, label="Nodes")

    if start is not None:
        plt.scatter(start[0], start[1], marker="o", s=100, label="Start")

    if goal is not None:
        plt.scatter(goal[0], goal[1], marker="*", s=150, label="Goal")

    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def plot_rrt_path(grid, path, start=None, goal=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, origin="lower", cmap="Greys")

    if path is not None and len(path) > 0:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        plt.plot(px, py, linewidth=3, label="Path")

    if start is not None:
        plt.scatter(start[0], start[1], marker="o", s=100, label="Start")

    if goal is not None:
        plt.scatter(goal[0], goal[1], marker="*", s=150, label="Goal")

    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def plot_rrt_result(grid, nodes, parent, start, goal):
    path = reconstruct_path(parent, start, goal)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, origin="lower", cmap="Greys")

    for node in nodes:
        if node in parent:
            p = parent[node]
            plt.plot([p[0], node[0]], [p[1], node[1]], alpha=0.25)

    if len(nodes) > 0:
        nx = [n[0] for n in nodes]
        ny = [n[1] for n in nodes]
        plt.scatter(nx, ny, s=10, alpha=0.5, label="Nodes")

    if path is not None:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        plt.plot(px, py, linewidth=3, label="Path")

    plt.scatter(start[0], start[1], marker="o", s=100, label="Start")
    plt.scatter(goal[0], goal[1], marker="*", s=150, label="Goal")

    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def path_length(path):
    if path is None or len(path) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(path)):
        p1 = np.asarray(path[i - 1], dtype=float)
        p2 = np.asarray(path[i], dtype=float)
        length += np.linalg.norm(p2 - p1)

    return length


def main():


    start = (5, 5)
    goal = (95, 95)

    grid = np.zeros((100, 100))
    grid[20:40, 20:40] = 1
    grid[55:70, 10:60] = 1
    grid[20:45, 60:80] = 1

    parent, nodes, visited = build_rrt_star(start, goal, grid, max_iter=3000, step_size=5, goal_radius=5,k=20)

    path = reconstruct_path(parent, start, goal)
    print(path_length(path))

    plot_rrt_result(grid, nodes, parent, start, goal)


if __name__=="__main__":
    main()
