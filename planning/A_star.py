import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import heapq


def is_valid(grid, node):
    x, y = node
    return (
        0 <= x < grid.shape[0]
        and 0 <= y < grid.shape[1]
        and grid[x, y] == 0
    )


def neighbors_4(node):
    x, y = node
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    neighbors = []

    for dx, dy in directions:
        neighbors.append((x + dx, y + dy))

    return neighbors


def heuristic(node, goal):

    node = np.array(node)
    goal = np.array(goal)

    return np.linalg.norm(goal - node)

def A_star(start,grid, goal):
    
    start = tuple(start)
    goal = tuple(goal)

    search_list = []
    heapq.heappush(search_list, (heuristic(start, goal), start))

    parent = {}

    g_score = np.inf * np.ones(grid.shape)
    g_score[start[0], start[1]] = 0

    visited = np.zeros(grid.shape, dtype=bool)

    while search_list:
        f, node = heapq.heappop(search_list)

        if visited[node[0], node[1]]:
            continue
        
        visited[node[0], node[1]] = True

        if node == goal:
            break

        for neighbor in neighbors_4(node):
            if not is_valid(grid, neighbor):
                continue

            if visited[neighbor[0], neighbor[1]]:
                continue

            tentative_g = g_score[node[0], node[1]] + 1

            if tentative_g<g_score[neighbor[0],neighbor[1]]:
                g_score[neighbor[0],neighbor[1]] = tentative_g
                parent[neighbor] = node

                f_neighbor = tentative_g + heuristic(neighbor,goal) 
                heapq.heappush(search_list,(f_neighbor,neighbor))

    return parent, visited, g_score 


def reconstruct_path(parent,start,goal):
    start = tuple(start)
    goal = tuple(goal)
    
    path = [goal]

    current = goal

    while current != start:

        current = parent[current]
        path.append(current)
        
    path.reverse()
    return path

def plot_a_star_result(grid, parent, visited, start, goal):
    path = reconstruct_path(parent, start, goal)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, origin="lower", cmap="Greys")

    vx, vy = np.where(visited)
    plt.scatter(vx, vy, s=10, alpha=0.35, label="Visited")

    if path is not None:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        plt.plot(px, py, linewidth=2.5, label="Path")

    plt.scatter(start[0], start[1], marker="o", s=100, label="Start")
    plt.scatter(goal[0], goal[1], marker="*", s=150, label="Goal")

    plt.xlim(-0.5, grid.shape[0] - 0.5)
    plt.ylim(-0.5, grid.shape[1] - 0.5)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()



grid = np.zeros((100, 100))
grid[20:40, 20:40] = 1
grid[55:70, 10:60] = 1
grid[20:45, 60:80] = 1

start = (5, 5)
goal = (95, 95)

parent, visited, g_score = A_star(start, grid, goal)

plot_a_star_result(grid, parent, visited, start, goal)