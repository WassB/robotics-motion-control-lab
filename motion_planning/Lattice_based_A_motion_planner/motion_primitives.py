import numpy as np
import heapq
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ============================================================
# Parameters
# ============================================================

@dataclass(frozen=True)
class ControllerParams:
    T: float = 2.0
    dt: float = 0.1

    v_f: float = 1.0
    v_b: float = 0.6
    omega_max: float = 0.6

    pos_error: float = 0.5
    theta_error: float = 0.3

    ntheta: int = 36

    # Planning resolution
    spatial_resolution: float = 0.1

    # Real map convention:
    # one cell of grid_real = real_cell_size units in the real world
    real_cell_size: float = 1.0

    # Robot radius in real units
    robot_radius: float = 2.0

    reverse_penalty: float = 1.5
    rotate_penalty: float = 0.4
    turn_penalty: float = 0.15

    lam: float = 1.0


# ============================================================
# Basic math
# ============================================================

def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def euclidean_distance_xy(p1, p2):
    p1 = np.asarray(p1[:2], dtype=float)
    p2 = np.asarray(p2[:2], dtype=float)
    return np.linalg.norm(p2 - p1)


def motion_model(state, control, dt):
    """
    Unicycle model:
        x_dot = v cos(theta)
        y_dot = v sin(theta)
        theta_dot = omega
    """
    x, y, theta = state
    v, omega = control

    x_next = x + dt * v * np.cos(theta)
    y_next = y + dt * v * np.sin(theta)
    theta_next = wrap_angle(theta + dt * omega)

    return np.array([x_next, y_next, theta_next], dtype=float)


# ============================================================
# Grid conversion and collision
# ============================================================

def discretize_real_grid(grid_real, real_cell_size, spatial_resolution):
    """
    Convert a coarse real occupancy grid into a finer discrete occupancy grid.

    Example:
        grid_real.shape = (100, 100)
        real_cell_size = 1.0
        spatial_resolution = 0.1

    -> returns a grid of shape (1000, 1000)
    """
    scale = real_cell_size / spatial_resolution
    scale_int = int(round(scale))

    if not np.isclose(scale, scale_int):
        raise ValueError(
            "real_cell_size / spatial_resolution must be an integer. "
            f"Got {real_cell_size} / {spatial_resolution} = {scale}"
        )

    grid_discrete = np.repeat(
        np.repeat(grid_real.astype(int), scale_int, axis=0),
        scale_int,
        axis=1
    )
    return grid_discrete


def inflate_obstacles_square(grid, radius_cells):
    """
    Inflate obstacles with a square structuring element.
    Simple and sufficient for a first version.
    """
    if radius_cells <= 0:
        return grid.copy()

    inflated = grid.copy()
    obstacle_indices = np.argwhere(grid == 1)

    for ix, iy in obstacle_indices:
        x_min = max(0, ix - radius_cells)
        x_max = min(grid.shape[0], ix + radius_cells + 1)
        y_min = max(0, iy - radius_cells)
        y_max = min(grid.shape[1], iy + radius_cells + 1)
        inflated[x_min:x_max, y_min:y_max] = 1

    return inflated


def project_state(state, spatial_resolution, ntheta):
    """
    Continuous real state -> discrete search key (ix, iy, itheta)
    """
    x, y, theta = state

    ix = int(round(x / spatial_resolution))
    iy = int(round(y / spatial_resolution))

    theta_2pi = theta % (2.0 * np.pi)
    dtheta = 2.0 * np.pi / ntheta
    itheta = int(round(theta_2pi / dtheta)) % ntheta

    return (ix, iy, itheta)


def is_collision_free_path(grid_discrete, path, spatial_resolution):
    """
    Check all sampled points of the path in the fine discrete occupancy grid.
    """
    for x, y, _ in path:
        ix = int(round(x / spatial_resolution))
        iy = int(round(y / spatial_resolution))

        if not (0 <= ix < grid_discrete.shape[0] and 0 <= iy < grid_discrete.shape[1]):
            return False

        if grid_discrete[ix, iy] == 1:
            return False

    return True


# ============================================================
# Motion primitives
# ============================================================

def build_motion_primitives(commands, T, dt, params):
    """
    Build primitives in the canonical local frame from (0,0,0).

    Stored data:
      - path_local
      - final_local
      - cost
      - command
    """
    primitives = {}
    n_steps = int(round(T / dt))

    for command in commands:
        q = np.array([0.0, 0.0, 0.0], dtype=float)
        path_local = [q.copy()]

        for _ in range(n_steps):
            q = motion_model(q, command, dt)
            path_local.append(q.copy())

        # Path length
        path_length = 0.0
        for i in range(1, len(path_local)):
            p_prev = path_local[i - 1][:2]
            p_curr = path_local[i][:2]
            path_length += np.linalg.norm(p_curr - p_prev)

        v, omega = command
        cost = path_length
        cost += params.turn_penalty * abs(omega) * T

        if v < 0:
            cost += params.reverse_penalty
        if abs(v) < 1e-9 and abs(omega) > 1e-9:
            cost += params.rotate_penalty

        primitives[command] = {
            "command": command,
            "path_local": path_local,
            "final_local": path_local[-1],
            "cost": cost,
        }

    return primitives


def apply_primitive(state, primitive):
    """
    Transform a local primitive into the world frame
    using the current state as the anchor rigid transform.
    """
    x0, y0, theta0 = state
    c = np.cos(theta0)
    s = np.sin(theta0)

    path_world = []
    for x_l, y_l, theta_l in primitive["path_local"]:
        x_w = x0 + c * x_l - s * y_l
        y_w = y0 + s * x_l + c * y_l
        theta_w = wrap_angle(theta0 + theta_l)
        path_world.append((x_w, y_w, theta_w))

    return path_world


# ============================================================
# A*
# ============================================================

def heuristic(state, goal, params):
    return euclidean_distance_xy(state, goal) + params.lam * np.abs(wrap_angle(goal[2]-state[2]))


def is_goal(state, goal, params):
    pos_err = np.linalg.norm(np.array(state[:2]) - np.array(goal[:2]))
    th_err = abs(wrap_angle(state[2] - goal[2]))
    return (pos_err < params.pos_error) and (th_err < params.theta_error)


def reconstruct_path(parent, segment_from_parent, state_for_key, start_key, goal_key):
    if goal_key is None:
        return None

    key_chain = []
    k = goal_key
    while k is not None:
        key_chain.append(k)
        k = parent[k]
    key_chain.reverse()

    full_path = [state_for_key[start_key]]

    for k in key_chain[1:]:
        seg = segment_from_parent[k]
        full_path.extend(seg[1:])

    return full_path


def trajectory_planning_A_star(start, goal, grid_discrete, params):
    commands = [
        (params.v_f, 0.0),
        (params.v_f,  params.omega_max),
        (params.v_f, -params.omega_max),
        (-params.v_b, 0.0),
        (-params.v_b,  params.omega_max),
        (-params.v_b, -params.omega_max),
    ]

    primitives = build_motion_primitives(commands, params.T, params.dt, params)

    start_key = project_state(start, params.spatial_resolution, params.ntheta)
    goal_key = project_state(goal, params.spatial_resolution, params.ntheta)

    # Validity checks
    if not is_collision_free_path(grid_discrete, [start], params.spatial_resolution):
        raise ValueError("Start is outside the map or inside an obstacle.")
    if not is_collision_free_path(grid_discrete, [goal], params.spatial_resolution):
        raise ValueError("Goal is outside the map or inside an obstacle.")

    g_score = {start_key: 0.0}
    parent = {start_key: None}
    segment_from_parent = {}
    state_for_key = {start_key: tuple(start)}
    command_for_key = {}

    closed = set()

    open_heap = []
    push_id = 0
    heapq.heappush(
        open_heap,
        (heuristic(start, goal, params), push_id, 0.0, start_key, tuple(start))
    )

    reached_goal_key = None

    while open_heap:
        f_current, _, g_current, current_key, current_state = heapq.heappop(
            open_heap)

        if g_current > g_score.get(current_key, np.inf) + 1e-12:
            continue

        if current_key in closed:
            continue

        closed.add(current_key)

        if is_goal(current_state, goal, params):
            reached_goal_key = current_key
            break

        for primitive in primitives.values():
            path_world = apply_primitive(current_state, primitive)

            if not is_collision_free_path(grid_discrete, path_world, params.spatial_resolution):
                continue

            successor_state = path_world[-1]
            successor_key = project_state(
                successor_state, params.spatial_resolution, params.ntheta)

            tentative_g = g_score[current_key] + primitive["cost"]

            if tentative_g + 1e-12 < g_score.get(successor_key, np.inf):
                g_score[successor_key] = tentative_g
                parent[successor_key] = current_key
                segment_from_parent[successor_key] = path_world
                state_for_key[successor_key] = successor_state
                command_for_key[successor_key] = primitive["command"]

                push_id += 1
                f_succ = tentative_g + heuristic(successor_state, goal, params)
                heapq.heappush(
                    open_heap,
                    (f_succ, push_id, tentative_g, successor_key, successor_state)
                )

    continuous_path = reconstruct_path(
        parent=parent,
        segment_from_parent=segment_from_parent,
        state_for_key=state_for_key,
        start_key=start_key,
        goal_key=reached_goal_key,
    )

    result = {
        "found": reached_goal_key is not None,
        "path": continuous_path,
        "closed": closed,
        "g_score": g_score,
        "parent": parent,
        "state_for_key": state_for_key,
        "command_for_key": command_for_key,
        "start_key": start_key,
        "goal_key": goal_key,
        "reached_goal_key": reached_goal_key,
    }

    return result


# ============================================================
# Visualization in real (x,y)
# ============================================================

def plot_arrow(x, y, theta, color="k", length=2.5, width=0.5):
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    plt.arrow(
        x, y, dx, dy,
        head_width=width,
        head_length=width * 1.2,
        fc=color, ec=color,
        length_includes_head=True,
        zorder=5
    )


def plot_result_xy(grid_real, start, goal, result, params):
    """
    Plot in real x-y coordinates, using the real coarse grid as background.
    """
    nx_real, ny_real = grid_real.shape
    x_max = nx_real * params.real_cell_size
    y_max = ny_real * params.real_cell_size

    plt.figure(figsize=(10, 10))

    # Background: real grid
    plt.imshow(
        grid_real.T,
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        extent=[0, x_max, 0, y_max],
        alpha=0.8
    )

    # Expanded states converted back to real coordinates
    if len(result["closed"]) > 0:
        closed_xy = np.array([
            (ix * params.spatial_resolution, iy * params.spatial_resolution)
            for ix, iy, _ in result["closed"]
        ])
        plt.scatter(
            closed_xy[:, 0],
            closed_xy[:, 1],
            s=2,
            c="orange",
            alpha=0.20,
            label="Expanded states"
        )

    # Planned path in real coordinates
    if result["found"] and result["path"] is not None:
        path = np.array(result["path"])
        plt.plot(path[:, 0], path[:, 1], "b-",
                 linewidth=2.5, label="Planned path")

    plt.scatter(start[0], start[1], c="green", s=100,
                marker="o", label="Start", zorder=6)
    plt.scatter(goal[0], goal[1], c="red", s=100,
                marker="x", label="Goal", zorder=6)

    plot_arrow(start[0], start[1], start[2], color="green")
    plot_arrow(goal[0], goal[1], goal[2], color="red")

    title = "A* motion planning in real (x, y)"
    title += " — PATH FOUND" if result["found"] else " — NO PATH"

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()


# ============================================================
# Demo real map
# ============================================================

def build_demo_grid_real():
    """
    Coarse real map:
    1 cell = 1 real unit
    total domain = [0,100] x [0,100]
    """
    grid_real = np.zeros((100, 100), dtype=int)

    grid_real[20:40, 20:40] = 1
    grid_real[70:80, 10:55] = 1
    grid_real[20:50, 60:80] = 1

    return grid_real


# ============================================================
# Main
# ============================================================

def main():
    params = ControllerParams(
        T=1.0,
        dt=0.1,
        v_f=1.0,
        v_b=0.6,
        omega_max=0.6,
        pos_error=0.5,
        theta_error=0.3,
        ntheta=36,
        spatial_resolution=1.0,
        real_cell_size=1.0,
        robot_radius=2.0,
        lam=1.5
    )
    start = (5.0, 5.0, 0.0)
    goal = (95.0, 95.0, 0.0)

    # 1) Real coarse grid
    grid_real = build_demo_grid_real()

    # 2) Convert to fine discrete grid for planning
    grid_discrete = discretize_real_grid(
        grid_real,
        real_cell_size=params.real_cell_size,
        spatial_resolution=params.spatial_resolution
    )

    # 3) Inflate obstacles in the discrete grid
    radius_cells = int(
        np.ceil(params.robot_radius / params.spatial_resolution))
    grid_discrete = inflate_obstacles_square(grid_discrete, radius_cells)

    # 4) Plan
    result = trajectory_planning_A_star(start, goal, grid_discrete, params)

    # 5) Print info
    if result["found"]:
        path = np.array(result["path"])
        path_length = 0.0
        for i in range(1, len(path)):
            path_length += np.linalg.norm(path[i, :2] - path[i - 1, :2])

        print("Path found.")
        print(f"Number of points in path: {len(path)}")
        print(f"Approximate path length: {path_length:.2f}")
        print(f"Expanded discrete states: {len(result['closed'])}")
    else:
        print("No path found.")
        print(f"Expanded discrete states: {len(result['closed'])}")

    # 6) Plot in real x-y coordinates
    plot_result_xy(grid_real, start, goal, result, params)


if __name__ == "__main__":
    main()
