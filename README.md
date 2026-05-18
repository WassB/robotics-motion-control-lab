# Robotics Motion Control Lab

A personal robotics portfolio focused on control, path planning, and motion planning algorithms.

This repository contains small, self-contained implementations of classical robotics algorithms, as well as larger integrated projects such as Model Predictive Control for autonomous racing and autonomous robot navigation in unknown environments.

## Projects

- **MPC F1 Racing**
- **Autonomous Robot in Unknown Environment**

---

# MPC F1 Racing

The F1 Racing project explores Model Predictive Control applied to a simplified racing car navigating a track under physical and control constraints.

## Objectives

- Track a racing line with minimal lateral error on different well-known race tracks
- Respect steering, acceleration, and velocity limits
- Compare different prediction horizons and cost functions
- Visualize the evolution of the trajectory, control inputs, and tracking error

## Methods

- Kinematic bicycle model
- Model Predictive Control
- Constrained optimization
- Trajectory tracking
- Simulation and visualization

## Results

Current results include:

- Stable trajectory tracking on simple tracks
- Visualization of the predicted MPC horizon
- Comparison between different controller parameters

## Implemented Algorithms

| Category | Algorithm | Status | Description |
|---|---|---|---|
| Control | LQR | Done | Optimal control for linear systems |
| Control | MPC | Done | Constrained predictive control |
| Control | EKF | Done | Estimation of unknown states |

---

# Autonomous Robot

The Autonomous Robot project explores motion planning algorithms for an autonomous robot moving in an unknown environment with obstacles, for example a rover operating on Mars.

## Objectives

- Find a feasible and near-optimal path for the robot
- Respect steering, acceleration, and velocity limits
- Compare different planning solutions and design meaningful cost functions
- Visualize the evolution of the trajectory, control inputs, and tracking error

## Methods

- Kinematic bicycle model
- Motion planning algorithms, such as Hybrid A* and State Lattice planning
- Constrained optimization
- Trajectory tracking
- Simulation and visualization

## Results

Current results include:

- Stable trajectory tracking in simple environments
- Visualization of planned paths and trajectories
- Comparison between different planning and control parameters

## Implemented Algorithms

| Category | Algorithm | Status | Description |
|---|---|---|---|
| Control | LQR | Done | Optimal control for linear systems |
| Control | MPC | Done | Constrained predictive control |
| Control | EKF | Done | Estimation of unknown states |
| Planning | Dijkstra | Done | Shortest path search on weighted graphs |
| Planning | A* | Done | Heuristic graph search |
| Motion Planning | RRT | Done | Sampling-based motion planning |
| Motion Planning | RRT* | Done | Asymptotically optimal variant of RRT |
| Motion Planning | Hybrid A* | In progress | Non-holonomic path planning |
| Motion Planning | State Lattice | Done | Motion primitive-based planning |
| Task Planning | Transition System × Büchi Automaton | Planned | Optimization of the robot strategy and action order |
| 2D Simulation | Integrated planning and control in a 2D environment | Planned | A robot has to fulfill goals in an unknown 2D environment |
| 3D Simulation | Extension of the 2D simulation | Planned | A robot has to fulfill goals in an unknown 3D environment |

---

## Repository Structure

```text
control/          Classical control algorithms
planning/         Graph search and path planning algorithms
motion_planning/  Sampling-based and vehicle-constrained planners
projects/         Larger integrated robotics projects
common/           Shared utilities
docs/             Images, animations, and technical notes
