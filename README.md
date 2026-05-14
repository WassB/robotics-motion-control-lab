# Robotics Motion Control Lab

A personal robotics portfolio focused on control, path planning, and motion planning algorithms.

This repository contains small, self-contained implementations of classical robotics algorithms, as well as a larger Model Predictive Control project applied to autonomous racing.

## Main Project: MPC F1 Racing

The main project explores Model Predictive Control for a simplified racing car navigating a track under physical and control constraints.

### Objectives

- Track a racing line with minimal lateral error on different famous races.
- Respect steering, acceleration, and velocity limits
- Compare different prediction horizons and cost functions
- Visualize the evolution of trajectory, control inputs, and tracking error

### Methods

- Kinematic bicycle model
- Model Predictive Control
- Constrained optimization
- Trajectory tracking
- Simulation and visualization

### Results

Current results include:

- Stable trajectory tracking on simple tracks
- Visualization of predicted MPC horizon
- Comparison between different controller parameters



## Implemented Algorithms

| Category | Algorithm | Status | Description |
|---|---|---|---|
| Control | PID | Done | Basic feedback control on simple systems |
| Control | LQR | In progress | Optimal control for linear systems |
| Control | MPC | In progress | Constrained predictive control |
| Planning | Dijkstra | Done | Shortest path on weighted graphs |
| Planning | A* | Done | Heuristic graph search |
| Motion Planning | RRT | Done | Sampling-based planning |
| Motion Planning | RRT* | In progress | Asymptotically optimal RRT |
| Motion Planning | Hybrid A* | Planned | Non-holonomic path planning |
| Motion Planning | State Lattice | Planned | Motion primitive-based planning |

## Repository Structure

```text
control/          Classical control algorithms
planning/         Graph search and path planning algorithms
motion_planning/  Sampling-based and vehicle-constrained planners
projects/         Larger integrated robotics projects
common/           Shared utilities
docs/             Images, animations, and technical notes
