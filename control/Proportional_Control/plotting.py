import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plotting_trajectrory(
    time: NDArray[np.float64],
    state_history: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> None:
    x: NDArray[np.float64] = state_history[:, 0]
    y: NDArray[np.float64] = state_history[:, 1]
    theta: NDArray[np.float64] = state_history[:, 2]

    xg: float = float(goal[0])
    yg: float = float(goal[1])

    # ----- Figure 1: trajectory in the plane -----
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Robot trajectory")
    plt.scatter(x[0], y[0], marker="o", label="Start")
    plt.scatter(xg, yg, marker="x", s=100, label="Goal")

    # optional: show final heading
    plt.quiver(
        x[-1],
        y[-1],
        np.cos(theta[-1]),
        np.sin(theta[-1]),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
        label="Final orientation",
    )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Robot trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ----- Figure 2: states versus time -----
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, x)
    plt.ylabel("x [m]")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, y)
    plt.ylabel("y [m]")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, theta)
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.grid(True)

    plt.suptitle("State evolution")
    plt.tight_layout()
    plt.show()
