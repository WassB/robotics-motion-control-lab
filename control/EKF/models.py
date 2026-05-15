import numpy as np
from numpy.typing import NDArray
from controller import wrap_angle


def motion_model(
    state: NDArray[np.float64],
    control: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """
    Discrete-time unicycle motion model using forward Euler integration.
    """
    x, y, theta = state
    v, omega = control

    x_next = x + dt * v * np.cos(theta)
    y_next = y + dt * v * np.sin(theta)
    theta_next = wrap_angle(theta + dt * omega)

    return np.array([x_next, y_next, theta_next], dtype=np.float64)


def measurement_model(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Full-state measurement model.
    Assumption: z = h(x) = x
    """
    x, y, theta = state
    return np.array([x, y, theta], dtype=np.float64)


def motion_jacobian(
    state: NDArray[np.float64],
    control: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """
    Jacobian of the motion model with respect to the state.
    """
    _, _, theta = state
    v, _ = control

    return np.array(
        [
            [1.0, 0.0, -dt * v * np.sin(theta)],
            [0.0, 1.0,  dt * v * np.cos(theta)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def measurement_jacobian() -> NDArray[np.float64]:
    """
    Jacobian of the full-state measurement model.
    """
    return np.eye(3, dtype=np.float64)