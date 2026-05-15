import numpy as np
from numpy.typing import NDArray
from controller import wrap_angle
from models import motion_model, measurement_model, motion_jacobian, measurement_jacobian


def simulate_true_state(
    previous_state: NDArray[np.float64],
    control: NDArray[np.float64],
    dt: float,
    process_noise_std: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Simulate the true system state with additive process noise.
    """
    noise = np.random.normal(loc=0.0, scale=process_noise_std, size=3)
    next_state = motion_model(previous_state, control, dt) + noise
    next_state[2] = wrap_angle(next_state[2])
    return next_state.astype(np.float64)


def simulate_measurement(
    true_state: NDArray[np.float64],
    measurement_noise_std: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Simulate a noisy measurement of the state.
    """
    noise = np.random.normal(loc=0.0, scale=measurement_noise_std, size=3)
    measurement = measurement_model(true_state) + noise
    measurement[2] = wrap_angle(measurement[2])
    return measurement.astype(np.float64)


def ekf_predict(
    x_est: NDArray[np.float64],
    P_est: NDArray[np.float64],
    control: NDArray[np.float64],
    dt: float,
    Q: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    EKF prediction step.
    """
    x_pred = motion_model(x_est, control, dt)
    F = motion_jacobian(x_est, control, dt)
    P_pred = F @ P_est @ F.T + Q
    z_pred = measurement_model(x_pred)

    return x_pred, z_pred, P_pred.astype(np.float64)


def ekf_update(
    x_pred: NDArray[np.float64],
    z_pred: NDArray[np.float64],
    z_meas: NDArray[np.float64],
    P_pred: NDArray[np.float64],
    R: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    EKF update step.
    """
    H = measurement_jacobian()

    innovation = z_meas - z_pred
    innovation[2] = wrap_angle(innovation[2])

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ innovation
    x_upd[2] = wrap_angle(x_upd[2])

    I = np.eye(P_pred.shape[0], dtype=np.float64)
    P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T

    return x_upd.astype(np.float64), P_upd.astype(np.float64)