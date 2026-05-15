import numpy as np
from numpy.typing import NDArray
from controller import control_law, ControllerParams
from EKF import simulate_true_state, simulate_measurement, ekf_predict, ekf_update


def robot_step(
    true_state_prev: NDArray[np.float64],
    x_est_prev: NDArray[np.float64],
    P_est_prev: NDArray[np.float64],
    goal: NDArray[np.float64],
    dt: float,
    controller_params: ControllerParams,
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    process_noise_std: NDArray[np.float64],
    measurement_noise_std: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Propagate the true system and EKF estimate by one time step.
    """
    control = control_law(x_est_prev, goal, controller_params)

    true_state = simulate_true_state(
        true_state_prev, control, dt, process_noise_std)
    measurement = simulate_measurement(true_state, measurement_noise_std)

    x_pred, z_pred, P_pred = ekf_predict(
        x_est_prev, P_est_prev, control, dt, Q)

    x_upd, P_upd = ekf_update(x_pred, z_pred, measurement, P_pred, R)

    return true_state, x_pred, x_upd, P_pred, P_upd
