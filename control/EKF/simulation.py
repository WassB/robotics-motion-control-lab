import numpy as np
from numpy.typing import NDArray

from robot_model import robot_step
from controller import ControllerParams


def simulation_trajectory(
    init_true_state: NDArray[np.float64],
    init_estimate: NDArray[np.float64],
    init_covariance: NDArray[np.float64],
    goal: NDArray[np.float64],
    dt: float,
    T: float,
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
    NDArray[np.float64],
]:
    """
    Simulate the closed-loop trajectory of a unicycle robot controlled
    with a proportional controller and observed with an EKF.

    Parameters
    ----------
    init_true_state : NDArray[np.float64], shape (3,)
        Initial true robot state [x, y, theta].
    init_estimate : NDArray[np.float64], shape (3,)
        Initial EKF state estimate [x_hat, y_hat, theta_hat].
    init_covariance : NDArray[np.float64], shape (3, 3)
        Initial EKF covariance matrix.
    goal : NDArray[np.float64], shape (2,)
        Goal position [x_goal, y_goal].
    dt : float
        Sampling time [s].
    T : float
        Total simulation time [s].
    controller_params : ControllerParams
        Parameters of the proportional controller.
    Q : NDArray[np.float64], shape (3, 3)
        Process noise covariance matrix.
    R : NDArray[np.float64], shape (3, 3)
        Measurement noise covariance matrix.
    process_noise_std : NDArray[np.float64], shape (3,)
        Standard deviation of simulated process noise.
    measurement_noise_std : NDArray[np.float64], shape (3,)
        Standard deviation of simulated measurement noise.

    Returns
    -------
    time : NDArray[np.float64], shape (N,)
        Simulation time vector.
    true_state_history : NDArray[np.float64], shape (N, 3)
        History of true states.
    prediction_history : NDArray[np.float64], shape (N, 3)
        History of EKF predicted states.
    estimate_history : NDArray[np.float64], shape (N, 3)
        History of EKF updated states.
    prediction_cov_history : NDArray[np.float64], shape (N, 3, 3)
        History of predicted covariance matrices.
    estimate_cov_history : NDArray[np.float64], shape (N, 3, 3)
        History of updated covariance matrices.
    """
    # --- Input validation ---
    assert init_true_state.shape == (3,), f"Expected init_true_state shape (3,), got {init_true_state.shape}"
    assert init_estimate.shape == (3,), f"Expected init_estimate shape (3,), got {init_estimate.shape}"
    assert init_covariance.shape == (3, 3), f"Expected init_covariance shape (3,3), got {init_covariance.shape}"
    assert goal.shape == (2,), f"Expected goal shape (2,), got {goal.shape}"
    assert Q.shape == (3, 3), f"Expected Q shape (3,3), got {Q.shape}"
    assert R.shape == (3, 3), f"Expected R shape (3,3), got {R.shape}"
    assert process_noise_std.shape == (3,), f"Expected process_noise_std shape (3,), got {process_noise_std.shape}"
    assert measurement_noise_std.shape == (3,), f"Expected measurement_noise_std shape (3,), got {measurement_noise_std.shape}"
    assert dt > 0.0, "Sampling time dt must be strictly positive."
    assert T >= 0.0, "Simulation horizon T must be non-negative."

    num_steps = int(round(T / dt))
    num_points = num_steps + 1

    time = np.linspace(0.0, num_steps * dt, num_points, dtype=np.float64)

    true_state_history = np.zeros((num_points, 3), dtype=np.float64)
    prediction_history = np.zeros((num_points, 3), dtype=np.float64)
    estimate_history = np.zeros((num_points, 3), dtype=np.float64)

    prediction_cov_history = np.zeros((num_points, 3, 3), dtype=np.float64)
    estimate_cov_history = np.zeros((num_points, 3, 3), dtype=np.float64)

    # --- Initialization ---
    true_state_history[0] = init_true_state
    prediction_history[0] = init_estimate
    estimate_history[0] = init_estimate

    prediction_cov_history[0] = init_covariance
    estimate_cov_history[0] = init_covariance

    # --- Main simulation loop ---
    for k in range(num_steps):
        (
            true_state_history[k + 1],
            prediction_history[k + 1],
            estimate_history[k + 1],
            prediction_cov_history[k + 1],
            estimate_cov_history[k + 1],
        ) = robot_step(
            true_state_prev=true_state_history[k],
            x_est_prev=estimate_history[k],
            P_est_prev=estimate_cov_history[k],
            goal=goal,
            dt=dt,
            controller_params=controller_params,
            Q=Q,
            R=R,
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
        )

    return (
        time,
        true_state_history,
        prediction_history,
        estimate_history,
        prediction_cov_history,
        estimate_cov_history,
    )