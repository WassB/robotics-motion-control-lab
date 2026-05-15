import numpy as np
from controller import ControllerParams
from plotting import plotting_trajectory
from simulation import simulation_trajectory


def main() -> None:
    # Initial true state
    x0 = 0.0
    y0 = 0.0
    theta0 = -np.pi
    init_true_state = np.array([x0, y0, theta0], dtype=np.float64)

    # Initial estimate
    init_estimate = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Initial covariance
    init_covariance = (np.diag([0.5, 0.5, np.deg2rad(20.0)]) ** 2).astype(np.float64)

    # Goal
    goal = np.array([10.0, 15.0], dtype=np.float64)

    # Sampling time
    dt = 0.05

    # Simulation duration
    T = 50.0

    # Controller parameters
    controller_params = ControllerParams(
        k_v=1.0,
        k_omega=2.0,
        v_max=1.0,
        omega_max=2.0,
        goal_tolerance=0.1,
    )

    # Process and measurement covariances
    Q = (np.diag([0.02, 0.02, np.deg2rad(1.0)]) ** 2).astype(np.float64)
    R = (np.diag([0.05, 0.05, np.deg2rad(2.0)]) ** 2).astype(np.float64)

    process_noise_std = np.array([0.02, 0.02, np.deg2rad(1.0)], dtype=np.float64)
    measurement_noise_std = np.array([0.05, 0.05, np.deg2rad(2.0)], dtype=np.float64)

    (
        time,
        state_history,
        prediction_history,
        update_history,
        prediction_cov_history,
        update_cov_history,
    ) = simulation_trajectory(
        init_true_state,
        init_estimate,
        init_covariance,
        goal,
        dt,
        T,
        controller_params,
        Q,
        R,
        process_noise_std,
        measurement_noise_std,
    )

    plotting_trajectory(
        time,
        state_history,
        prediction_history,
        update_history,
        prediction_cov_history,
        update_cov_history,
        goal,
    )


if __name__ == "__main__":
    main()