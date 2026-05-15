import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plotting_trajectory(
    time: NDArray[np.float64],
    true_state_history: NDArray[np.float64],
    prediction_history: NDArray[np.float64],
    estimate_history: NDArray[np.float64],
    prediction_cov_history: NDArray[np.float64],
    estimate_cov_history: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> None:
    """
    Plot the closed-loop trajectory and EKF estimation results.

    Parameters
    ----------
    time : NDArray[np.float64], shape (N,)
        Simulation time vector [s].
    true_state_history : NDArray[np.float64], shape (N, 3)
        True state history [x, y, theta].
    prediction_history : NDArray[np.float64], shape (N, 3)
        EKF predicted state history.
    estimate_history : NDArray[np.float64], shape (N, 3)
        EKF updated state history.
    prediction_cov_history : NDArray[np.float64], shape (N, 3, 3)
        EKF predicted covariance history.
    estimate_cov_history : NDArray[np.float64], shape (N, 3, 3)
        EKF updated covariance history.
    goal : NDArray[np.float64], shape (2,)
        Goal position [x_goal, y_goal].

    Returns
    -------
    None
    """
    # --- Basic input validation ---
    assert time.ndim == 1, f"Expected time to be 1D, got shape {time.shape}"
    assert true_state_history.shape[1] == 3, (
        f"Expected true_state_history shape (N, 3), got {true_state_history.shape}"
    )
    assert prediction_history.shape[1] == 3, (
        f"Expected prediction_history shape (N, 3), got {prediction_history.shape}"
    )
    assert estimate_history.shape[1] == 3, (
        f"Expected estimate_history shape (N, 3), got {estimate_history.shape}"
    )
    assert prediction_cov_history.shape[1:] == (3, 3), (
        f"Expected prediction_cov_history shape (N, 3, 3), got {prediction_cov_history.shape}"
    )
    assert estimate_cov_history.shape[1:] == (3, 3), (
        f"Expected estimate_cov_history shape (N, 3, 3), got {estimate_cov_history.shape}"
    )
    assert goal.shape == (2,), f"Expected goal shape (2,), got {goal.shape}"

    num_points = time.shape[0]
    assert true_state_history.shape[0] == num_points
    assert prediction_history.shape[0] == num_points
    assert estimate_history.shape[0] == num_points
    assert prediction_cov_history.shape[0] == num_points
    assert estimate_cov_history.shape[0] == num_points

    x_goal, y_goal = goal

    # --- Extract states ---
    x_true = true_state_history[:, 0]
    y_true = true_state_history[:, 1]
    theta_true = true_state_history[:, 2]

    x_pred = prediction_history[:, 0]
    y_pred = prediction_history[:, 1]
    theta_pred = prediction_history[:, 2]

    x_est = estimate_history[:, 0]
    y_est = estimate_history[:, 1]
    theta_est = estimate_history[:, 2]

    # --- Standard deviations from covariance diagonals ---
    pred_std_x = np.sqrt(np.maximum(prediction_cov_history[:, 0, 0], 0.0))
    pred_std_y = np.sqrt(np.maximum(prediction_cov_history[:, 1, 1], 0.0))
    pred_std_theta = np.sqrt(np.maximum(prediction_cov_history[:, 2, 2], 0.0))

    est_std_x = np.sqrt(np.maximum(estimate_cov_history[:, 0, 0], 0.0))
    est_std_y = np.sqrt(np.maximum(estimate_cov_history[:, 1, 1], 0.0))
    est_std_theta = np.sqrt(np.maximum(estimate_cov_history[:, 2, 2], 0.0))

    # --- State estimation errors ---
    error_x = x_est - x_true
    error_y = y_est - y_true
    error_theta = _wrap_angle_array(theta_est - theta_true)

    # --- Figure 1: 2D trajectory ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x_true, y_true, label="True trajectory", linewidth=2)
    ax1.plot(x_pred, y_pred, "--", label="Predicted trajectory", linewidth=1.5)
    ax1.plot(x_est, y_est, "-.", label="Estimated trajectory", linewidth=1.5)

    ax1.scatter(
        true_state_history[0, 0],
        true_state_history[0, 1],
        marker="o",
        s=80,
        label="Start",
    )
    ax1.scatter(
        x_goal,
        y_goal,
        marker="*",
        s=180,
        label="Goal",
    )

    ax1.set_title("Robot trajectory in the plane")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    # --- Figure 2: state evolution ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes2[0].plot(time, x_true, label="True x", linewidth=2)
    axes2[0].plot(time, x_pred, "--", label="Predicted x", linewidth=1.5)
    axes2[0].plot(time, x_est, "-.", label="Estimated x", linewidth=1.5)
    axes2[0].set_ylabel("x [m]")
    axes2[0].set_title("State evolution")
    axes2[0].grid(True)
    axes2[0].legend()

    axes2[1].plot(time, y_true, label="True y", linewidth=2)
    axes2[1].plot(time, y_pred, "--", label="Predicted y", linewidth=1.5)
    axes2[1].plot(time, y_est, "-.", label="Estimated y", linewidth=1.5)
    axes2[1].set_ylabel("y [m]")
    axes2[1].grid(True)
    axes2[1].legend()

    axes2[2].plot(time, theta_true, label="True theta", linewidth=2)
    axes2[2].plot(time, theta_pred, "--", label="Predicted theta", linewidth=1.5)
    axes2[2].plot(time, theta_est, "-.", label="Estimated theta", linewidth=1.5)
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("theta [rad]")
    axes2[2].grid(True)
    axes2[2].legend()

    fig2.tight_layout()

    # --- Figure 3: estimation errors with ±2 sigma bounds ---
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes3[0].plot(time, error_x, label="Estimation error on x", linewidth=1.5)
    axes3[0].plot(time, 2.0 * est_std_x, "--", label="+2σ estimate", linewidth=1.2)
    axes3[0].plot(time, -2.0 * est_std_x, "--", label="-2σ estimate", linewidth=1.2)
    axes3[0].set_ylabel("Error x [m]")
    axes3[0].set_title("EKF estimation error vs ±2σ bounds")
    axes3[0].grid(True)
    axes3[0].legend()

    axes3[1].plot(time, error_y, label="Estimation error on y", linewidth=1.5)
    axes3[1].plot(time, 2.0 * est_std_y, "--", label="+2σ estimate", linewidth=1.2)
    axes3[1].plot(time, -2.0 * est_std_y, "--", label="-2σ estimate", linewidth=1.2)
    axes3[1].set_ylabel("Error y [m]")
    axes3[1].grid(True)
    axes3[1].legend()

    axes3[2].plot(time, error_theta, label="Estimation error on theta", linewidth=1.5)
    axes3[2].plot(time, 2.0 * est_std_theta, "--", label="+2σ estimate", linewidth=1.2)
    axes3[2].plot(time, -2.0 * est_std_theta, "--", label="-2σ estimate", linewidth=1.2)
    axes3[2].set_xlabel("Time [s]")
    axes3[2].set_ylabel("Error theta [rad]")
    axes3[2].grid(True)
    axes3[2].legend()

    fig3.tight_layout()

    # --- Figure 4: covariance evolution ---
    fig4, axes4 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes4[0].plot(time, pred_std_x, label="Predicted σ_x", linewidth=1.5)
    axes4[0].plot(time, est_std_x, "--", label="Updated σ_x", linewidth=1.5)
    axes4[0].set_ylabel("σ_x [m]")
    axes4[0].set_title("Covariance evolution")
    axes4[0].grid(True)
    axes4[0].legend()

    axes4[1].plot(time, pred_std_y, label="Predicted σ_y", linewidth=1.5)
    axes4[1].plot(time, est_std_y, "--", label="Updated σ_y", linewidth=1.5)
    axes4[1].set_ylabel("σ_y [m]")
    axes4[1].grid(True)
    axes4[1].legend()

    axes4[2].plot(time, pred_std_theta, label="Predicted σ_theta", linewidth=1.5)
    axes4[2].plot(time, est_std_theta, "--", label="Updated σ_theta", linewidth=1.5)
    axes4[2].set_xlabel("Time [s]")
    axes4[2].set_ylabel("σ_theta [rad]")
    axes4[2].grid(True)
    axes4[2].legend()

    fig4.tight_layout()

    plt.show()


def _wrap_angle_array(angle_array: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Wrap an array of angles to [-pi, pi].
    """
    wrapped = (angle_array + np.pi) % (2.0 * np.pi) - np.pi
    return np.asarray(wrapped, dtype=np.float64)