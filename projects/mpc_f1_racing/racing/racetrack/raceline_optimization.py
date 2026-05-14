import os
import numpy as np
import cvxpy as cp
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from racing.racetrack.racetrack import RaceTrack
from racing.racetrack.utils import plot_states, plot_inputs, plot_raceline
from racing.models.kinematic_bicycle import (
    KinematicBicycle,
    HEADING_INDEX,
    VX_INDEX,
    VY_INDEX,
    OMEGA_INDEX,
    STEERING_INDEX,
    N_INDEX,
    XI_INDEX,
    THROTTLE_INDEX,
    STEERING_RATE_INDEX,
)


# ---------- 1) Raceline convexe (min courbure) ----------
def compute_min_curvature_raceline_convex(racetrack: RaceTrack, resolution: float = 1.0, verbose: bool = False):
    """
    Calcule une raceline de courbure minimale via optimisation convexe.
    Retourne:
      - vehicle_positions: (N,2) positions XY sur la raceline initiale
      - n: (N,) déports latéraux par rapport au centre.
    """
    n_points = int(racetrack.length / resolution) + 1
    s_values = np.linspace(0, racetrack.length, n_points)

    vehicle_positions = cp.Variable((n_points, 2))
    n = cp.Variable(n_points)

    # Contraintes piste
    track_constraints = [cp.abs(n) <= racetrack.track_width / 2.0]
    for i, s in enumerate(s_values):
        track_constraints += [
            vehicle_positions[i, :] == racetrack.position(
                s) + n[i] * racetrack.normal_vector(s)
        ]

    # Discrétisations (diff finies périodiques)
    h = resolution
    D1 = np.zeros((n_points, n_points))
    for i in range(n_points):
        D1[i, i] = -1.0 / h
        D1[i, (i + 1) % n_points] = 1.0 / h

    D2 = np.zeros((n_points, n_points))
    for i in range(n_points):
        D2[i, i] = -2.0 / h**2
        D2[i, (i - 1) % n_points] = 1.0 / h**2
        D2[i, (i + 1) % n_points] = 1.0 / h**2

    # Fermeture de boucle (position + tangente)
    track_constraints += [
        vehicle_positions[0, :] == vehicle_positions[-1, :],
        vehicle_positions[1, :] - vehicle_positions[0, :]
        == vehicle_positions[-1, :] - vehicle_positions[-2, :],
    ]

    # Objectif convexe: L2 courbure + régularisation tangente
    curvature_L2 = cp.sum_squares(D2 @ vehicle_positions)
    tangent_energy = cp.sum_squares(D1 @ vehicle_positions)

    w_curv = 300.0
    w_tan = 0.1
    objective = w_curv * curvature_L2 + w_tan * tangent_energy

    prob = cp.Problem(cp.Minimize(objective), track_constraints)
    prob.solve(solver=cp.CLARABEL, verbose=verbose)

    return vehicle_positions.value, n.value


# ---------- 2) Raceline dynamique (non convexe, transcription directe) ----------
def compute_optimal_raceline_dynamic(
    model: KinematicBicycle,
    racetrack: RaceTrack,
    resolution: float = 1.0,
    plot_initial_conditions: bool = False,
):
    """
    Optimise la raceline avec un modèle (non convexe) en transcription directe.
    Et superpose sur un même graphe:
      - la raceline optimisée,
      - la raceline initiale (convexe).
    Retourne: x_ref_fun, u_ref_fun, k_ref_fun, s_values, x_initial_fun, u_initial_fun
    """

    # ---------------- Infos / maillage ----------------
    n_points = int(racetrack.length / resolution) + 1
    n_states = 7
    n_inputs = 2

    s_values = np.linspace(0, racetrack.length, n_points)
    ds = s_values[1] - s_values[0]

    print("RACETRACK OPTIMIZATION USING DIRECT TRANSCRIPTION")
    print("Vehicle model:", model.car_model)
    print("Racetrack:", racetrack.name)
    print("Resolution:", resolution)
    print("Num states:", model.n_states)
    print("Num inputs:", model.n_inputs)
    print(f"Discretization: {ds:.3f} m  ({n_points} points)")

    # ---------------- 2.1) Guess convexe ----------------
    try:
        vehicle_positions, n_values = compute_min_curvature_raceline_convex(
            racetrack, resolution=resolution
        )
        print("Convex initial guess: OK")
    except Exception as e:
        print("Convex initial guess FAILED:", e)
        return None, None

    curvature = compute_curvature(vehicle_positions)
    tangent = compute_tangent_vector(vehicle_positions)
    centerline_curvature = np.array([racetrack.curvature(s) for s in s_values])

    # ---------------- 2.2) Guess états/entrées ----------------
    max_a_long = model.parameters.max_drive_force / model.parameters.mass
    max_d_long = model.parameters.max_break_force / model.parameters.mass
    max_a_lat = model.parameters.mu * model.gravity

    V = np.sqrt(model.parameters.mu * 9.81 /
                np.maximum(np.abs(curvature), 1e-6))
    V = np.clip(V, 0.0, model.parameters.max_velocity * 0.5)

    for i in range(1, len(V) - 1):
        up = max_a_long * ds / max(V[i - 1], 1e-3)
        dn = max_d_long * ds / max(V[i - 1], 1e-3)
        V[i] = np.clip(V[i], V[i - 1] - dn, V[i - 1] + up)

    initial_omega_guess = np.clip(
        curvature * V, -max_a_lat / np.maximum(V, 1e-3), max_a_lat / np.maximum(V, 1e-3))
    initial_beta_guess = np.arctan(
        initial_omega_guess / np.maximum(V, 1e-3) * model.parameters.l_r)
    initial_vx_guess = V * np.cos(initial_beta_guess)
    initial_vy_guess = V * np.sin(initial_beta_guess)
    initial_heading_guess = np.unwrap(
        np.array([[np.arctan2(t[1], t[0]) for t in tangent]]))
    initial_xi_guess = np.unwrap(
        initial_heading_guess - np.array([[racetrack.heading(s) for s in s_values]]))
    initial_n_guess = n_values
    initial_steering_guess = np.arctan(model.parameters.L * curvature)
    initial_steering_guess = np.clip(
        initial_steering_guess,
        -model.parameters.max_steering_angle,
        model.parameters.max_steering_angle,
    )

    s_dot_guess = ((initial_vx_guess * np.cos(initial_xi_guess) - initial_vy_guess * np.sin(initial_xi_guess)) /
                   (1 - centerline_curvature * initial_n_guess)).flatten()
    dt = 1 / (s_dot_guess[:-1] / ds)
    steering_rate_guess = np.diff(initial_steering_guess).flatten() / dt
    steering_rate_guess = np.clip(
        steering_rate_guess, -
        model.parameters.max_steering_rate, model.parameters.max_steering_rate
    )
    throttle_guess = [model.force_to_throttle(
        f) for f in model.parameters.mass * np.diff(initial_vx_guess).flatten() / dt]

    print("Initial estimated lap time:", float(
        np.sum(ds / np.maximum(s_dot_guess, 1e-6))))

    initial_state_guess = np.vstack(
        (
            initial_heading_guess,
            initial_vx_guess,
            initial_vy_guess,
            initial_omega_guess,
            initial_steering_guess,
            initial_n_guess,
            initial_xi_guess,
        )
    )

    initial_inputs_guess = np.vstack((throttle_guess, steering_rate_guess))

    x_initial_fun = interp1d(
        s_values, initial_state_guess, axis=1, kind="previous", bounds_error=True)
    u_initial_fun = interp1d(
        s_values,
        np.hstack(
            (initial_inputs_guess, initial_inputs_guess[:, 0][:, np.newaxis])),
        axis=1,
        kind="previous",
        bounds_error=True,
    )

    if plot_initial_conditions:
        plot_states(curvilinear_coordinate=s_values, states=initial_state_guess,
                    curvature_centerline=centerline_curvature)
        plot_inputs(
            curvilinear_coordinate=s_values[:-1], inputs=initial_inputs_guess)
        fig, ax = racetrack.draw()
        plot_raceline(x_initial_fun, racetrack, name="Initial Guess (states)")
        ax.plot(
            vehicle_positions[:, 0],
            vehicle_positions[:, 1],
            linestyle="--",
            marker="o",
            markersize=2,
            label="Convex Raceline (XY)",
        )
        ax.legend()

    # ---------------- 3) Transcription directe ----------------
    opti = ca.Opti()
    state_scaling = model.x_scaling[:, np.newaxis]
    input_scaling = model.u_scaling[:, np.newaxis]

    x_scaled = opti.variable(n_states, n_points)
    u_scaled = opti.variable(n_inputs, n_points - 1)
    u = u_scaled * input_scaling
    x = x_scaled * state_scaling

    opti.set_initial(x_scaled, initial_state_guess / state_scaling)
    opti.set_initial(u_scaled, initial_inputs_guess / input_scaling)

    g_x = model.get_state_constraints_function()
    g_u = model.get_input_constraints_function()

    for ii in range(n_points - 1):
        kappa_step = racetrack.curvature(s_values[ii])
        x_next = model.step_ds(x[:, ii], u[:, ii], kappa_step, ds=ds)
        opti.subject_to((x[:, ii + 1] - x_next) / state_scaling == 0)
        opti.subject_to(g_x(x[:, ii], kappa_step) <= 0.0)
        opti.subject_to(g_u(u[:, ii], kappa_step) <= 0.0)

    x_last = x[:, -1]
    opti.subject_to(g_x(x_last, kappa_step) <= 0.0)

    # boucle fermée sur quelques états
    opti.subject_to(x[XI_INDEX, -1] - x[XI_INDEX, 0] == 0)
    opti.subject_to(x[N_INDEX, -1] - x[N_INDEX, 0] == 0)
    opti.subject_to(x[VX_INDEX, -1] - x[VX_INDEX, 0] == 0)
    opti.subject_to(x[VY_INDEX, -1] - x[VY_INDEX, 0] == 0)

    # Coût ~ min temps: somme 1/s_dot * ds
    cost = 0.0
    for ii in range(n_points):
        vx = x[VX_INDEX, ii]
        vy = x[VY_INDEX, ii]
        xi = x[XI_INDEX, ii]
        n = x[N_INDEX, ii]
        kappa = racetrack.curvature(s_values[ii])
        s_dot = (vx * ca.cos(xi) - vy * ca.sin(xi)) / (1 - kappa * n)
        cost += (1.0 / (s_dot + 1e-3)) * ds

    p_opts = {
        "qpsol": "osqp",
        "max_iter": 150,
        "hessian_approximation": "exact",
        "convexify_strategy": "regularize",
        "convexify_margin": 1e-3,
        "init_feasible": False,
        "elastic_mode": True,
        "second_order_corrections": True,
        "c1": 1e-2,
        "beta": 0.9,
        "tol_du": 2.5e-2,
        "tol_pr": 1e-4,
        "print_iteration": True,
        "qpsol_options.error_on_fail": 0,
        "qpsol_options.print_time": 0,
        "qpsol_options.verbose": 0,
        "qpsol_options.warm_start_primal": False,
        "qpsol_options.osqp.verbose": False,
        "qpsol_options.osqp.eps_abs": 1e-5,
        "qpsol_options.osqp.eps_rel": 1e-5,
        "qpsol_options.osqp.polish": True,
    }
    opti.solver("sqpmethod", p_opts)
    opti.minimize(cost)

    try:
        print("Start Optimization ...")
        solution = opti.solve()
        print("Optimization successful!")
    except Exception as e:
        print("Optimization failed:", e)
        # on continue avec la meilleure solution connue d'Opti

    print("Extracting reference trajectory ...")
    x_sol = opti.debug.value(x)
    u_sol = opti.debug.value(u)
    u_sol = np.hstack((u_sol, u_sol[:, 0][:, np.newaxis]))

    # Fonctions d'interpolation
    u_ref_fun = interp1d(s_values, u_sol, axis=1,
                         kind="previous", bounds_error=True)
    x_ref_fun = interp1d(s_values, x_sol, axis=1,
                         kind="previous", bounds_error=True)
    k_ref_fun = interp1d(s_values, centerline_curvature, bounds_error=True)

    # ---------- SUPERPOSITION sur un seul graphe ----------
    try:
        fig, ax = racetrack.draw()
        # Raceline optimisée (via x_ref_fun -> n(s))
        plot_raceline(x_ref_fun, racetrack, name="Optimized raceline")

        # Raceline initiale convexe (déjà en XY)
        ax.plot(
            vehicle_positions[:, 0],
            vehicle_positions[:, 1],
            linestyle="--",
            linewidth=1.8,
            color="0.3",
            label="Initial guess (convex)",
        )

        ax.legend(loc="best")
        ax.set_title(
            f"Raceline — initial guess vs optimized ({racetrack.name})")
        plt.show()
    except Exception:
        # le plotting ne doit pas casser le pipeline
        pass

    return x_ref_fun, u_ref_fun, k_ref_fun, s_values, x_initial_fun, u_initial_fun


# ---------- Utils géométriques ----------
def compute_curvature(points: np.ndarray) -> np.ndarray:
    """Courbure (discrète) pour une boucle fermée (N,2)."""
    N = len(points)
    p_prev = np.roll(points, 1, axis=0)
    p_next = np.roll(points, -1, axis=0)

    dx = (p_next[:, 0] - p_prev[:, 0]) / 2.0
    dy = (p_next[:, 1] - p_prev[:, 1]) / 2.0
    ddx = p_next[:, 0] - 2.0 * points[:, 0] + p_prev[:, 0]
    ddy = p_next[:, 1] - 2.0 * points[:, 1] + p_prev[:, 1]

    num = dx * ddy - dy * ddx
    denom = (dx**2 + dy**2) ** 1.5

    kappa = np.zeros(N)
    mask = denom > 1e-12
    kappa[mask] = num[mask] / denom[mask]
    return kappa


def compute_tangent_vector(points: np.ndarray) -> np.ndarray:
    """Tangente unitaire (N,2) pour une boucle fermée."""
    N = len(points)
    p_prev = np.roll(points, 1, axis=0)
    p_next = np.roll(points, -1, axis=0)

    dx = (p_next[:, 0] - p_prev[:, 0]) / 2.0
    dy = (p_next[:, 1] - p_prev[:, 1]) / 2.0

    tangents = np.vstack((dx, dy)).T
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    ok = norms[:, 0] > 1e-12
    tangents[ok] /= norms[ok]
    return tangents


# ---------- Sauvegarde / chargement ----------
def save_optimal_raceline(racetrack_name: str, directory: str, x_ref_fun: interp1d, u_ref_fun: interp1d, s_values: np.ndarray):
    """Sauvegarde x_ref/u_ref sur disque (avec s en dernière ligne)."""
    if u_ref_fun is None or x_ref_fun is None:
        raise ValueError("Optimal reference not computed yet.")

    x_ref = np.vstack((x_ref_fun(s_values), s_values[np.newaxis, :]))
    u_ref = np.vstack((u_ref_fun(s_values), s_values[np.newaxis, :]))

    np.save(os.path.join(directory, f"{racetrack_name}_x_ref.npy"), x_ref.T)
    np.save(os.path.join(directory, f"{racetrack_name}_u_ref.npy"), u_ref.T)


def get_reference(folder_path: str, racetrack_name: str, car_model: str) -> tuple[interp1d, interp1d]:
    """Charge x_ref/u_ref et renvoie les fonctions d'interpolation correspondantes."""
    x_ref_file = f"{folder_path}/{racetrack_name}/{car_model}/{racetrack_name}_x_ref.npy"
    u_ref_file = f"{folder_path}/{racetrack_name}/{car_model}/{racetrack_name}_u_ref.npy"

    x_ref = np.load(x_ref_file).T
    u_ref = np.load(u_ref_file).T

    s_ref = x_ref[-1, :]
    x_ref = x_ref[:-1, :]
    u_ref = u_ref[:-1, :]

    extended_s_ref = np.hstack((s_ref, s_ref[-1] + s_ref[1:]))
    extended_x_ref = np.hstack((x_ref, x_ref[:, 1:]))
    extended_u_ref = np.hstack((u_ref, u_ref[:, 1:]))

    # unwrap heading
    extended_x_ref[HEADING_INDEX, :] = np.unwrap(
        extended_x_ref[HEADING_INDEX, :])

    u_ref_fun = interp1d(extended_s_ref, extended_u_ref,
                         kind="previous", axis=1, bounds_error=True)
    x_ref_fun = interp1d(extended_s_ref, extended_x_ref,
                         kind="previous", axis=1, bounds_error=True)
    return x_ref_fun, u_ref_fun
