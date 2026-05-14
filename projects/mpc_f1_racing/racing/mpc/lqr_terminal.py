# racing/mpc/lqr_terminal.py
import numpy as np
import casadi as ca

def _sym(M):  # symmetrize (robuste numériquement)
    return 0.5 * (M + M.T)

def _dare_iter(A, B, Q, R, iters=500, tol=1e-10):
    """Résout la DARE par itérations de valeur (fallback robuste sans SciPy)."""
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        G   = R + BtP @ B
        K   = np.linalg.solve(G, BtP @ A)      # K = (R+B'PB)^-1 B'PA
        Pn  = A.T @ P @ A - A.T @ P @ B @ K + Q
        Pn  = _sym(Pn)
        if np.linalg.norm(Pn - P, 'fro') <= tol * (1.0 + np.linalg.norm(P, 'fro')):
            P = Pn
            break
        P = Pn
    BtP = B.T @ P
    K   = np.linalg.solve(R + BtP @ B, BtP @ A)
    return _sym(P), K

def _linearize_step_ds(model, x_bar, u_bar, kappa, ds):
    """A,B = d(step_ds)/dx, d(step_ds)/du au point (x_bar,u_bar,kappa)."""
    x = ca.SX.sym("x", model.n_states)
    u = ca.SX.sym("u", model.n_inputs)
    k = ca.SX.sym("k", 1)
    f = ca.Function("f", [x, u, k], [model.step_ds(x, u, k, ds=ds)])
    Jx = ca.Function("Jx", [x, u, k], [ca.jacobian(f(x, u, k), x)])
    Ju = ca.Function("Ju", [x, u, k], [ca.jacobian(f(x, u, k), u)])
    A = np.array(Jx(x_bar, u_bar, kappa))
    B = np.array(Ju(x_bar, u_bar, kappa))
    return A, B

def _scale_AB(A, B, x_scaling, u_scaling):
    """
    Conversion vers l'espace normalisé:
      x_scaled = x / Sx,  u_scaled = u / Su
      => Abar = Sx^{-1} A Sx,   Bbar = Sx^{-1} B Su
    """
    Sx_inv = np.diag(1.0 / np.asarray(x_scaling).flatten())
    Sx     = np.diag(np.asarray(x_scaling).flatten())
    Su     = np.diag(np.asarray(u_scaling).flatten())
    Abar = Sx_inv @ A @ Sx
    Bbar = Sx_inv @ B @ Su
    return Abar, Bbar

def compute_terminal_P(model,
                       x_bar: np.ndarray,
                       u_bar: np.ndarray,
                       kappa: float,
                       ds: float,
                       Q: np.ndarray,
                       R: np.ndarray,
                       x_scaling: np.ndarray,
                       u_scaling: np.ndarray):
    """
    Calcule P (et K) pour Qt via LQR sur la linéarisation discrète.
    ATTENTION: Q et R doivent être ceux utilisés dans TON coût (sur x_scaled, u_scaled).

    Retourne:
      P (numpy), K (numpy), (Abar, Bbar) dans l'espace normalisé.
    """
    # 1) Linéarisation (physique)
    A, B = _linearize_step_ds(model, x_bar, u_bar, kappa, ds)

    # 2) Passage en coordonnées normalisées (cohérent avec ton coût)
    Abar, Bbar = _scale_AB(A, B, x_scaling, u_scaling)

    # 3) Solve DARE (SciPy si dispo, sinon fallback)
    try:
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(Abar, Bbar, Q, R)
        P = _sym(P)
        BtP = Bbar.T @ P
        K   = np.linalg.solve(R + BtP @ Bbar, BtP @ Abar)
    except Exception:
        P, K = _dare_iter(Abar, Bbar, Q, R)

    return P, K, (Abar, Bbar)
