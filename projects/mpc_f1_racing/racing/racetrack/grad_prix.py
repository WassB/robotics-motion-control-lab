import sys
import numpy as np
from racing.racetrack.racetrack import RaceTrack
from racing.mpc.mpc import NMPC
from racing.models.kinematic_bicycle import (
    HEADING_INDEX,
    VX_INDEX,
    VY_INDEX,
    OMEGA_INDEX,
    STEERING_INDEX,
    N_INDEX,
    XI_INDEX,
    THROTTLE_INDEX,
    STEERING_RATE_INDEX
)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import perf_counter


class RacingResult:
    """
    Class to store the results of a racing session.
    It contains the states and inputs trajectories, curvilinear coordinates,
    MPC computation times, and the racetrack used.
    """

    def __init__(self, lap_time: float, states_trj: np.ndarray, inputs_trj: np.ndarray, s_values: np.ndarray, mpc_time: np.ndarray, racetrack: RaceTrack):
        self.lap_time: float = lap_time
        self.states_trj: np.ndarray = states_trj
        self.inputs_trj: np.ndarray = inputs_trj
        self.s_values: np.ndarray = s_values
        self.mpc_time: np.ndarray = mpc_time
        self.racetrack: RaceTrack = racetrack


class FreePractice:
    """
    FreePractice class to run a free practice session on a given racetrack.
    It allows to set the vehicle model and the reference trajectory for the session.
    """

    def __init__(self, racetrack: RaceTrack, mpc: "NMPC", perturbe_parameters: bool = True):
        """
        Initializes the FreePractice session.
        """
        self.racetrack = racetrack
        self.mpc = mpc

        if not self.mpc.are_references_set():
            raise ValueError(
                "References are not set in the MPC. Please set the references before starting the free practice session.")

        if perturbe_parameters:
            self.mpc.model.parameters.I_z *= np.random.uniform(0.9, 1.1)
            self.mpc.model.parameters.l_f *= np.random.uniform(0.98, 1.01)
            self.mpc.model.parameters.l_r *= np.random.uniform(0.98, 1.01)
            self.mpc.model.parameters.mass *= np.random.uniform(0.9, 1.1)
            self.mpc.model.parameters.Aref *= np.random.uniform(0.9, 1.1)
            self.mpc.model.parameters.Cd *= np.random.uniform(0.9, 1.1)
            self.mpc.model.parameters.air_density *= np.random.uniform(
                0.8, 1.2)

        self.mpc._setup()  # Ensure the MPC is set up after any parameter changes

    def animated_lap(self, s_0: float):
        """
        Runs the free practice session with an animated lap and dynamic plots of states and inputs.
        """

        # -------------------
        # Initialization
        # -------------------
        initial_state = self.mpc.x_ref_fun(s_0)
        dt = 0.020
        lap_time = 0.0
        num_frames = 1000000  # Adjust for simulation length
        num_failures = 0

        current_s = s_0
        current_x = initial_state.copy()

        n_states = self.mpc.model.n_states
        n_inputs = self.mpc.model.n_inputs

        # Pre-allocate arrays
        states_trj = np.zeros((num_frames, n_states))
        inputs_trj = np.zeros((num_frames, n_inputs))
        s_values = np.zeros((num_frames,))
        mpc_time = np.zeros((num_frames,))
        states_trj[0] = current_x
        s_values[0] = current_s

        # --- store XY for comparison plot at the end ---
        xy_driven = []   # (x, y) the car actually followed
        xy_ref = []   # (x, y) raceline reference at same s

        # -------------------
        # Racetrack figure
        # -------------------
        fig_track, ax_track = self.racetrack.draw()
        ax_track.set_aspect('equal')

        car_dot, = ax_track.plot([], [], 'ro', label="Car")
        prediction_line, = ax_track.plot(
            [], [], 'r-', linewidth=2, label="Prediction")

        # -------------------
        # Update function
        # -------------------
        def update(frame):
            nonlocal current_s, current_x, lap_time, num_failures

            ###############################################
            # Compute Optimal Input
            ###############################################
            current_kappa = self.mpc.k_ref_fun(current_s)
            reference_state = self.mpc.x_ref_fun(current_s)
            try:
                start_time = perf_counter()
                control_input = self.mpc.compute_input(
                    s_0=current_s, x_0=current_x).flatten()
                mpc_time[frame] = perf_counter() - start_time
            except Exception as exc:
                print("MPC failed at s =", current_s, "progress =",
                      current_s/self.racetrack.length*100, "%")
                print("The error returned is:", exc)
                print("Resetting MPC controller on the reference trajectory")
                current_x = reference_state.copy()
                control_input = self.mpc.compute_input(
                    s_0=current_s, x_0=current_x).flatten()
                num_failures += 1

            x_opt, _, _ = self.mpc.get_optimal_trajectory()

            ###################################################
            # Plot current state and predicted state trajectory
            ###################################################
            # Predicted trajectory
            prediction_range = self.mpc.s_range_n_1 + current_s
            # wrap beyond track length
            prediction_range = [pred_s - self.racetrack.length if pred_s > self.racetrack.length else pred_s
                                for pred_s in prediction_range]

            x_predicted = [x_opt[N_INDEX, ii] * self.racetrack.normal_vector(s)[0] + self.racetrack.position(s)[0]
                           for ii, s in enumerate(prediction_range)]
            y_predicted = [x_opt[N_INDEX, ii] * self.racetrack.normal_vector(s)[1] + self.racetrack.position(s)[1]
                           for ii, s in enumerate(prediction_range)]

            prediction_line.set_data(x_predicted, y_predicted)
            car_dot.set_data(
                [self.racetrack.position(current_s)[
                    0] + current_x[N_INDEX] * self.racetrack.normal_vector(current_s)[0]],
                [self.racetrack.position(current_s)[
                    1] + current_x[N_INDEX] * self.racetrack.normal_vector(current_s)[1]]
            )

            ###################################################
            # Store trajectories & 2D positions
            ###################################################
            states_trj[frame] = current_x.copy()
            inputs_trj[frame] = control_input
            s_values[frame] = current_s

            # --- append driven and raceline XY at current s ---
            pos_s = self.racetrack.position(current_s)
            normal = self.racetrack.normal_vector(current_s)

            x_cur = pos_s[0] + current_x[N_INDEX] * normal[0]
            y_cur = pos_s[1] + current_x[N_INDEX] * normal[1]
            xy_driven.append((x_cur, y_cur))

            x_ref_s = pos_s[0] + reference_state[N_INDEX] * normal[0]
            y_ref_s = pos_s[1] + reference_state[N_INDEX] * normal[1]
            xy_ref.append((x_ref_s, y_ref_s))

            ###################################################
            # Advance simulation
            ###################################################
            current_s += self.mpc.model.s_dot(current_x, current_kappa) * dt
            lap_time += dt
            current_x = self.mpc.model.step_dt(
                current_x, control_input, kappa=current_kappa, dt=dt).full().flatten()

            if current_s >= self.racetrack.length:
                ani.event_source.stop()
                print("Lap completed!")
                print("Total lap time:", lap_time, "seconds")
                print("Average cpu time per mpc call:",
                      np.mean(mpc_time), "seconds")
                print("Number of MPC failures during the lap:", num_failures)

                # --- build arrays and plot comparison ---
                xy_driven_np = np.array(xy_driven)
                xy_ref_np = np.array(xy_ref)

                fig_cmp, ax_cmp = self.racetrack.draw()
                ax_cmp.set_aspect('equal')
                ax_cmp.plot(xy_ref_np[:, 0],    xy_ref_np[:, 1],
                            '--', linewidth=2, label='Raceline')
                ax_cmp.plot(
                    xy_driven_np[:, 0], xy_driven_np[:, 1], '-',  linewidth=2, label='Driven')
                ax_cmp.set_title("Raceline vs Driven Trajectory")

                _, labels = ax_cmp.get_legend_handles_labels()
                if labels:
                    ax_cmp.legend(loc='best')

                ax_cmp.grid(True)

                # show without blocking the animation window immediately
                plt.show(block=False)

                return [car_dot, prediction_line]

            return [car_dot, prediction_line]

        # -------------------
        # Animate
        # -------------------
        ani = FuncAnimation(fig_track, update,
                            frames=num_frames, interval=30, blit=True)
        plt.show()

        result = RacingResult(
            lap_time=lap_time,
            states_trj=states_trj.T,
            inputs_trj=inputs_trj.T,
            s_values=s_values,
            mpc_time=mpc_time,
            racetrack=self.racetrack
        )

        return result

    def fast_lap(self, s_0: float):
        """
        fast run
        """

        # -------------------
        # Initialization
        # -------------------
        initial_state = self.mpc.x_ref_fun(s_0)
        dt = 0.020
        lap_time = 0.0
        num_frames = 100000  # Adjust for simulation length
        num_failures = 0

        current_s = s_0
        current_x = initial_state.copy()

        n_states = self.mpc.model.n_states
        n_inputs = self.mpc.model.n_inputs

        # Pre-allocate arrays
        states_trj = np.zeros((num_frames, n_states))
        inputs_trj = np.zeros((num_frames, n_inputs))
        s_values = np.zeros((num_frames,))
        mpc_time = np.zeros((num_frames,))
        states_trj[0] = current_x
        s_values[0] = current_s

        for frame in range(num_frames):
            ###############################################
            # Compute Optimal Input
            ###############################################
            current_kappa = self.mpc.k_ref_fun(current_s)
            reference_state = self.mpc.x_ref_fun(current_s)
            try:
                start_time = perf_counter()
                control_input = self.mpc.compute_input(
                    s_0=current_s, x_0=current_x).flatten()
                mpc_time[frame] = perf_counter() - start_time
            except Exception as exc:
                print("MPC failed at s =", current_s, "progress =",
                      current_s/self.racetrack.length*100, "%")
                print("The error returned is:", exc)
                print("Resetting MPC controller on the reference trajectory")
                current_x = reference_state.copy()
                control_input = self.mpc.compute_input(
                    s_0=current_s, x_0=current_x).flatten()
                num_failures += 1
                return

            ###################################################
            # Store trajectories
            ###################################################
            states_trj[frame] = current_x.copy()
            inputs_trj[frame] = control_input
            s_values[frame] = current_s

            print("Percentage progress along the track:",
                  (current_s/self.racetrack.length)*100)
            print("state error norm:", np.linalg.norm(
                current_x - reference_state))

            ###################################################
            # Advance simulation
            ###################################################
            current_s += self.mpc.model.s_dot(current_x, current_kappa) * dt
            lap_time += dt
            current_x = self.mpc.model.step_dt(
                current_x, control_input, kappa=current_kappa, dt=dt).full().flatten()

            if current_s >= self.racetrack.length:
                print("Lap completed!")
                print("Total lap time:", lap_time, "seconds")
                print("Average cpu time per mpc call:",
                      np.mean(mpc_time), "seconds")
                print("Number of MPC failures during the lap:", num_failures)
                break

        result = RacingResult(
            lap_time=lap_time,
            states_trj=states_trj.T,
            inputs_trj=inputs_trj.T,
            s_values=s_values,
            mpc_time=mpc_time,
            racetrack=self.racetrack
        )

        return result


class GrandPrix:
    """
    GrandPrix class to run a competitive session between two vehicles
    on the same racetrack using two separate MPC controllers.
    """

    def __init__(self, racetrack: RaceTrack, mpc1: "NMPC", mpc2: "NMPC"):
        """
        Initializes the GrandPrix session.
        """
        self.racetrack = racetrack
        self.mpc1 = mpc1
        self.mpc2 = mpc2

        for mpc in [mpc1, mpc2]:
            if not mpc.are_references_set():
                raise ValueError("References are not set in one of the MPCs.")

    def animated_race(self, s0_1: float, s0_2: float):
        """
        Runs the animated race with dynamic state and input plots for both vehicles.
        """
        # -------------------
        # Initialization
        # -------------------
        dt = 0.013
        num_frames = 1000
        lap_time = 0.

        n_states = self.mpc1.model.n_states
        n_inputs = self.mpc1.model.n_inputs

        # Initial states
        x1 = self.mpc1.x_ref_fun(s0_1).copy()
        x2 = self.mpc2.x_ref_fun(s0_2).copy()
        s1 = s0_1
        s2 = s0_2

        states1 = np.zeros((num_frames, n_states))
        states2 = np.zeros((num_frames, n_states))
        inputs1 = np.zeros((num_frames, n_inputs))
        inputs2 = np.zeros((num_frames, n_inputs))
        s_values1 = np.zeros((num_frames,))
        s_values2 = np.zeros((num_frames,))

        states1[0] = x1
        states2[0] = x2
        s_values1[0] = s1
        s_values2[0] = s2

        # -------------------
        # Racetrack figure
        # -------------------
        fig_track, ax_track = self.racetrack.draw()
        ax_track.set_aspect('equal')

        car1_dot, = ax_track.plot([], [], 'ro', label="Car 1")
        car2_dot, = ax_track.plot([], [], 'bo', label="Car 2")
        pred1_line, = ax_track.plot(
            [], [], 'r-', linewidth=2, label="Prediction 1")
        pred2_line, = ax_track.plot(
            [], [], 'b-', linewidth=2, label="Prediction 2")

        # -------------------
        # States figure
        # -------------------
        fig_states, axes_states = plt.subplots(
            n_states, 1, figsize=(6, 2*n_states), sharex=True)
        state_lines1 = [axes_states[i].plot([], [], 'r', label=f"x{i} car1")[
            0] for i in range(n_states)]
        state_lines2 = [axes_states[i].plot([], [], 'b', label=f"x{i} car2")[
            0] for i in range(n_states)]
        state_names = [r'$\psi$', r'$v_x$', r'$v_y$',
                       r'$\dot{\psi}$', r'$\delta$', r'$s$', r'$\xi$']
        for i, ax in enumerate(axes_states):
            ax.set_ylabel(state_names[i])
            ax.grid(True)

        # -------------------
        # Inputs figure
        # -------------------
        fig_inputs, axes_inputs = plt.subplots(
            n_inputs, 1, figsize=(6, 2*n_inputs), sharex=True)
        input_lines1 = [axes_inputs[i].plot(
            [], [], 'r--', label=f"u{i} car1")[0] for i in range(n_inputs)]
        input_lines2 = [axes_inputs[i].plot(
            [], [], 'b--', label=f"u{i} car2")[0] for i in range(n_inputs)]
        input_names = [r'$F_{throttle}$', r'$\dot{\delta}$']
        for i, ax in enumerate(axes_inputs):
            ax.set_ylabel(input_names[i])
            ax.grid(True)

        # -------------------
        # Update function
        # -------------------
        def update(frame):
            nonlocal x1, x2, s1, s2

            # -------------------
            # Compute control inputs
            # -------------------
            kappa1 = self.mpc1.k_ref_fun(s1)
            kappa2 = self.mpc2.k_ref_fun(s2)

            u1 = self.mpc1.compute_input(s_0=s1, x_0=x1).flatten()
            u2 = self.mpc2.compute_input(s_0=s2, x_0=x2).flatten()

            self.mpc1.get_optimal_trajectory()
            self.mpc2.get_optimal_trajectory()

            # -------------------
            # Predicted trajectories
            # -------------------
            x_pred1 = [self.mpc1.x_opt[N_INDEX, ii] * self.racetrack.normal_vector(s)[0] + self.racetrack.position(s)[0]
                       for ii, s in enumerate(s1 + self.mpc1.s_range_n_1)]
            y_pred1 = [self.mpc1.x_opt[N_INDEX, ii] * self.racetrack.normal_vector(s)[1] + self.racetrack.position(s)[1]
                       for ii, s in enumerate(s1 + self.mpc1.s_range_n_1)]

            x_pred2 = [self.mpc2.x_opt[N_INDEX, ii] * self.racetrack.normal_vector(s)[0] + self.racetrack.position(s)[0]
                       for ii, s in enumerate(s2 + self.mpc2.s_range_n_1)]
            y_pred2 = [self.mpc2.x_opt[N_INDEX, ii] * self.racetrack.normal_vector(s)[1] + self.racetrack.position(s)[1]
                       for ii, s in enumerate(s2 + self.mpc2.s_range_n_1)]

            pred1_line.set_data(x_pred1, y_pred1)
            pred2_line.set_data(x_pred2, y_pred2)

            # -------------------
            # Update car positions
            # -------------------
            car1_dot.set_data([self.racetrack.position(s1)[0] + x1[N_INDEX] * self.racetrack.normal_vector(s1)[0]],
                              [self.racetrack.position(s1)[1] + x1[N_INDEX] * self.racetrack.normal_vector(s1)[1]])
            car2_dot.set_data([self.racetrack.position(s2)[0] + x2[N_INDEX] * self.racetrack.normal_vector(s2)[0]],
                              [self.racetrack.position(s2)[1] + x2[N_INDEX] * self.racetrack.normal_vector(s2)[1]])

            # -------------------
            # Store trajectories
            # -------------------
            states1[frame] = x1
            states2[frame] = x2
            inputs1[frame] = u1
            inputs2[frame] = u2
            s_values1[frame] = s1
            s_values2[frame] = s2

            # -------------------
            # Update state subplots
            # -------------------
            for i in range(n_states):
                state_lines1[i].set_data(
                    s_values1[:frame+1], states1[:frame+1, i])
                state_lines2[i].set_data(
                    s_values2[:frame+1], states2[:frame+1, i])
                axes_states[i].relim()
                axes_states[i].autoscale_view()

            # -------------------
            # Update input subplots
            # -------------------
            for i in range(n_inputs):
                input_lines1[i].set_data(
                    s_values1[:frame+1], inputs1[:frame+1, i])
                input_lines2[i].set_data(
                    s_values2[:frame+1], inputs2[:frame+1, i])
                axes_inputs[i].relim()
                axes_inputs[i].autoscale_view()

            # -------------------
            # Advance simulation
            # -------------------
            s1 += self.mpc1.model.s_dot(x1, kappa1) * dt
            s2 += self.mpc2.model.s_dot(x2, kappa2) * dt
            x1 = self.mpc1.model.step_dt(
                x1, u1, kappa=kappa1, dt=dt).full().flatten()
            x2 = self.mpc2.model.step_dt(
                x2, u2, kappa=kappa2, dt=dt).full().flatten()

            return [car1_dot, car2_dot, pred1_line, pred2_line] + state_lines1 + state_lines2 + input_lines1 + input_lines2

        # -------------------
        # Animate
        # -------------------
        ani = FuncAnimation(fig_track, update,
                            frames=num_frames, interval=10, blit=True)
        plt.show()


def post_racine_analysis_and_scoring(racing_result: RacingResult, mpc_controller: NMPC):
    """
    Analyzes the racing results and computes the scoring based on performance metrics.
    """
    # finish this (if needed for your report)
