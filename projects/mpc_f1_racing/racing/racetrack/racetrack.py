import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import os
import yaml
from scipy.interpolate import interp1d


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from racing.mpc.mpc_first import NMPC  # imported only during type checking


# This module is used to create b-splines and optimize them.
# Functions are inspired by : https://pomax.github.io/bezierinfo/


# https://pomax.github.io/bezierinfo/#:~:text=the%20Inconsolata%20typeface.-,previoustable%20of%20contentsnext,B%C3%A9zier%20curvatures%20as%20matrix%20operations,-We%20can%20also

class CubicBspline:
    def __init__(self, p_0: np.ndarray, p_1: np.ndarray, p_2: np.ndarray, p_3: np.ndarray):
        """
        Creates a cubic spline from four control points. The spline ranges the intrisic coordinate [0,1]. You can use the scaling factor to change the range to 
        [0, 1/scaling_factor]. This is useful if you need to conctanetate multiple splines to obtain a long path where the curve is still parameterized between
        [0,1]. The control points are given in the order p_0, p_1, p_2, p_3.

        :param p_0: First control point (start control point) 
        :type p_0: numpy array of shape (2,)
        :param p_1: Second control point (numpy array of shape (2,))
        :param p_2: Third control point (numpy array of shape (2,))
        :param p_3: Fourth control point (end control point) (numpy array of shape (2,))

        :raises ValueError: If any of the control points are not 2D arrays

        """

        self.P = np.vstack([p_0.flatten(), p_1.flatten(),
                           p_2.flatten(), p_3.flatten()])
        # Bézier basis matrix
        self.M_B = np.array([[-1, 3, -3, 1],
                             [3, -6, 3, 0],
                             [-3, 3, 0, 0],
                             [1, 0, 0, 0]
                             ])

        self.t_table, self.arc_length = self._get_tables()
        self._s2t = interp1d(self.arc_length, self.t_table,
                             bounds_error=True)  # Maps arc length to t
        self._t2s = interp1d(self.t_table, self.arc_length,
                             bounds_error=True)  # Maps t to arc length

        # Length of the Bézier curve, which is the arc length at t=1.0
        self.length = self.t2s(1.0)

    def t2s(self, t: float):
        """
        Maps the parameter t to the arc length s using the precomputed mapping.
        :param t: Parameter t in the range [0, 1]
        :return: Corresponding arc length s
        """
        t = np.clip(t, 0., 1.)  # Ensure t is within the valid range [0, 1]
        # Returns the arc length corresponding to the parameter t
        return self._t2s(t)

    def s2t(self, s: float):
        """
        Maps the arc length s to the parameter t using the precomputed mapping.
        :param s: Arc length
        :return: Corresponding parameter t
        """
        s = np.clip(s, 0., self.length)  # Ensure s is within the valid range
        return self._s2t(s)

    def bezier_point(self, t):
        """ Computes the point on the Bézier curve at parameter t."""
        T = np.array([t**3, t**2, t, 1])
        return (T @ self.M_B @ self.P)  # Returns [x(t), y(t)]

    def velocity(self, t):
        """ Computes the velocity vector at parameter t. Velocity is with respect to the coordinate t as dx/dt"""
        T = np.array([3*t**2, 2*t, 1, 0])
        return (T @ self.M_B @ self.P)

    def acceleration(self, t):
        """ Computes the acceleration vector at parameter t. Acceleration is with respect to the coordinate t. ddx/ddt"""
        T = np.array([6*t, 2, 0, 0])
        return (T @ self.M_B @ self.P)

    def tangent_vector(self, t):
        """ Computes the unit tangent vector at parameter t."""
        vel = self.velocity(t)
        norm_2 = vel[0]**2 + vel[1]**2
        if norm_2 == 0:
            return np.array([0, 0])
        return vel / np.sqrt(norm_2)

    def heading(self, t):
        """ Computes the heading angle at parameter t."""
        tangent = self.tangent_vector(t)
        heading = np.arctan2(tangent[1], tangent[0])

        return heading

    def normal_vector(self, t):
        """ Computes the unit normal vector at parameter t."""
        tangent = self.tangent_vector(t)
        return np.array([-tangent[1], tangent[0]])

    def curvature(self, t):
        """ Computes the curvature at parameter t. Curvature is defined as the rate of change of the heading angle with respect to dt. Hence this is dtheta/dt"""
        vel = self.velocity(t)
        acc = self.acceleration(t)
        norm_vel = np.sqrt(vel[0]**2 + vel[1]**2)

        if norm_vel == 0:
            return 0.0

        cross_prod = vel[0] * acc[1] - vel[1] * acc[0]
        curvature = cross_prod / (norm_vel ** 3)

        return curvature

    def _get_tables(self):
        """
        Computes the length of the Bézier curve by numerical integration.
        :return: Length of the Bézier curve
        """
        t_values = np.linspace(0, 1., 1000)
        length_values = np.zeros(len(t_values))

        length = 0.

        for ii in range(len(t_values) - 1):
            t0 = t_values[ii]
            t1 = t_values[ii + 1]
            p0 = self.bezier_point(t0)
            p1 = self.bezier_point(t1)
            # Euclidean distance between two points
            length += np.linalg.norm(p1 - p0)
            length_values[ii + 1] = length

        return t_values, length_values

    def draw(self, ax=None, mode: str = "position", with_control_points: bool = False):

        if mode not in ["position", "velocity", "acceleration"]:
            raise ValueError(
                "Mode must be 'position', 'velocity', or 'acceleration'.")

        # Generate points along the Bézier curve
        t_values = np.linspace(0, 1., 100000)
        if mode == "position":
            curve_points = np.array([self.bezier_point(t) for t in t_values])
        elif mode == "velocity":
            curve_points = np.array([self.velocity(t) for t in t_values])
        elif mode == "acceleration":
            curve_points = np.array([self.acceleration(t) for t in t_values])

        if ax is None:
            fig, ax = plt.subplots()

        # Plot the Bézier curve
        ax.plot(curve_points[:, 0], curve_points[:, 1], c="b")

        if with_control_points:
            # Plot control points
            if mode == "position":
                ax.plot(self.P[:, 0], self.P[:, 1],
                        'ro--', label='Control Points')

        ax.set_title('Bézier Curve from Control Points')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        ax.grid()
        ax.axis('equal')


class ParametricCubicBspline:

    def __init__(self, p_start: np.ndarray, p_end: np.ndarray):

        self.p_start = np.array([[p_start[0], p_start[1]]])
        self.p_end = np.array([[p_end[0], p_end[1]]])

        self.p_1 = None
        self.p_2 = None

        # Bézier basis matrix
        self.M_B = np.array([[-1, 3, -3, 1],
                             [3, -6, 3, 0],
                             [-3, 3, 0, 0],
                             [1, 0, 0, 0]
                             ])

    def is_initialized(self):
        return self.p_1 is not None and self.p_2 is not None

    def set_parametric_control_points(self, p_1: ca.MX, p_2: ca.MX):
        self.p_1 = p_1
        self.p_2 = p_2
        self.P = ca.vcat([self.p_start, self.p_1, self.p_2, self.p_end])

    def velocity(self, t):
        if not self.is_initialized():
            raise ValueError("Control points are not initialized.")

        T = np.array([[3*t**2, 2*t, 1, 0]])
        return ca.mtimes(T, ca.mtimes(self.M_B, self.P))

    def acceleration(self, t):
        if not self.is_initialized():
            raise ValueError("Control points are not initialized.")

        T = np.array([[6*t, 2, 0, 0]])
        return ca.mtimes(T, ca.mtimes(self.M_B, self.P))


class RaceTrack:
    def __init__(self, interpolation_points: np.ndarray, track_width: float = 1., name="RaceTrack"):
        """
        Creates a race track from a given set of interpolation points.
        :param interpolation_points: A 2D numpy array of shape (n, 2) where n is the number of interpolation points.
        :type interpolation_points: np.ndarray
        """

        self.splines: list[CubicBspline] = []
        self.interpolation_points: np.ndarray = np.array(interpolation_points)
        self.track_width: float = track_width
        self.n_splines: int = 0
        self.name: str = name

        self.ax: plt.Axes = None
        self.fig: plt.Figure = None

        # optimal reference trajectory for the given racetrack
        # optimal reference input profile for each s -> u_ref(s)
        self.u_ref_fun: interp1d = None
        # optimal reference state profile for each s -> x_ref(s)
        self.x_ref_fun: interp1d = None
        # interpolated curvature of the track for each s -> k_ref(s) (this is used for fast retrival of the curvature)
        self.k_ref_fun: interp1d = None

        # curvilinear coordinate s for each point on the racetrack
        self.s_values: np.ndarray = None

        # computes the racetrack (setting the splines for the racetrack)
        self._generate_racetrack()

        # compute the length of the racetrack and give presentation of the racetrack
        self.length: float = self.compute_length()
        self.progress_vector: np.ndarray = np.append(
            np.array([0]), np.cumsum([spline.length for spline in self.splines]))
        print(self)

    @classmethod
    def get_racetrack(cls, name: str):

        file_name = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), "maps", "arrays", name + ".yaml")

        try:
            loaded_data = yaml.safe_load(open(file_name, "r"))
            points = np.array(loaded_data['points'])
            track_width = loaded_data['track_width']

            print("Track width:", track_width)
            print("Points shape:", points.shape)
        except Exception as e:
            avilable_maps = os.listdir(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "maps", "arrays"))
            raise ValueError(
                f"Racetrack '{name}' not found. Check that the track is available. Available maps are {avilable_maps}")

        return RaceTrack(interpolation_points=points, track_width=track_width, name=name)

    def compute_length(self):
        """
        Computes the total length of racetrack
        """
        length = 0
        for spline in self.splines:
            length += spline.length

        return length

    def _generate_racetrack(self):
        """
        Generates the race track by creating B-spline segments between the interpolation points.
        """

        first_point: np.ndarray = self.interpolation_points[0]
        last_point: np.ndarray = self.interpolation_points[-1]
        if not np.array_equal(first_point, last_point):
            # add first point to the end of the list in order toc lose the loop
            self.interpolation_points = np.vstack(
                [self.interpolation_points, first_point.copy()])

        n_points = self.interpolation_points.shape[0]
        if n_points < 3:
            raise ValueError(
                "At least three interpolation points are required to create minimum curvature B-splines.")

        parameteric_splines: list[ParametricCubicBspline] = []
        for jj in range(n_points-1):
            p_start = self.interpolation_points[jj]
            p_end = self.interpolation_points[jj+1]
            spline = ParametricCubicBspline(p_start, p_end)
            parameteric_splines.append(spline)

        optimizer = ca.Opti()
        self.n_splines = n_points - 1

        # set variables for each spline
        for spline in parameteric_splines:

            # each spline has four control points.
            # the middle points p_1 and p_2 are the control variables
            # the end points are instead fixed values

            p_1 = optimizer.variable(1, 2)
            p_2 = optimizer.variable(1, 2)

            p_start = self.interpolation_points[jj]
            p_end = self.interpolation_points[jj+1]
            difference = p_end - p_start

            # set iniital position of control point on the axis from p_start to p_end
            optimizer.set_initial(p_1, p_start + difference * 0.33)
            optimizer.set_initial(p_2, p_start + difference * 0.66)
            spline.set_parametric_control_points(p_1, p_2)

        # set C2 continuity constraints
        for jj, spline in enumerate(parameteric_splines[:-1]):

            next_spline = parameteric_splines[jj+1]

            # C2 continuity at the end of the current spline and start of the next spline
            optimizer.subject_to(spline.velocity(
                1.) == next_spline.velocity(0.))
            optimizer.subject_to(spline.acceleration(
                1.) == next_spline.acceleration(0.))

        # set closure constraint
        optimizer.subject_to(parameteric_splines[0].velocity(
            0.) == parameteric_splines[-1].velocity(1.))
        optimizer.subject_to(parameteric_splines[0].acceleration(
            0.) == parameteric_splines[-1].acceleration(1.))

        # minimize sum of the curvature at the interpolation points
        curvature_cost = 0.
        for jj, spline in enumerate(parameteric_splines):

            for t in np.linspace(0., 1., 10):
                # curvature at the start point
                velocity_norm2 = spline.velocity(
                    t)[0]**2 + spline.velocity(t)[1]**2
                cross_prod = (spline.velocity(t)[
                              0]*spline.acceleration(t)[1] - (spline.velocity(t)[1]*spline.acceleration(t)[0]))
                curvature = (cross_prod*cross_prod) / \
                    (velocity_norm2**(3/2) + 1E-6)

                # minimize the inverse of the curvature to avoid high curvature values
                curvature_cost += (100*curvature + 2E-6)**2

        optimizer.minimize(curvature_cost)
        optimizer.solver('ipopt')

        solution = optimizer.solve()

        # create splines from optimized control points
        for jj, spline in enumerate(parameteric_splines):
            p_1 = solution.value(spline.p_1)
            p_2 = solution.value(spline.p_2)

            p_start = spline.p_start
            p_end = spline.p_end

            # scaling allows to have the full race track parameterized between [0,1]
            optimal_spline = CubicBspline(p_start, p_1, p_2, p_end)
            self.splines.append(optimal_spline)

    def draw(self, with_frames: bool = False, with_curvature: bool = False):

        resolution_points = 1000

        # Draw the optimal splines
        self.fig, self.ax = plt.subplots()
        for spline in self.splines:
            spline.draw(ax=self.ax, mode="position")

        left_border = []
        right_border = []
        for t in np.linspace(0., self.length, resolution_points):
            point = self.position(t)
            normal = self.normal_vector(t)
            # Draw the border points
            left_border.append(point + normal * self.track_width/2)
            right_border.append(point - normal * self.track_width/2)

        left_border = np.array(left_border)
        right_border = np.array(right_border)

        self.ax.plot(left_border[:, 0], left_border[:, 1], c='b')
        self.ax.plot(right_border[:, 0], right_border[:, 1], c='b')

        for t in np.linspace(0., self.length, int(self.length/50)):
            point = self.position(t)
            normal = self.normal_vector(t)
            self.ax.text(point[0] + normal[0] * self.track_width*1, point[1] + normal[1] *
                         self.track_width*1, f"{t:.2f}", fontsize=8, ha='center', va='center', color='black')

        if with_frames:
            # Draw the frames of the splines

            for t in np.linspace(0., self.length, resolution_points):
                point = self.position(t)
                tangent = self.tangent_vector(t)
                normal = self.normal_vector(t)
                # Draw the frame
                self.ax.arrow(point[0], point[1], tangent[0] * 0.3, tangent[1] * 0.3,
                              head_width=0.05, head_length=0.1, fc='r', ec='r', width=0.005)
                self.ax.arrow(point[0], point[1], normal[0] * 0.3, normal[1] * 0.3,
                              head_width=0.05, head_length=0.1, fc='b', ec='b', width=0.005)

        if with_curvature:
            # Draw curvature combs
            for t in np.linspace(0., self.length, resolution_points):
                point = self.position(t)
                normal = self.normal_vector(t)
                curvature = self.curvature(t)
                # Draw the curvature comb
                self.ax.arrow(point[0], point[1], normal[0] * np.clip(1/curvature, -100, 100), normal[1] * np.clip(
                    1/curvature, -100, 100), head_width=0.05, head_length=0.1, fc='g', ec='g', width=0.005)

        # put the start
        start_point = self.position(0.0)
        self.ax.scatter(start_point[0], start_point[1],
                        s=100, color='k', label="Start")

        self.ax.set_aspect('equal', adjustable='box')

        return self.fig, self.ax

    def which_spline(self, s: float):
        if s == 0.0:
            return 0
        if s == self.length:
            return self.n_splines - 1

        spline_index = np.searchsorted(self.progress_vector, s, side='right')-1

        return spline_index

    def position(self, s: float):
        """
        Position along the racetrack with s >= 0 and s<= length
        """

        spline_number = self.which_spline(s)
        spline = self.splines[spline_number]
        # Normalize t to [0, 1] within the spline
        t = spline.s2t((s - self.progress_vector[spline_number]))

        return self.splines[spline_number].bezier_point(t)

    def _velocity(self, s: float):
        """
          Velocity along the racetrack with s >= 0 and s<= length
        """
        spline_number = self.which_spline(s)
        spline = self.splines[spline_number]
        # Normalize t to [0, 1] within the spline
        t = spline.s2t((s - self.progress_vector[spline_number]))

        # normalize velocity to meters unit. velocity of the spline is dp/dt but you want dp/dm
        return self.splines[spline_number].velocity(t)/np.linalg.norm(self.splines[spline_number].velocity(t))

    def _acceleration(self, s):

        spline_number = self.which_spline(s)
        spline = self.splines[spline_number]
        # Normalize t to [0, 1] within the spline
        t = spline.s2t((s - self.progress_vector[spline_number]))

        # normalize acceleration to meters unit squared
        return self.splines[spline_number].acceleration(t)/np.linalg.norm(self.splines[spline_number].velocity(t))**2

    def tangent_vector(self, s):

        spline_number = self.which_spline(s)
        spline = self.splines[spline_number]
        # Normalize t to [0, 1] within the spline
        t = spline.s2t((s - self.progress_vector[spline_number]))

        return self.splines[spline_number].tangent_vector(t)

    def heading(self, s):

        spline_number = self.which_spline(s)
        spline = self.splines[spline_number]
        # Normalize t to [0, 1] within the spline
        t = spline.s2t((s - self.progress_vector[spline_number]))

        return self.splines[spline_number].heading(t)

    def heading_rate(self, s):

        curvature = self.curvature(s)
        velocity = self._velocity(s)

        # Normalize by the length of the track to get a normalized heading rate
        return (curvature * (np.linalg.norm(velocity)))

    def normal_vector(self, s):

        spline_number = self.which_spline(s)
        spline = self.splines[spline_number]
        # Normalize t to [0, 1] within the spline
        t = spline.s2t((s - self.progress_vector[spline_number]))

        return self.splines[spline_number].normal_vector(t)

    def curvature(self, s):

        vel = self._velocity(s)
        acc = self._acceleration(s)
        norm_vel = np.sqrt(vel[0]**2 + vel[1]**2)

        if norm_vel == 0:
            return 0.0

        cross_prod = vel[0] * acc[1] - vel[1] * acc[0]
        curvature = cross_prod / (norm_vel ** 3 + 1E-6)

        return curvature  # Normalize by the length of the track to get a normalized curvature

    def __repr__(self):

        msg = ""
        msg += "RaceTrack representation:\n"
        msg += f"Number of splines : {self.n_splines} \n"
        msg += f"Track length      : {self.length:.2f} m \n"
        msg += f"Track width       : {self.track_width:.2f} m \n"
        msg += f"Name              : {self.name}"

        return msg

    def __str__(self):
        msg = ""
        msg += "RaceTrack representation:\n"
        msg += f"Number of splines : {self.n_splines} \n"
        msg += f"Track length      : {self.length:.2f} m \n"
        msg += f"Track width       : {self.track_width:.2f} m \n"
        msg += f"Name              : {self.name}"

        return msg
