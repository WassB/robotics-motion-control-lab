import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

@dataclass(frozen=True)
class ControllerParams:
    k_v : float = 1.0
    k_omega : float = 2.0
    v_max: float = 1.0
    omega_max: float = 2.0
    goal_tolerance: float = 0.1



def wrap_angle(angle: float) -> float:
    """
    Compute the corresponding angle in the interval [-pi,pi]

    Parameters
    ----------
    Angle : radians

    Return
    ------
    Angle [-pi,pi] : radians    

    """
    return (angle + np.pi ) % (2.0 * np.pi) - np.pi


def control_law(state: NDArray[np.float64],goal: NDArray[np.float64],params: ControllerParams) -> NDArray[np.float64]: 
    """Compute linear and angular velocity commands for a goal-to-goal task. 

        Parameters
        ----------
        state : array_like, shape(3,)
            Robot state [x,y,theta]

        goal : array_like, shape(2,)
                Goal position [x_goal,y_goal]

        Returns
        -------
        control : NDArray[np.float64] [
        v : float 
            Linear velocity command [m/s]
        omega : float 
            Angular velocity command [rad/s]
            ]
    """

    x,y,theta = state
    x_g,y_g = goal

    dx = x_g - x
    dy = y_g - y

    diff = np.array([dx,dy])

    dist = float(np.linalg.norm(diff))
    desired_heading = float(np.arctan2(dy,dx))
    heading_error = wrap_angle(desired_heading-theta)

    v : float = params.k_v * dist * max(0.0,np.cos(heading_error))
    omega : float = params.k_omega * heading_error

    v = float(np.clip(v,0.0,params.v_max))
    omega = float(np.clip(omega,-params.omega_max,params.omega_max))

    control = np.array([v,omega], dtype=np.float64)

    return control