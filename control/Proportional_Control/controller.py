import numpy as np
from numpy.typing import NDArray


def wrap_angle(angle : float) -> float:
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


def control_law(state: NDArray[np.float64],goal: NDArray[np.float64]) -> tuple[float,float]: 
    """Compute linear and angular velocity commands for a goal-to-goal task. 

        Parameters
        ----------
        state : array_like, shape(3,)
            Robot state [x,y,theta]

        goal : array_like, shape(2,)
                Goal position [x_goal,y_goal]

        Returns
        -------
        v : float 
            Linear velocity command [m/s]
        omega : float 
            Angular velocity command [m/s]
    """

    x,y,theta = state
    x_g,y_g = goal

    dx = float(x_g - x)
    dy = float (y_g - y)

    diff = np.array([dx,dy])

    dist = float(np.linalg.norm(diff))
    desired_heading = float(np.arctan2(dy,dx))
    heading_error = wrap_angle(desired_heading-theta)

    k_v : float = 1.0
    k_omega : float = 2.0

    v_max : float = 1.0
    omega_max : float = 2.0

    v : float = k_v * dist * max(0.0,np.cos(heading_error))
    omega : float = k_omega * heading_error

    v = float(np.clip(v,0.0,v_max))
    omega = float(np.clip(omega,-omega_max,omega_max))

    return v, omega