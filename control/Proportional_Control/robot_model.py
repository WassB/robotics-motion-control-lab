#robot_model
import numpy as np
from controller import control_law
from numpy.typing import NDArray
from controller import wrap_angle

def robot_step(previous_state : NDArray[np.float64], goal : NDArray[np.float64],  delta_t :float) -> NDArray[np.float64]: 
    """ Propagate robot state by one time step using Euler integration"""

    x,y,theta  = previous_state
    
    v,omega = control_law(previous_state,goal)

    x_next = float(x + delta_t * v*np.cos(theta))
    y_next  = float(y + delta_t * v * np.sin(theta))
    theta_next = wrap_angle(float(theta + delta_t * omega))
    
    return np.array([x_next, y_next, theta_next],  dtype=np.float64)

















