import numpy as np
from numpy.typing import NDArray
from robot_model import robot_step


def simulation_trajectory(init : NDArray[np.float64] ,goal : NDArray[np.float64] ,delta_t : float,T : float) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """ Simulate robot trajectory from init toward goal

    Returns
    -------
    time : ndarray,shape (N,)
    state_history : ndarray, shape (N,3)

    """    
    nb_steps = int(round(T / delta_t))

    nb_points : int = nb_steps + 1

    time : NDArray[np.float64] = np.linspace(0.0,T,nb_points, dtype=np.float64)
    state_history : NDArray[np.float64] = np.zeros((nb_points,3),dtype=np.float64)

    state_history[0] = init
    
    for step in range(nb_steps):

        state_history[step+1] = robot_step(state_history[step],goal,delta_t)
        
    return time,state_history