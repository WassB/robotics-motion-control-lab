from plotting import plotting_trajectrory
from simulation import simulation_trajectory
import numpy as  np



def main() -> None :

    #initial position
    x0 : float = 0.0 
    y0 : float = 0.0
    theta0 : float = -np.pi

    init = np.array([x0,y0,theta0], dtype=np.float64)

    #goal coordinates
    xg : float = 10.0
    yg : float = 15.0
    goal = np.array([xg,yg],dtype=np.float64)

    #time sample
    delta_t : float = 0.05 

    #time of the simulation
    T : float = 500.0

    time, state_history = simulation_trajectory(init, goal, delta_t, T)
    plotting_trajectrory(time, state_history, goal)
    

if __name__ == "__main__":
    main()







    








