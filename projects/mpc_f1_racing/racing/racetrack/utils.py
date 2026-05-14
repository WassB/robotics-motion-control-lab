import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from racing.racetrack.racetrack import RaceTrack

from racing.models.kinematic_bicycle import (
                            HEADING_INDEX,
                            VX_INDEX,
                            VY_INDEX,
                            OMEGA_INDEX,
                            STEERING_INDEX,
                            N_INDEX, 
                            XI_INDEX, 
                            THROTTLE_INDEX,
                            STEERING_RATE_INDEX)




def plot_states(curvilinear_coordinate : np.ndarray, states: np.ndarray, curvature_centerline : np.ndarray, title: str = "Optimal Racing Line States"):
    """
    Plots the states of the vehicle along the raceline.

    :param racetrack             : The RaceTrack object containing the track geometry.
    :param curvilinear_coordinate: The curvilinear coordinates along the raceline.
    :param curvature_centerline  : The curvature of the raceline at each point.
    :param states                : The states of the vehicle along the raceline (7, N).
    """


    fig, ax = plt.subplots(4, 2, figsize=(10, 10))

    
    # plot initial state and control guesses
    ax[0, 0].plot(curvilinear_coordinate, states[HEADING_INDEX, :],color = 'r', marker='o')
    
    ax[0, 0].set_title('Heading')
    ax[0, 0].set_xlabel('s (m)')
    ax[0, 0].set_ylabel(r'$\psi$ (rad)')
    ax[0, 0].grid()
    
    ax[1, 0].plot(curvilinear_coordinate,states[VX_INDEX, :],color = 'r', marker='o')
    
    ax[1, 0].set_title('Longitudinal Velocity')
    ax[1, 0].set_xlabel('s (m)')
    ax[1, 0].set_ylabel(r'$v_x$ (m/s)')
    ax[1, 0].grid()
    
    ax[2, 0].plot(curvilinear_coordinate,states[VY_INDEX, :],color = 'r', marker='o' )
    ax[2, 0].set_title('Lateral Velocity')
    ax[2, 0].set_xlabel('s (m)')
    ax[2, 0].set_ylabel(r'$v_y$ (m/s)')
    ax[2, 0].grid()
    
    ax[3, 0].plot(curvilinear_coordinate,states[OMEGA_INDEX, :],color = 'r', marker='o' )
    ax[3, 0].set_title('Heading Rate')
    ax[3, 0].set_xlabel('s (m)')
    ax[3, 0].set_ylabel(r'$\omega$ (rad/s)')
    ax[3, 0].grid()


    ax[0, 1].plot(curvilinear_coordinate,states[STEERING_INDEX, :],color = 'r', marker='o' )
    ax[0, 1].set_title('Steering')
    ax[0, 1].set_xlabel('s (m)')
    ax[0, 1].set_ylabel(r'$\delta$ (rad)')
    ax[0, 1].grid()
    
    ax[1, 1].plot(curvilinear_coordinate,states[N_INDEX, :], color = 'r', marker='o')
    ax[1, 1].set_title('Lateral Curvilinear Coordinate')
    ax[1, 1].set_xlabel('s (m)')
    ax[1, 1].set_ylabel('n (m)')
    ax[1, 1].grid()
    
    ax[2, 1].plot(curvilinear_coordinate,states[XI_INDEX, :], color = 'r', marker='o')
    ax[2, 1].set_title('Lateral Angular Displacement')
    ax[2, 1].set_xlabel('s (m)')
    ax[2, 1].set_ylabel(r'$\xi$ (rad)')
    ax[2, 1].grid()
    
    vx     = states[VX_INDEX, :]
    vy     = states[VY_INDEX, :]
    n      = states[N_INDEX, :]
    xi     = states[XI_INDEX, :]
    kappa  = curvature_centerline  # curvature of the path at the current s value
    s_dot  = (vx * np.cos(xi) - vy * np.sin(xi)) / (1 - kappa * n)

    
    ax[3, 1].plot(curvilinear_coordinate, s_dot, color = 'r', marker='o')
    ax[3, 1].set_title('Initial 1/s_dot Guess')
    ax[3, 1].set_xlabel('s (m)')
    ax[3, 1].set_ylabel('s_dot (rad/s)')
    ax[3, 1].grid()

    
    plt.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.8)
    plt.tight_layout()




def plot_inputs(curvilinear_coordinate : np.ndarray, inputs: np.ndarray, title: str = "Optimal Racing Line Inputs"):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(curvilinear_coordinate, inputs[THROTTLE_INDEX, :], color = 'r', marker='o')
    ax[0].set_title('Throttle')
    ax[0].set_xlabel('s (m)')
    ax[0].set_ylabel('Throttle')
    ax[0].grid()

    ax[1].plot(curvilinear_coordinate, inputs[STEERING_RATE_INDEX, :],color = 'r', marker='o' )
    ax[1].set_title('Steering Rate')
    ax[1].set_xlabel('s (m)')
    ax[1].set_ylabel(r'$\dot{\delta}$ (rad/s)')
    ax[1].grid()

    plt.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.8)

    plt.tight_layout()
    

def plot_raceline( x_ref_fun : interp1d, racetrack : RaceTrack, name : str) :
    """
    Plots the optimal raceline on the racetrack.
    :param racetrack: The RaceTrack object containing the track geometry.
    :param x_ref_fun: The interpolation function for the reference states.
    """

    s_values = np.linspace(0, racetrack.length, int(racetrack.length))

    if racetrack.ax is None :
        racetrack.draw()
    
    x_coord = []
    y_coord = []

    for s in s_values:
        x = x_ref_fun(s)
        # Extract the position from the state vector
        n = x[N_INDEX]

        pos    = racetrack.position(s)
        normal = racetrack.normal_vector(s)

        vehicle_position = pos + normal * n  # Adjust the position based on the lateral curvilinear coordinate
        
        x_coord += [vehicle_position[0]]
        y_coord += [vehicle_position[1]]    

    # Draw the reference trajectory on the racetrack
    racetrack.ax.plot(x_coord, y_coord, linestyle='--', marker='o', markersize=2, label = name)
    racetrack.ax.legend()





def get_reference(folder_path:str, racetrack_name : str) -> tuple[interp1d, interp1d]: 
    """
    Loads the reference trajectory for a given racetrack from .npy files and creates interpolation functions.
    :param folder_path      : The folder path where the reference files are stored.
    :param racetrack_name   : The name of the racetrack. The file names are expected to be in the format '{racetrack_name}_x_ref.npy' and '{racetrack_name}_u_ref.npy'.
    :return                 : A tuple containing the interpolation functions for states and inputs.
    """

    # loading reference racing_lines
    # Setting the reference trajectory for each competitor
    x_ref_file = folder_path + "/" + racetrack_name + "_x_ref" + ".npy"
    u_ref_file = folder_path + "/" + racetrack_name + "_u_ref" + ".npy"

    # load the reference a numpy array
    try :
        x_ref = np.load(x_ref_file).T
        u_ref = np.load(u_ref_file).T
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Reference files not found for racetrack '{racetrack_name}' in folder '{folder_path}'. Please ensure the files '{racetrack_name}_x_ref.npy' and '{racetrack_name}_u_ref.npy' exist in the specified folder.") from e

    # load the reference a numpy array
    s_ref   = x_ref[-1,:]   # curvilinear coordinate of the reference trajectory
    x_ref   = x_ref[:-1,:]
    u_ref   = u_ref[:-1,:]

    # create interpolated trajectory functions
    u_ref_fun = interp1d(s_ref, u_ref,kind='previous', axis=1, bounds_error=True)
    x_ref_fun = interp1d(s_ref, x_ref, axis=1, bounds_error=True)

    return x_ref_fun, u_ref_fun






