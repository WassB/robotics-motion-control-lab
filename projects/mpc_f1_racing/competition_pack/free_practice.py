import matplotlib.pyplot as plt
import os
import numpy as np
from   scipy.interpolate import interp1d
import yaml

from   racing.racetrack import RaceTrack, FreePractice, get_reference, plot_raceline
from   racing.mpc.mpc   import NMPC


#################################################################
##############  Free Practice Mode for Racing Simulation ########
#################################################################


racetrack_name = "montecarlo"  # pick one among : "catalunya","montecarlo","oval"
team_dir       = os.path.dirname(__file__) 

# Setting the map and the vehicles models

with open(team_dir + f"/{racetrack_name}_par_mpc.yaml") as stream:
    parameters = yaml.safe_load(stream) 

racetrack    : RaceTrack = RaceTrack.get_racetrack(racetrack_name)
mpc          : NMPC      = NMPC(  car_model  = parameters["car_model"],
                                  racetrack  = racetrack,
                                  Q          = np.array(parameters["Q"]),
                                  R          = np.array(parameters["R"]),
                                  N          = parameters["N"],
                                  ds         = parameters["ds"],)

# extract reference for this racetrack
x_ref_fun, u_ref_fun  = get_reference(team_dir, racetrack_name, car_model=parameters["car_model"])

# plot reference raceline if you want
plot_raceline(x_ref_fun = x_ref_fun, 
               racetrack = racetrack, 
               name      = "reference")


# Set reference on the MPC side
mpc.set_reference_trajectory(x_ref = x_ref_fun, u_ref = u_ref_fun)

# start free trials
free_practice = FreePractice(racetrack = racetrack,
                              mpc       = mpc)

# run this for fast lap (no animation)
# result = free_practice.fast_lap(s_0= 0.)

# run this for animated lap (slower)
result = free_practice.animated_lap(s_0=34.)

plt.show()
