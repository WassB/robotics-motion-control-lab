import numpy as np
from   racing.racetrack.racetrack import RaceTrack
import matplotlib.pyplot as plt
import os
import yaml
# Define control points
name = "oval"
track_width = 6.0
P =  np.array([
    [-100, 100], 
    [-80, 120], 
    [-20, 100], 
    [50, 80], 
    [50, 10], 
    [30, -20],
    [0, -30],
]) 
# spline = CubicBspline(P[0], P[1], P[2], P[3])
# # Draw the B-spline
# spline.draw(mode = "velocity")
# plt.show()

filedir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"arrays")

np.save(filedir+"/" + name, P)

race_track = RaceTrack(P, track_width=track_width)
race_track.draw()

data = {
    'track_width': track_width,
    'points': P.tolist()
}

with open(filedir+"/" + name + ".yaml", "w") as f:
    yaml.dump(data, f)


plt.show()
