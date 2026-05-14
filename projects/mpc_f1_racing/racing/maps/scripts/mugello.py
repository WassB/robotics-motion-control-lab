import numpy as np
from   racing.racetrack.racetrack import RaceTrack
import matplotlib.pyplot as plt
import os
import yaml

# Define control points
name = "mugello"
track_width = 5.

points =  np.array([
    [99.222615, 40.340909], [84.946996, 40.909091], [71.519435, 42.329545],
    [58.515901, 44.034091], [28.975265, 43.892045], [22.614841, 47.869318],
    [24.452297, 54.261364], [35.194346, 57.244318], [42.968198, 60.511364],
    [44.381625, 69.318182], [49.611307, 73.721591], [66.007067, 71.306818],
    [88.621908, 67.187500], [96.254417, 73.295455], [101.908127, 74.289773],
    [123.533569, 68.181818],
    [149.681979, 12.357955], [138.515901, 10.795455], [134.134276, 13.778409],
    [130.459364, 20.454545], [125.512367, 22.443182], [94.763251, 18.892045],
    [84.522968, 22.017045], [84.522968, 28.977273], [91.731449, 32.102273],
    [102.897527, 29.119318], [110.812721, 29.971591], [116.325088, 32.386364],
    [130.459364, 28.693182], [159.717314, 21.875000],
    [168.763251, 29.829545], [162.968198, 36.789773], [146.713781, 39.204545],
    [133.286219, 39.346591]
]) *np.array([2., 2.5])


filedir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"arrays")


race_track = RaceTrack(points, track_width=track_width)
race_track.draw()


data = {
    'track_width': track_width,
    'points': points.tolist()
}

with open(filedir+"/" + name + ".yaml", "w") as f:
    yaml.dump(data, f)


plt.show()