from racing.racetrack import RaceTrack
from racing.racetrack import compute_optimal_raceline_dynamic, plot_raceline, save_optimal_raceline
from racing.models import KinematicBicycle
import matplotlib.pyplot as plt
import os


racetrack_names = ["montecarlo", "aragon", "catalunya", "oval"]
car_models = ["lamborghini"]
current_dir = os.path.dirname(os.path.abspath(__file__))
res = 0.25  # m

# create a folder in the current directory for each name
for name in racetrack_names:
    for car_model in car_models:
        dir_path = os.path.join(current_dir, name, car_model)
        os.makedirs(dir_path, exist_ok=True)

        print("Processing racetrack:", name, "with car model:", car_model)
        print("Saving results to:", dir_path)

        # Get racetrack
        racetrack: RaceTrack = RaceTrack.get_racetrack(name)
        model: KinematicBicycle = KinematicBicycle(
            car_model=car_model, track_width=racetrack.track_width)

        # Compute optimal raceline
        x_ref_fun, u_ref_fun, k_ref_fun, s_values, x_initial_guess_fun, u_initial_guess_fun = compute_optimal_raceline_dynamic(
            model, racetrack, resolution=res)
        plot_raceline(x_ref_fun, racetrack=racetrack, name=name)
        plot_raceline(x_initial_guess_fun, racetrack=racetrack,
                      name="convex optimization guess")

        save_optimal_raceline(racetrack_name=name,
                              directory=dir_path,
                              x_ref_fun=x_ref_fun,
                              u_ref_fun=u_ref_fun,
                              s_values=s_values)

plt.show()
