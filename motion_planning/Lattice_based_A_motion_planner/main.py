
import numpy as np
import matplotlib.pyplot as plt
from motion_primitives import motion_primitives

start = (5, 5)
goal = (95, 95)


grid = np.zeros((100, 100))
grid[20:40, 20:40] = 1
grid[70:80, 10:55] = 1
grid[20:50, 60:80] = 1

plt.imshow(grid.T, origin="lower", cmap="Greys")


plt.scatter(start[0], start[1], marker="o", s=100, label="Start")
plt.scatter(goal[0], goal[1], marker="*", s=150, label="Goal")

plt.grid(True, alpha=0.2)
plt.legend()
plt.show()
