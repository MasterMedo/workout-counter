"""A file.
"""
import numpy as np
import matplotlib.pyplot as plt

from body_part_detection import BodyPart

with open("./workout_data/single-arm-dumbbell-triceps-extension.py") as f:
    F, X, Y, C = eval(f.read())


plt.title("tricep extensions using RIGHT_WRIST as a point")
plt.plot(F, np.array(X[BodyPart.RIGHT_WRIST]), label="x coordinate")
plt.plot(F, np.array(Y[BodyPart.RIGHT_WRIST]), label="y coordinate")
plt.xlabel("frame number")
plt.ylabel("coordinate position")
plt.legend()
plt.show()
