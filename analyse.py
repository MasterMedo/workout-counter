"""A file.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from body_part_detection import BodyPart


def get_derivation(X):
    return abs(np.array(X)[:-1] - np.array(X)[1:])


def smoothen(X, window_length=51, polyorder=3):
    return savgol_filter(X, window_length, polyorder)


with open("./workout_data/single-arm-dumbbell-triceps-extension.py") as f:
    F, X, Y, C = eval(f.read())


body_part = BodyPart.RIGHT_WRIST
plt.plot(F, smoothen(X[body_part]), label="wrist movement")
plt.plot(F, smoothen(Y[body_part]), label="y coordinate")
plt.title("tricep extensions using RIGHT_WRIST as a point")
plt.xlabel("frame number")
plt.ylabel("coordinate position")
plt.legend()
plt.show()
