"""A file.
"""
import os
import numpy as np

# import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_prominences

from body_part_detection import BodyPart


def get_derivation(X):
    return abs(np.array(X)[:-1] - np.array(X)[1:])


def smoothen(X, window_length=51, polyorder=3):
    if window_length >= len(X):
        return np.array()
    return savgol_filter(X, window_length, polyorder)


for workout in os.listdir("./workout_data/"):
    with open(f"./workout_data/{workout}") as f:
        F, Y, X, C = eval(f.read())

    print(workout)
    for body_part in map(BodyPart, range(17)):
        x = smoothen(X[body_part])
        y = smoothen(Y[body_part])
        peaksX = find_peaks(x)[0]
        peaksY = find_peaks(y)[0]
        prominencesX = peak_prominences(x, peaksX)[0]
        prominencesY = peak_prominences(y, peaksY)[0]
        peaksX = [
            peak for peak, prominence in zip(peaksX, prominencesX) if prominence > 1e-1
        ]
        peaksY = [
            peak for peak, prominence in zip(peaksY, prominencesY) if prominence > 1e-1
        ]
        s = ""
        if len(peaksX) > 3:
            s += f"\n    X: {len(peaksX)}"
        if len(peaksY) > 3:
            s += f"\n    Y: {len(peaksY)}"
        if s:
            print(f"  {body_part.name}{s}")
        # plt.plot(F, smoothen(X), label="wrist movement")
        # plt.plot(F, smoothen(Y), label="y coordinate")
        # plt.title("tricep extensions using RIGHT_WRIST as a point")
        # plt.xlabel("frame number")
        # plt.ylabel("coordinate position")
        # plt.legend()
        # plt.show()
