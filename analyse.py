"""A file.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import starmap
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy.ndimage.filters import uniform_filter1d

from body_part_detection import BodyPart
from pixel_meter_ratio import pixel_meter_ratio as pmr
from constants import DEFAULT_FPS as DEFAULT_FPS


def get_derivation(X):
    return abs(np.array(X)[:-1] - np.array(X)[1:])


def smoothen(X, fps=DEFAULT_FPS, polyorder=3):
    window_length = (fps // 6) * 2 + 1
    if window_length >= len(X):
        return np.array([])

    return savgol_filter(X, window_length, polyorder)


def smoothen2(X, fps=DEFAULT_FPS):
    size = fps // 3
    return uniform_filter1d(X, size=size)


# if __name__ == "__main__":
for workout in os.listdir("./workout_data/"):
    # workout = "single-arm-dumbbell-triceps-extension.py"
    # workout = "dumbbell-bicep-curl.py"
    workout = workout[:-3]
    print(workout)

    with open(f"./workout_data/{workout}.py") as f:
        data = [eval(row) for row in f if row.strip()]

    ratio = pmr(data[len(data) // 2])

    # 1 / x / ratio -> 1 / x of a meter
    minimal_prominence = 1 / 4 / ratio

    # transform matrix 540x17x3 to 3x17x540
    X, Y, C = map(list, starmap(zip, zip(*starmap(zip, data))))
    F = list(range(len(X[0])))

    for body_part in map(BodyPart, range(17)):
        # body_part = BodyPart.LEFT_WRIST
        x = X[body_part]
        y = Y[body_part]
        x_smooth = smoothen(x)
        y_smooth = smoothen(y)
        peaksX = find_peaks(x)[0]
        peaksY = find_peaks(y)[0]
        prominencesX = peak_prominences(x, peaksX)[0]
        prominencesY = peak_prominences(y, peaksY)[0]
        peaksX = [
            peak
            for peak, prominence in zip(peaksX, prominencesX)
            if prominence > minimal_prominence
        ]
        peaksY = [
            peak
            for peak, prominence in zip(peaksY, prominencesY)
            if prominence > minimal_prominence
        ]
        s = ""
        if len(peaksX) > 3:
            s += f"\n    X: {len(peaksX)}"
        if len(peaksY) > 3:
            s += f"\n    Y: {len(peaksY)}"
        if s:
            print(f"  {body_part.name}{s}")

        # plt.plot(F, y, label="raw")
        # plt.plot(F, smoothen(y), label="savitsky-golayski")
        # plt.plot(F, smoothen2(y), label="moving average")
        # plt.title(f"{workout} using {body_part.name.lower()} as a point")
        # plt.xlabel("frame number")
        # plt.ylabel("coordinate position")
        # plt.legend()
        # plt.show()
        # exit()
