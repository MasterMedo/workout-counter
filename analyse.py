"""A file.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import starmap
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy.ndimage.filters import uniform_filter1d

from body_part_detection import BodyPart
from pixel_meter_ratio import pixel_meter_ratio as pmr
from constants import DEFAULT_FPS, MINIMAL_CONFIDENCE


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


def draw(workout="", body_part=""):
    plt.title(
        f"{workout} using {body_part} as a point",
        fontsize=18,
    )
    plt.xlabel("frame number")
    plt.ylabel("coordinate position")
    plt.legend()
    plt.show()


def is_relevant_body_part(x, minimal_prominence):
    x = smoothen(x)
    peaksX = find_peaks(x)[0]
    prominencesX = peak_prominences(x, peaksX)[0]
    peaksX = [
        peak
        for peak, prominence in zip(peaksX, prominencesX)
        if prominence > minimal_prominence
    ]
    return len(peaksX) > 3


if __name__ == "__main__":
    # for workout in os.listdir("./workout_data/"):
    # workout = "single-arm-dumbbell-triceps-extension.py"
    workout = "dumbbell-bicep-curl.py"
    workout = workout[:-3]
    print(workout)

    with open(f"./workout_data/{workout}.py") as f:
        data = [eval(row) for row in f if row.strip()]

    alpha = 0.6

    body_parts_history = data[0]
    data_history = []
    peak_history = [[[0, 0] for _ in range(2)] for _ in range(17)]
    count_history = [[0, 0] for _ in range(17)]
    for body_parts in data:
        # 1 / x / ratio -> 1 / x of a meter
        ratio = pmr(body_parts)

        # motion difference between starting and ending point in an exercise
        minimal_prominence = 1 / 4 / ratio

        # maximal movement difference between two frames
        maximal_movement = 1 / 8 / ratio

        for body_part in map(BodyPart, range(17)):
            x, y, c = body_parts[body_part]
            x_, y_, c_ = body_parts_history[body_part]

            # if confidence dropped by more than 20% from last frame
            # and the body part moved more than 12cm, don't move.
            # this should be dynamic (the less confident the less movement)
            if c + 0.2 < c_:
                if abs(x - x_) > maximal_movement:
                    x = x_

                if abs(y - y_) > maximal_movement:
                    y = y_

            # smoothing
            body_parts_history[body_part] = (
                x_ * alpha + (1 - alpha) * x,
                y_ * alpha + (1 - alpha) * y,
                c_ * alpha + (1 - alpha) * c,
            )

            if c < MINIMAL_CONFIDENCE:
                continue

            x, y, c = body_parts_history[body_part]
            xmax, xmin = peak_history[body_part][0]
            ymax, ymin = peak_history[body_part][1]
            peak_history[body_part] = [
                [max(x, xmax), min(x, xmin)],
                [max(y, ymax), min(y, ymin)],
            ]

            if x > xmin + minimal_prominence:
                peak_history[body_part][0][1] = x
                count_history[body_part][0] += 1
            if y > ymin + minimal_prominence:
                peak_history[body_part][1][1] = y
                count_history[body_part][1] += 1
            if x < xmax - minimal_prominence:
                peak_history[body_part][0][0] = x
                count_history[body_part][0] += 1
            if y < ymax - minimal_prominence:
                peak_history[body_part][1][0] = y
                count_history[body_part][1] += 1

        data_history.append(deepcopy(body_parts_history))

    X, Y, C = map(list, starmap(zip, zip(*starmap(zip, data))))
    X_, Y_, C_ = map(list, starmap(zip, zip(*starmap(zip, data_history))))
    F = list(range(len(X[0])))

    y = Y[BodyPart.LEFT_WRIST]
    y_ = Y_[BodyPart.LEFT_WRIST]
    print(count_history)
    plt.plot(F, y, label="raw data")
    plt.plot(F, y_, label="processed data")
    draw()
