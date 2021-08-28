import cv2
import numpy as np
import math

current_count = 0
div_average = None
average_polar = []
ALPHA = 0.99
MAX_BUFFER_SIZE = 100

def get_mean_polar_coordinates(A):
    X = np.average(A[..., 0])
    Y = np.average(A[..., 1])

    return (math.sqrt(Y**2 + X**2), math.atan2(Y, X))

def f2(lastOF, curOF):
    global div_average, \
           current_count, \
           average_polar, \
           ALPHA, \
           MAX_BUFFER_SIZE

    div = curOF - lastOF
    print(curOF, lastOF)
    print(div)

    if (div_average is None):
        div_average = div
    else:
        div_average += ALPHA * div_average + (1 - ALPHA) * div

    mag, phi = get_mean_polar_coordinates(div_average)

    average_polar.append((mag, phi))
    if (len(average_polar) > MAX_BUFFER_SIZE):
        average_polar = average_polar[1:]

    print(average_polar[-5:])
    detection = False

    # Mislav's code here:
    if (detection):
        current_count += 1

    return (current_count // 2)
