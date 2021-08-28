import cv2
import numpy as np

currentCount = 0
EPSILON = 1.3
firstOF = None
nulVector = None
movStart = False

def get_mag_diff(OF1, OF2):
    x1 = OF2[..., 0] - OF1[..., 0]
    y1 = OF2[..., 1] - OF1[..., 1]

    return cv2.cartToPolar(x1, y1)[0]

def f2(lastOF, curOF):
    global currentCount, \
           EPSILON, \
           firstOF, \
           nulVector, \
           movStart

    if (firstOF is None):
        firstOF = lastOF
        nulVector = EPSILON * get_mag_diff(lastOF, curOF)

    difference = get_mag_diff(lastOF, curOF)

    detection = ((difference <= nulVector).all() \
                 and movStart)

    movStart = (difference > nulVector).any()

    # Mislav's code here:
    if (detection):
        currentCount += 1
        firstOF = None
        movStart = False

    return currentCount
