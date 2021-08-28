import numpy as np
import cv2 as cv
from itertools import count

from repetition_detector import f2


cap = cv.VideoCapture(0)
gray = cv.cvtColor(cap.read()[1], cv.COLOR_BGR2GRAY)
old_points = None
points = None
old_optical_flow = None
optical_flow = None

h, w = cap.get(3), cap.get(4)
grid = np.array(
    [[[i * h / 10, j * w / 10]] for i in range(1, 10) for j in range(1, 10)],
    dtype=np.float32,
)

for i in count():
    is_valid, frame = cap.read()
    if not is_valid:
        break

    if i % 5 != 0:
        continue

    old_gray = gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    old_points = points

    points, status, err = cv.calcOpticalFlowPyrLK(
        old_gray,
        gray,
        grid,
        None,
        winSize=(10, 10),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1.5),
    )

    optical_flow_old = optical_flow
    if old_points is not None:
        optical_flow = old_points - points

    if optical_flow_old is not None:
        print(f2(optical_flow_old, optical_flow))

    # cv.imshow("frame", gray)
    # if cv.waitKey(1) == ord("q"):
    #     break


cap.release()
cv.destroyAllWindows()
