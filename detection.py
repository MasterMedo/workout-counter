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
v = None

n = 33


height, width = cap.get(3), cap.get(4)
grid = np.array(
    [
        [[i * height / n, j * width / n]]
        for i in range(1, n + 1)
        for j in range(1, n + 1)
    ],
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
        winSize=(n, n),
        maxLevel=4,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1.5),
    )

    optical_flow_old = optical_flow
    if old_points is not None:
        optical_flow = old_points - points
        magnitudes, angles = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        for i in range(len(optical_flow)):
            for j in range(len(optical_flow[0])):
                if magnitudes[i][j] > 50:
                    optical_flow[i][j][0] = 0
                    optical_flow[i][j][1] = 0

    for point in grid.astype(int):
        cv.circle(frame, tuple(point[0]), 1, (0, 0, 255), 1)

    if optical_flow is not None:
        for i, (p1, p2) in enumerate(zip(grid.astype(int), optical_flow.astype(int))):
            if status[i]:
                cv.arrowedLine(
                    frame, tuple(p1[0]), tuple(p1[0] + p2[0]), (0, 255, 0), 2
                )

    if optical_flow_old is not None:
        print(
            f2(
                optical_flow_old.reshape((n, n, 2)),
                optical_flow.reshape((n, n, 2)),
            )
        )

    # cv.imshow("frame", gray)
    cv.imshow("Frame", cv.resize(frame, (640, 480)))
    if cv.waitKey(1) == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
