import numpy as np
import cv2 as cv
from itertools import count


cap = cv.VideoCapture(0)
gray = cv.cvtColor(cap.read()[1], cv.COLOR_BGR2GRAY)

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
    points, status, err = cv.calcOpticalFlowPyrLK(
        old_gray,
        gray,
        grid,
        None,
        winSize=(10, 10),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1.5),
    )

    # cv.imshow("frame", gray)
    # if cv.waitKey(1) == ord("q"):
    #     break


cap.release()
cv.destroyAllWindows()
