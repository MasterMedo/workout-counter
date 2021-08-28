import cv2 as cv


cap = cv.VideoCapture(0)

while True:
    is_valid, frame = cap.read()
    if not is_valid:
        break

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("frame", gray)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
