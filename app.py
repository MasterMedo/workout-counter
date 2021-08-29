import cv2 as cv

from body_part_detection import detect_body_parts


if __name__ == "__main__":
    # read the video from the camera
    cap = cv.VideoCapture(0)

    # press 'q' to exit the video
    while cv.waitKey(1) != ord("q"):
        is_valid, frame = cap.read()
        if not is_valid:
            print("what")
            break

        for x, y, confidence in detect_body_parts(frame, 256):
            if confidence > 0.2:
                # draw a point on each body part
                cv.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # show the frame in a window
        cv.imshow("Pose detection", frame)

    # release the memory
    cap.release()
    cv.destroyAllWindows()
