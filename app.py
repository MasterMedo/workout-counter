import os
import cv2 as cv

from itertools import count
from pixel_meter_ratio import pixel_meter_ratio as pmr
from body_part_detection import detect_body_parts


# for workout in os.listdir("./workout_videos/"):
workout = "dumbbell-bicep-curl.MOV"
if __name__ == "__main__":

    # read the video from the camera
    cap = cv.VideoCapture(f"./workout_videos/{workout}")

    # press 'q' to exit the video
    for i in count():
        is_valid, frame = cap.read()
        if not is_valid or cv.waitKey(1) == ord("q"):
            break

        # F.append(i)
        body_parts = detect_body_parts(frame, 256)
        # for j, (x, y, confidence) in enumerate(body_parts):
        #     X[j].append(x)
        #     Y[j].append(y)
        #     C[j].append(confidence)

        for x, y, confidence in body_parts:
            if confidence > 0.3:
                # draw a point on each body part
                cv.circle(
                    img=frame,
                    center=(x, y),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )

        # show the frame in a window
        # cv.imshow("Pose detection", frame)
        print(pmr(body_parts))

    # release the memory
    cap.release()
    cv.destroyAllWindows()
