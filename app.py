import os  # noqa
import cv2 as cv
import numpy as np

from typing import TextIO
from itertools import count
from pixel_meter_ratio import pixel_meter_ratio as pmr
from body_part_detection import detect_body_parts

from constants import MINIMAL_CONFIDENCE, MOVENET_PIXEL_SIZE


def main(input_path: str, output_file: TextIO) -> None:
    # read the video from the camera
    cap = cv.VideoCapture(input_path)

    # press 'q' to exit the video
    for i in count():
        is_valid, frame = cap.read()
        if not is_valid or cv.waitKey(1) == ord("q"):
            break

        body_parts = detect_body_parts(frame, MOVENET_PIXEL_SIZE)
        save(output_file, body_parts)
        # draw_screen(frame, body_parts)

    # release the memory
    cap.release()
    cv.destroyAllWindows()


def draw_screen(frame: np.array, body_parts):
    ratio = pmr(body_parts)
    cv.line(
        img=frame,
        pt1=(1, 0),
        pt2=(1, 1 // ratio),
        color=(0, 0, 255),
        thickness=3,
    )
    for x, y, confidence in body_parts:
        if confidence > MINIMAL_CONFIDENCE:
            # draw a point on each body part
            cv.circle(
                img=frame,
                center=(x, y),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )

    # show the frame in a window
    cv.imshow("Pose detection", frame)


def save(f: TextIO, body_parts: np.array) -> None:
    print(body_parts, file=f)


if __name__ == "__main__":
    # for workout in os.listdir("./workout_videos/"):
    workout = "pullups"
    with open(f"./workout_data/{workout}.py", "w") as output:
        main(f"./workout_videos/{workout}.MOV", output)
