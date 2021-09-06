import os  # noqa
import cv2 as cv
import numpy as np

from typing import TextIO
from itertools import count
from pixel_meter_ratio import pixel_meter_ratio as pmr
from body_part_detection import detect_body_parts

from body_part_detection import BodyPart
from constants import MINIMAL_CONFIDENCE, MOVENET_PIXEL_SIZE


alpha = 0.6


def main(input_path: str, output_file: TextIO) -> None:
    # read the video from the camera
    cap = cv.VideoCapture(0)

    # press 'q' to exit the video
    _, frame = cap.read()
    body_parts_prev = detect_body_parts(frame, MOVENET_PIXEL_SIZE)
    peaks = [[[0, 0] for _ in range(2)] for _ in range(17)]
    counts = [[0, 0] for _ in range(17)]

    for i in count():
        is_valid, frame = cap.read()
        if not is_valid or cv.waitKey(1) == ord("q"):
            break

        body_parts = detect_body_parts(frame, MOVENET_PIXEL_SIZE)
        # save(output_file, body_parts)
        # draw_screen(frame, body_parts)

        ratio = pmr(body_parts)
        if ratio <= 0:
            continue
        minimal_prominence = 1 / 4 / ratio
        maximal_movement = 1 / 8 / ratio

        for body_part in map(BodyPart, range(17)):
            x, y, c = body_parts[body_part]
            x_, y_, c_ = body_parts_prev[body_part]

            # if confidence dropped by more than 20% from last frame
            # and the body part moved more than 12cm, don't move.
            # this should be dynamic (the less confident the less movement)
            if c + 0.2 < c_:
                if abs(x - x_) > maximal_movement:
                    x = x_

                if abs(y - y_) > maximal_movement:
                    y = y_

            # smoothing
            body_parts_prev[body_part] = (
                x_ * alpha + (1 - alpha) * x,
                y_ * alpha + (1 - alpha) * y,
                c_ * alpha + (1 - alpha) * c,
            )

            if c < MINIMAL_CONFIDENCE:
                continue

            x, y, c = body_parts_prev[body_part]
            xmax, xmin = peaks[body_part][0]
            ymax, ymin = peaks[body_part][1]
            peaks[body_part] = [
                [max(x, xmax), min(x, xmin)],
                [max(y, ymax), min(y, ymin)],
            ]

            if x > xmin + minimal_prominence:
                peaks[body_part][0][1] = x
                counts[body_part][0] += 1
                print(body_part.name, counts[body_part])
            if y > ymin + minimal_prominence:
                peaks[body_part][1][1] = y
                counts[body_part][1] += 1
                print(body_part.name, counts[body_part])
            if x < xmax - minimal_prominence:
                peaks[body_part][0][0] = x
                counts[body_part][0] += 1
                print(body_part.name, counts[body_part])
            if y < ymax - minimal_prominence:
                peaks[body_part][1][0] = y
                counts[body_part][1] += 1
                print(body_part.name, counts[body_part])

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
