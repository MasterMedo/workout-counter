import os  # noqa
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from itertools import count
from pixel_meter_ratio import pixel_meter_ratio as pmr
from body_part_detection import detect_body_parts

from body_part_detection import BodyPart
from constants import MINIMAL_CONFIDENCE, MOVENET_PIXEL_SIZE


alpha = 0.6


def main(input_path: str, output_path: str) -> None:
    # read the video from the camera
    cap = cv.VideoCapture(input_path)

    # press 'q' to exit the video
    _, frame = cap.read()
    body_parts_smooth = detect_body_parts(frame, MOVENET_PIXEL_SIZE)
    ratio = pmr(body_parts_smooth)
    peak = [[0, 0] for _ in range(17)]
    valley = [[0, 0] for _ in range(17)]
    workout_count = [[0, 0] for _ in range(17)]

    peak_history = []
    valley_history = []
    F = []
    Y = []
    Y_s = []

    for i in count():
        is_valid, frame = cap.read()
        if not is_valid or cv.waitKey(1) == ord("q"):
            break

        body_parts = detect_body_parts(frame, MOVENET_PIXEL_SIZE)
        # save(output_path, body_parts)
        # draw_screen(frame, body_parts)

        ratio = ratio * alpha + (1 - alpha) * pmr(body_parts)
        if ratio <= 0:
            continue

        minimal_prominence = 1 / 4 / ratio  # 1/4 meters
        maximal_movement = 1 / 8 / ratio  # 1/8 meters

        for body_part in map(BodyPart, range(17)):
            x, y, c = body_parts[body_part]
            x_, y_, c_ = body_parts_smooth[body_part]

            # if confidence dropped by more than 20% from last frame
            # and the body part moved more than 12cm, don't move.
            # this should be dynamic (the less confident the less movement)
            if c + 0.2 < c_:
                if abs(x - x_) > maximal_movement:
                    x = x_

                if abs(y - y_) > maximal_movement:
                    y = y_

            body_parts_smooth[body_part] = (
                x_ * alpha + (1 - alpha) * x,
                y_ * alpha + (1 - alpha) * y,
                c_ * alpha + (1 - alpha) * c,
            )

            if body_part == 9:
                F.append(i)
                Y.append(y)
                Y_s.append(body_parts_smooth[body_part][1])

            # don't count joints with small confidence
            if c < MINIMAL_CONFIDENCE:
                continue

            x, y, c = body_parts_smooth[body_part]
            xmax, ymax = peak[body_part]
            xmin, ymin = valley[body_part]
            peak[body_part] = [max(x, xmax), max(y, ymax)]
            valley[body_part] = [min(x, xmin), min(y, ymin)]

            if x > xmin + minimal_prominence:
                valley[body_part][0] = x
                workout_count[body_part][0] += 1
                # print(body_part.name, workout_count[body_part])
            if y > ymin + minimal_prominence:
                valley[body_part][1] = y
                workout_count[body_part][1] += 1
                if body_part == 9:
                    peak_history.append((i, y))
                # print(body_part.name, workout_count[body_part])
            if x < xmax - minimal_prominence:
                peak[body_part][0] = x
                workout_count[body_part][0] += 1
                # print(body_part.name, workout_count[body_part])
            if y < ymax - minimal_prominence:
                peak[body_part][1] = y
                workout_count[body_part][1] += 1
                if body_part == 9:
                    valley_history.append((i, y))
                # print(body_part.name, workout_count[body_part])

    # release the memory
    cap.release()
    cv.destroyAllWindows()

    plt.plot(F, Y, label="position")
    plt.plot(F, Y_s, label="smooth position")
    if peak_history:
        a, b = list(zip(*peak_history))
        plt.scatter(a, b, label="peaks")
    if valley_history:
        a, b = list(zip(*valley_history))
        plt.scatter(a, b, label="valleys")
    plt.xlabel("frame number")
    plt.ylabel("pixel number (position)")
    plt.legend()
    plt.show()


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


def save(file_path: str, body_parts: np.array) -> None:
    with open(file_path, "a") as f:
        print(body_parts, file=f)


if __name__ == "__main__":
    # for workout in os.listdir("./workout_videos/"):
    workout = "dumbbell-bicep-curl"
    main(f"../workout_videos/{workout}.MOV", f"./workout_data/{workout}.py")
