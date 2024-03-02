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
pmr_alpha = 0.99
body_part_to_track = BodyPart.RIGHT_WRIST


def main(input_path: str, output_path: str) -> None:
    global peaks, valleys, looking_for
    camera = input_path == 0
    # read the video from the file/camera
    cap = cv.VideoCapture(input_path, cv.CAP_AVFOUNDATION)

    # press 'q' to exit the video
    success, frame = cap.read()
    if not success:
        if input_path == 0:
            print("There's a problem with the camera, no feed detected.")
        else:
            print("There's a problem with the input file, no video detected.")
        exit()
    body_parts_smooth = detect_body_parts(frame, MOVENET_PIXEL_SIZE)
    ratio = pmr(body_parts_smooth)
    # peak = [body_parts_smooth[i][:2] for i in range(17)]
    # valley = [body_parts_smooth[i][:2] for i in range(17)]
    # workout_count = [[0, 0] for _ in range(17)]
    ratio_history = [ratio]

    peaks = [
        [
            [(0, body_parts_smooth[i][0])],
            [(0, body_parts_smooth[i][1])],
        ]
        for i in range(17)
    ]
    valleys = [
        [
            [(0, body_parts_smooth[i][0])],
            [(0, body_parts_smooth[i][1])],
        ]
        for i in range(17)
    ]
    looking_for = [["valley"] * 2 for i in range(17)]
    F = []
    Y = []
    Y_s = []

    for i in count():
        is_valid, frame = cap.read()
        if not is_valid or cv.waitKey(1) == ord("q"):
            break

        body_parts = detect_body_parts(frame, MOVENET_PIXEL_SIZE)
        # save(output_path, body_parts)
        if camera:
            draw_screen(frame, body_parts)

        ratio = ratio * pmr_alpha + (1 - pmr_alpha) * pmr(body_parts)
        if ratio <= 0:  # no body parts could be detected
            continue
        ratio_history.append(1 / 4 / ratio)

        minimal_prominence = 1 / 4 / ratio  # 1/4 meters
        maximal_movement = 1 / 8 / ratio  # 1/8 meters

        for body_part in map(BodyPart, range(17)):
            body_parts[body_part] = adjust_body_part(
                body_parts[body_part],
                body_parts_smooth[body_part],
                maximal_movement,
            )
            body_parts_smooth[body_part] = smoothen_body_part(
                body_parts[body_part],
                body_parts_smooth[body_part],
            )
            x_, y_, c_ = body_parts[body_part]
            x, y, c = body_parts_smooth[body_part]

            if body_part == body_part_to_track:
                F.append(i)
                Y.append(y_)
                Y_s.append(y)

            # don't count joints with small confidence

            x, y, c = body_parts_smooth[body_part]
            find_peaks_and_valleys(x, minimal_prominence, body_part, i, 0)
            find_peaks_and_valleys(y, minimal_prominence, body_part, i, 1)

    # release the memory
    cap.release()
    cv.destroyAllWindows()

    # plt.plot(F, Y, label="position")
    plt.plot(F, Y_s, label="smooth position")
    a, b = list(zip(*peaks[body_part_to_track][1]))
    plt.scatter(a, b, label="peaks")
    a, b = list(zip(*valleys[body_part_to_track][1]))
    plt.scatter(a, b, label="valleys")
    # plt.plot(list(range(len(ratio_history))), ratio_history, label="ratio")
    plt.xlabel("frame number")
    plt.ylabel("pixel number (position)")
    plt.legend()
    plt.show()


def draw_screen(frame: np.array, body_parts):
    ratio = pmr(body_parts) or 1
    cv.line(
        img=frame,
        pt1=(1, 0),
        pt2=(1, int(1 // ratio)),
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


def adjust_body_part(
    body_part: list[float, float, float],
    body_part_prev: list[float, float, float],
    maximal_movement: float,
) -> np.array:
    """If confidence dropped by more than 20% from last frame
    and the body part moved more than 12cm, don't move.
    this should be dynamic (the less confident the less movement)
    """
    x, y, c = body_part
    x_, y_, c_ = body_part_prev
    if c + 0.2 < c_:
        if abs(x - x_) > maximal_movement:
            x = x_

        if abs(y - y_) > maximal_movement:
            y = y_

    return [x, y, c]


def smoothen_body_part(
    body_part: np.array,
    body_part_smooth: np.array,
) -> np.array:
    x, y, c = body_part
    x_, y_, c_ = body_part_smooth
    return (
        x_ * alpha + (1 - alpha) * x,
        y_ * alpha + (1 - alpha) * y,
        c_ * alpha + (1 - alpha) * c,
    )


def find_peaks_and_valleys(
    value,
    minimal_prominence,
    body_part,
    i,
    xy,
):
    global peaks, valleys, looking_for
    last_peak_index, last_peak = peaks[body_part][xy][-1]
    last_valley_index, last_valley = valleys[body_part][xy][-1]
    if looking_for[body_part][xy] == "peak":
        if value < last_valley:
            valleys[body_part][xy][-1] = (i, value)
        elif value > last_valley + minimal_prominence:
            peaks[body_part][xy].append((i, value))
            looking_for[body_part][xy] = "valley"
    elif looking_for[body_part][xy] == "valley":
        if value > last_peak:
            peaks[body_part][xy][-1] = (i, value)
        elif value < last_peak - minimal_prominence:
            valleys[body_part][xy].append((i, value))
            looking_for[body_part][xy] = "peak"


if __name__ == "__main__":
    # for workout in os.listdir("./workout_videos/"):
    base_directory = os.path.dirname(__file__)
    workout = "3_4-sit-up"
    main(
        # base_directory + f"/workout_videos/{workout}.MOV",
        0,  # 0 means camera input
        base_directory + f"/workout_data/{workout}.py",
    )
