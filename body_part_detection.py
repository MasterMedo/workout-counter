"""Used to find human poses in images.
"""
import os
import numpy as np
from enum import IntEnum

# ignore warnings and information messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa E402


# load the MoveNet model
model = tf.saved_model.load("./movenet_model")
movenet = model.signatures["serving_default"]


class BodyPart(IntEnum):
    """Access body parts returned by the detect_body_parts function.

    Example usage:
        body_parts = detect_body_parts(frame, 256)
        print(body_parts[BodyPart.RIGHT_SHOULDER])
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


def detect_body_parts(frame: np.ndarray, size: int = 256) -> np.ndarray:
    """Takes the image in form of a numpy array, detects the positions
    and confidence of body parts and returns them in a numpy array.

    :param frame np.ndarray: Frame of the image.
    :param size int: Size of the square frame MoveNet detects body parts.
    :return: Returns 17 points containing coordinates [x, y, confidence]
        x - integer pixel coordinates on the original frame
        y - integer pixel coordinates on the original frame
        confidence - float from 0 to 1 indicating the probability of detection
    :rtype: list[tuple[int, int, float]]
    """
    global movenet

    # format the video frame to a square tensor required by MoveNet
    image = tf.convert_to_tensor(frame)
    image = tf.image.resize_with_pad(image, size, size)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.int32)

    # get the body parts positions and score
    keypoints = movenet(image)["output_0"]
    keypoints = keypoints.numpy().squeeze()

    # return keypoints
    # calculate the positions of body parts on the original frame
    height, width, *_ = frame.shape
    box_size = max(height, width)
    return [
        (
            int(x * box_size - (box_size - width) / 2),
            int(y * box_size - (box_size - height) / 2),
            confidence,
        )
        for y, x, confidence in keypoints
    ]
