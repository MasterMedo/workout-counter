import os
import numpy as np

# ignore warnings and information messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa E402


body_parts = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


# load the MoveNet model
model = tf.saved_model.load("./movenet_model")
movenet = model.signatures["serving_default"]


def detect_body_parts(frame: np.ndarray, size: int = 256) -> np.ndarray:
    """Takes the image in form of a numpy array, detects the positions
    and confidence of body parts and returns them in a numpy array.

    :param frame np.ndarray: Frame of the image.
    :param size int: Size of the square frame MoveNet detects body parts.
    :return: Returns 17 points containing coordinates [x, y, confidence]
    :rtype: np.ndarray[np.ndarray[np.float32]]]]
    """
    global movenet

    # format the video frame to a square
    image = tf.convert_to_tensor(frame)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, size, size), dtype=tf.int32)

    # get the body parts positions and score
    return movenet(image)["output_0"].numpy()[0][0]
