import os
import cv2 as cv

# ignore warnings and information messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa E402


# read the video from the camera
cap = cv.VideoCapture(0)

# load the MoveNet model
model = tf.saved_model.load("./")
movenet = model.signatures["serving_default"]

# size of the square frame sent to the MoveNet
size = 256


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

# press 'q' to exit the video
while cv.waitKey(1) != ord("q"):
    is_valid, frame = cap.read()
    if not is_valid:
        break

    height, width, *_ = frame.shape
    box_size = max(height, width)

    # format the video frame to a square
    image = tf.convert_to_tensor(frame)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, size, size), dtype=tf.int32)

    # get the body parts positions and score
    output = movenet(image)
    keypoints = output["output_0"].numpy()

    for y, x, confidence in keypoints[0][0]:
        if confidence > 0.2:
            # calculate the positions of body parts on the frame
            x = int(x * box_size - (box_size - width) / 2)
            y = int(y * box_size - (box_size - height) / 2)

            # draw a point on each body part
            cv.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # show the frame in a window
    cv.imshow("Pose detection", frame)

# release the memory
cap.release()
cv.destroyAllWindows()
