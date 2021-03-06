import csv

from functools import cache
from numpy import mean as midpoint
from scipy.spatial.distance import euclidean

from body_part_detection import BodyPart
from constants import MINIMAL_CONFIDENCE


@cache
def get_segment_lengths():
    with open("segment_lengths.csv", "r") as f:
        reader = csv.DictReader(f)
        return {row["id"]: float(row["average length"]) for row in reader}


def get_segment_length(segment_name):
    return get_segment_lengths()[segment_name]


def pixel_meter_ratio(body_part_positions, person_height=1.75):
    def body_part_midpoint(a, b):
        a = body_part_positions[a]
        b = body_part_positions[b]
        return list(midpoint(([a[:-1], b[:-1]]), axis=0)) + [min(a[-1], b[-1])]

    def body_part_distance(a, b):
        a = body_part_positions[a]
        b = body_part_positions[b]
        if min(a[-1], b[-1]) < MINIMAL_CONFIDENCE:
            return 0

        return euclidean(
            a[:-1],
            b[:-1],
        )

    def abdomen_thorax_length(shoulder_midpoint, hip_midpoint):
        if min(shoulder_midpoint[-1], hip_midpoint[-1]) < MINIMAL_CONFIDENCE:
            return 0, 0, 0

        length = euclidean(
            shoulder_midpoint,
            hip_midpoint,
        )
        thorax = get_segment_length("thorax")
        abdomen = get_segment_length("abdomen")
        pelvis = get_segment_length("pelvis")
        together = thorax + abdomen + pelvis
        return (
            length * (thorax / together),
            length * (abdomen / together),
            length * (pelvis / together),
        )

    upper_arm_left = body_part_distance(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW)
    upper_arm_right = body_part_distance(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW)
    forearm_left = body_part_distance(BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST)
    forearm_right = body_part_distance(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST)
    thigh_left = body_part_distance(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE)
    thigh_right = body_part_distance(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE)
    leg_left = body_part_distance(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE)
    leg_right = body_part_distance(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
    biacromial = body_part_distance(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    bi_iliac = body_part_distance(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    shoulder_midpoint = body_part_midpoint(
        BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
    )
    hip_midpoint = body_part_midpoint(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    thorax, abdomen, pelvis = abdomen_thorax_length(
        shoulder_midpoint,
        hip_midpoint,
    )

    pixel_heights = [
        upper_arm_left / get_segment_length("upper-arm"),
        upper_arm_right / get_segment_length("upper-arm"),
        forearm_left / get_segment_length("forearm"),
        forearm_right / get_segment_length("forearm"),
        thigh_left / get_segment_length("thigh"),
        thigh_right / get_segment_length("thigh"),
        leg_left / get_segment_length("leg"),
        leg_right / get_segment_length("leg"),
        thorax / get_segment_length("thorax"),
        abdomen / get_segment_length("abdomen"),
        pelvis / get_segment_length("pelvis"),
        biacromial / get_segment_length("biacromial"),
        bi_iliac / get_segment_length("bi-iliac"),
    ]

    m = max(pixel_heights)
    return person_height / max(pixel_heights) if m > 0 else 0
