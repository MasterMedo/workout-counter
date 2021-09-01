import csv
import math
from functools import cache

minimal_certainty = 0.5

@cache
def get_segment_dictionary():
    dictionary = {}
    with open('segments.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dictionary[row['id']] = float(row[' average length'])
    return dictionary


@cache
def get_enumerated_body_parts_dictionary():
    dictionary2 = {}
    with open('body_parts_enumareted.csv', 'r') as f2:
        reader2 = csv.DictReader(f2)
        for row2 in reader2:
            dictionary2[row2['PART']] = int(row2['NUMBER'])
    return dictionary2


def get_segment_height_ratio_by_name(segment_name):
    return get_segment_dictionary()[segment_name]


def pixel_meter_ratio(body_part_positions, height):
    global minimal_certainty

    def certain(part_name):
        part_to_number_dictionary = get_enumerated_body_parts_dictionary()
        if certainties[part_to_number_dictionary[part_name]] >= minimal_certainty:
            return True
        return False

    def euclidean_distance(a, b):
        return math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

    # Returns -1 if invalid certainty
    def get_distance_between_parts(part_a_name, part_b_name):
        part_to_number_dictionary = get_enumerated_body_parts_dictionary()
        if certain(part_a_name) and certain(part_b_name):
            return euclidean_distance(body_part_positions_coordinates_only[part_to_number_dictionary[part_a_name]], body_part_positions_coordinates_only[part_to_number_dictionary[part_b_name]])
        else:
            return -1

    def find_midpoint(part_a_name, part_b_name):
        def get_location_by_name(part_name):
            part_to_number_dictionary = get_enumerated_body_parts_dictionary()
            return body_part_positions_coordinates_only[part_to_number_dictionary[part_name]]
        part_a = get_location_by_name(part_a_name)
        part_b = get_location_by_name(part_b_name)
        return [int((part_a[0] + part_b[0])/2), int((part_a[1] + part_b[1])/2)]

    pmr = None       # Pixel to Meter ratio e.g. pixel length in meters
    n_parts = 17     # Number of distinct body parts

    # Slicing
    body_part_positions_coordinates_only = []
    certainties = []
    for body_part in body_part_positions:
        body_part_positions_coordinates_only.append([body_part[0], body_part[1]])
        certainties.append(body_part[2])

    # Length calculations
    upper_arm_left = get_distance_between_parts('LEFT_SHOULDER', 'LEFT_ELBOW')
    upper_arm_right = get_distance_between_parts('RIGHT_SHOULDER', 'RIGHT_ELBOW')
    forearm_left = get_distance_between_parts('LEFT_ELBOW', 'LEFT_WRIST')
    forearm_right = get_distance_between_parts('RIGHT_ELBOW', 'RIGHT_WRIST')
    thigh_left = get_distance_between_parts('LEFT_HIP', 'LEFT_KNEE')
    thigh_right = get_distance_between_parts('RIGHT_HIP', 'RIGHT_KNEE')
    leg_left = get_distance_between_parts('LEFT_KNEE', 'LEFT_ANKLE')
    leg_right = get_distance_between_parts('RIGHT_KNEE', 'RIGHT_ANKLE')
    biacromial = get_distance_between_parts('LEFT_SHOULDER', 'RIGHT_SHOULDER')
    biiliac = get_distance_between_parts('LEFT_HIP', 'RIGHT_HIP')
    ear_to_ear = get_distance_between_parts('LEFT_EAR', 'RIGHT_EAR')
    shoulder_midpoint = find_midpoint('LEFT_SHOULDER', 'RIGHT_SHOULDER')
    hip_midpoint = find_midpoint('LEFT_HIP', 'RIGHT_HIP')
    thorax_and_abdomen = euclidean_distance(shoulder_midpoint, hip_midpoint)

    # Part to segment mapping
    upper_arm = max(upper_arm_left, upper_arm_right)
    foream = max(forearm_left, forearm_right)
    thigh = max(thigh_left, thigh_right)
    leg = max(leg_left, leg_right)

    thorax_to_abdomen_ratio = get_segment_height_ratio_by_name('thorax') / get_segment_height_ratio_by_name('abdomen')
    abdomen = thorax_and_abdomen / (thorax_to_abdomen_ratio + 1)
    thorax = thorax_and_abdomen - abdomen

    # Caclculating height predictions
    pixel_heights = [
        upper_arm / get_segment_height_ratio_by_name('upper-arm'),
        foream / get_segment_height_ratio_by_name('forearm'),
        thigh / get_segment_height_ratio_by_name('thigh'),
        leg / get_segment_height_ratio_by_name('leg'),
        abdomen / get_segment_height_ratio_by_name('abdomen'),
        thorax / get_segment_height_ratio_by_name('thorax')]
    return height / max(pixel_heights)
