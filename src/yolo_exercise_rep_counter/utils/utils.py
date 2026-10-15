import numpy as np


# Calculate an angle between three points given as [x_coord, y_coord_ confidence_score]
def calculate_angle(a, b, c, required_confidence):
    if (a[2] < required_confidence
            or b[2] < required_confidence
            or c[2] < required_confidence):
        return None

    # Get X and Y coordinates of points
    a_point = np.array(a[:2])
    b_point = np.array(b[:2])
    c_point = np.array(c[:2])

    # Create vectors outgoing from "elbow" point
    ba_vector = a_point - b_point
    bc_vector = c_point - b_point

    # Calculate dot product of vectors and their magnitudes
    # Dot product: BA * BC = (BA_x * BC_x) + (BA_y * BC_y)
    dot_product = np.dot(ba_vector, bc_vector)
    magnitude_ba_vector = np.linalg.norm(ba_vector)
    magnitude_bc_vector = np.linalg.norm(bc_vector)

    # Since BA * BC = ||BA|| * ||BC|| * cos(t), where || are magnitudes of vectors and
    # cos(t) is the cosine of the angle between them, then having the values of all
    # variables except t, we can calculate it.

    if magnitude_ba_vector == 0 or magnitude_bc_vector == 0:
        return 0.0

    cosine_angle = np.clip(dot_product / (magnitude_ba_vector * magnitude_bc_vector), -1.0, 1.0)

    # Get the angle in degrees
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# Checks if two lists of exercise states contain the same sequence
def are_same_states(state_list_a, state_list_b) -> bool:
    result = True

    if len(state_list_a) != len(state_list_b):
        result = False

    if result:
        for state_a, state_b in zip(state_list_a, state_list_b):
            result = state_a.name == state_b.name

    return result
