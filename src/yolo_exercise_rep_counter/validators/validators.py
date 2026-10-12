from abc import ABC, abstractmethod
from .states import ExerciseState, BicepCurlState
import numpy as np

coco_key_points = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"
}


class ExerciseValidator(ABC):
    def __init__(self, state=ExerciseState.INITIAL, required_confidence=0.6):
        self.state = state
        self.required_confidence = required_confidence
        self.rep_count = 0

    @abstractmethod
    def check_constraints(self, key_points):
        pass

    @abstractmethod
    def update_state(self, constraints_passed: bool, details: dict[str: float]):
        pass


class BicepCurlValidator(ExerciseValidator):
    def __init__(self, state=BicepCurlState.UNDETECTED, required_confidence=0.6):
        super().__init__(state, required_confidence)
        self.previous_frame_elbow_angle = None

    def check_constraints(self, key_points):
        if key_points.size == 0:
            return False, {}

        first_person_points = key_points[0]
        constraints_passed = True
        details = {}

        # Constraints
        # None for now

        # Details
        elbow_angle = self._check_elbow_angle(first_person_points[6], first_person_points[8], first_person_points[10])
        details["elbow_angle"] = elbow_angle

        return constraints_passed, details

    def update_state(self, constraints_passed, details):
        if not constraints_passed:
            return

        elbow_angle = details["elbow_angle"]

        if elbow_angle is None:
            self.state = BicepCurlState.UNDETECTED
        elif elbow_angle >= 150:
            self.state = BicepCurlState.INITIAL
        elif elbow_angle < 150 and self.previous_frame_elbow_angle and elbow_angle < self.previous_frame_elbow_angle:
            self.state = BicepCurlState.MOVING_UP
        elif elbow_angle > 40 and self.previous_frame_elbow_angle and elbow_angle > self.previous_frame_elbow_angle:
            self.state = BicepCurlState.MOVING_DOWN

        if elbow_angle and elbow_angle <= 40:
            self.state = BicepCurlState.PEAK

        self.previous_frame_elbow_angle = elbow_angle

        print(self.state)

    def _check_elbow_angle(self, shoulder_point, elbow_point, wrist_point):
        if (shoulder_point[2] < self.required_confidence
                or elbow_point[2] < self.required_confidence
                or wrist_point[2] < self.required_confidence):
            return None

        # Get X and Y coordinates of points
        shoulder = np.array(shoulder_point[:2])
        elbow = np.array(elbow_point[:2])
        wrist = np.array(wrist_point[:2])

        # Create vectors outgoing from "elbow" point
        elbow_shoulder_vector = shoulder - elbow
        elbow_wrist_vector = wrist - elbow

        # Calculate dot product of vectors and their magnitudes
        # Dot product: BA * BC = (BA_x * BC_x) + (BA_y * BC_y)
        dot_product = np.dot(elbow_shoulder_vector, elbow_wrist_vector)
        magnitude_elbow_shoulder_vector = np.linalg.norm(elbow_shoulder_vector)
        magnitude_elbow_wrist_vector = np.linalg.norm(elbow_wrist_vector)

        # Since BA * BC = ||BA|| * ||BC|| * cos(t), where || are magnitudes of vectors and
        # cos(t) is the cosine of the angle between them, then having the values of all
        # variables except t, we can calculate it.

        if magnitude_elbow_shoulder_vector == 0 or magnitude_elbow_wrist_vector == 0:
            return 0.0

        cosine_angle = np.clip(dot_product / (magnitude_elbow_shoulder_vector * magnitude_elbow_wrist_vector), -1.0, 1.0)

        # Get the angle in degrees
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg
