from abc import ABC, abstractmethod
from .states import ExerciseState, BicepCurlState

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
    def __init__(self, state=ExerciseState.INITIAL):
        self.state = state
        self.rep_count = 0

    @abstractmethod
    def check_constraints(self, key_points):
        pass

    @abstractmethod
    def update_state(self, constraints_passed, key_points_details):
        pass


class BicepCurlValidator(ExerciseValidator):
    def __init__(self, state=BicepCurlState.INITIAL):
        super().__init__(state)

    def check_constraints(self, key_points):
        if key_points.size == 0:
            return

        indices_of_interest = list(range(5, 13))
        points_of_interest = key_points.squeeze()[indices_of_interest]

    def update_state(self, constraints_passed, key_points_details):
        pass
