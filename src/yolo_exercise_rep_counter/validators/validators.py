from abc import ABC, abstractmethod
from .states import ExerciseState, BicepCurlState
from yolo_exercise_rep_counter.utils.utils import calculate_angle, are_same_states

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
    def __init__(self, state=ExerciseState.INITIAL, required_confidence=0.6, min_phase_length=3):
        self.state = state
        self.required_confidence = required_confidence
        self.min_phase_length = min_phase_length
        self.last_moves = []
        self.continuous_phases = []
        self.rep_count = 0

    @abstractmethod
    def check_constraints(self, key_points):
        pass

    @abstractmethod
    def update_state(self, constraints_passed: bool, details: dict[str: float]):
        pass


class BicepCurlValidator(ExerciseValidator):
    def __init__(self, state=BicepCurlState.UNDETECTED, required_confidence=0.6, min_phase_length=3):
        super().__init__(state, required_confidence, min_phase_length)
        self.previous_frame_elbow_angle = None
        self.proper_exercise_pattern = [BicepCurlState.INITIAL,
                                        BicepCurlState.MOVING_UP,
                                        BicepCurlState.PEAK,
                                        BicepCurlState.MOVING_DOWN,
                                        BicepCurlState.INITIAL]

    def check_constraints(self, key_points):
        if key_points.size == 0:
            return False, {}

        first_person_points = key_points[0]
        constraints_passed = True
        details = {}

        # Constraints
        # None for now

        # Details
        elbow_angle = calculate_angle(first_person_points[6],
                                      first_person_points[8],
                                      first_person_points[10],
                                      self.required_confidence)

        details["elbow_angle"] = elbow_angle

        return constraints_passed, details

    def update_state(self, constraints_passed, details):
        if not constraints_passed:
            return

        elbow_angle = details["elbow_angle"]

        # Update state
        if elbow_angle is None:
            self.state = BicepCurlState.UNDETECTED
        elif elbow_angle >= 150:
            self.state = BicepCurlState.INITIAL
        elif elbow_angle < 150 and self.previous_frame_elbow_angle and elbow_angle < self.previous_frame_elbow_angle:
            self.state = BicepCurlState.MOVING_UP
        elif elbow_angle > 45 and self.previous_frame_elbow_angle and elbow_angle > self.previous_frame_elbow_angle:
            self.state = BicepCurlState.MOVING_DOWN

        if elbow_angle and elbow_angle <= 45:
            self.state = BicepCurlState.PEAK

        # Update previous angle
        self.previous_frame_elbow_angle = elbow_angle

        # Update phase lists

        # Last moves (of length min_phase or less) is the last states read by the model. We use a window instead
        # of single values to deal with potential mismatches between what the model sees and what is actually happening.
        # (e.g. a single MOVING DOWN between many MOVING UP is likely a system's inaccuracy)

        last_moves_filled = len(self.last_moves) == self.min_phase_length

        if last_moves_filled:
            self.last_moves.pop(0)
        self.last_moves.append(self.state)

        # If at least min_phase_length of last registered states are the same, we consider them an actual
        # output, not an error.

        last_moves_has_one_state_value = len(set(self.last_moves)) == 1
        current_state_not_logged = len(self.continuous_phases) == 0 or self.continuous_phases[-1] != self.state

        if last_moves_filled and last_moves_has_one_state_value and current_state_not_logged:
            self.continuous_phases.append(self.state)

        # If there were enough distinct phases registered, a match is ran to determine whether the exercise has
        # been performed correctly. If that is the case, a repetition is counted.

        enough_distinct_phases = len(self.continuous_phases) >= len(self.proper_exercise_pattern)
        latest_phases_match_exercise_pattern = are_same_states(
            self.continuous_phases[-len(self.proper_exercise_pattern):],
            self.proper_exercise_pattern)

        if enough_distinct_phases and latest_phases_match_exercise_pattern:
            self.rep_count += 1
            print(f"REP\nCurrent reps: {self.rep_count}")
            self.continuous_phases.clear()
