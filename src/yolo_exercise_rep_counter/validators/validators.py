from abc import ABC, abstractmethod
from states import ExerciseState


class ExerciseValidator(ABC):
    def __init__(self):
        self.state = ExerciseState.INITIAL
        self.rep_count = 0

    @abstractmethod
    def check_constraints(self, key_points):
        pass

    @abstractmethod
    def update_state(self, constraints_passed, key_points_details):
        pass
