import cv2


class ExerciseInfoVisualizer:
    def __init__(self,
                 name_position=(30, 50),
                 phase_position=(30, 100),
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_size=1.0,
                 font_color=(0, 255, 0),
                 font_thickness=2,
                 line_type=cv2.LINE_AA,
                 phase_size_scale=0.8):

        self.name_position = name_position
        self.phase_position = phase_position
        self.font = font
        self.exercise_font_size = font_size
        self.phase_font_size = font_size * phase_size_scale
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.line_type = line_type

    def visualize_name(self, frame, exercise_name: str):
        cv2.putText(
            frame,
            exercise_name,
            self.name_position,
            self.font,
            self.exercise_font_size,
            self.font_color,
            self.font_thickness,
            self.line_type
        )

    def visualize_phase(self, frame, exercise_phase: str):
        cv2.putText(
            frame,
            exercise_phase,
            self.phase_position,
            self.font,
            self.phase_font_size,
            self.font_color,
            self.font_thickness,
            self.line_type
        )
