import cv2

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


class RealTimeBarVisualizer:
    def __init__(self,
                 points_to_measure: list[int],
                 bar_width: int = 75,
                 bar_padding: int = 10,
                 width_padding: int = 15,
                 height_padding: int = 10,
                 right_text_correction_multiplier: int = 4,
                 text_scale: float = 0.4):

        self.key_points_scores = {index: 0 for index in points_to_measure}
        self.bar_width = bar_width
        self.bar_padding = bar_padding
        self.width_padding = width_padding
        self.height_padding = height_padding
        self.right_text_correction_multiplier = right_text_correction_multiplier
        self.text_scale = text_scale

    def visualize_bars(self, key_points, annotated_frame):
        # Initialize dictionaries
        person_key_points, left_key_points, right_key_points = {}, {}, {}
        if len(key_points) > 0:
            person_key_points = key_points[0]

        # Put Left features on the left-hand-side, others on right-hand-side
        for index in self.key_points_scores.keys():
            if len(person_key_points) <= index:
                continue
            if "Left" in coco_key_points[index]:
                left_key_points[coco_key_points[index].split(" ")[-1]] = person_key_points[index][2]
            else:
                right_key_points[coco_key_points[index].split(" ")[-1]] = person_key_points[index][2]

        # Plot the bar charts for left features (Blue), then other features (Red) with names of the features
        for i, (name, confidence) in enumerate(left_key_points.items()):
            frame_height = annotated_frame.data.shape[0]
            confidence_score = int(confidence * frame_height * 0.3)
            cv2.rectangle(annotated_frame,
                          (i * (self.bar_width + self.bar_padding), frame_height - confidence_score),
                          (i * (self.bar_width + self.bar_padding) + self.bar_width, frame_height),
                          (255, 0, 0),
                          -1)
            cv2.putText(annotated_frame,
                        name,
                        (i * (self.bar_width + self.bar_padding) + self.width_padding,
                         frame_height - self.height_padding),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.text_scale,
                        (255, 255, 255),
                        1)

        for i, (name, confidence) in enumerate(right_key_points.items()):
            frame_height = annotated_frame.data.shape[0]
            frame_widtht = annotated_frame.data.shape[1]
            confidence_score = int(confidence * frame_height * 0.3)
            cv2.rectangle(annotated_frame,
                          (frame_widtht - i * (self.bar_width + self.bar_padding), frame_height - confidence_score),
                          (frame_widtht - i * (self.bar_width + self.bar_padding) - self.bar_width, frame_height),
                          (0, 0, 255),
                          -1)
            cv2.putText(annotated_frame,
                        name,
                        (frame_widtht - i * (self.bar_width + self.bar_padding)
                            - self.right_text_correction_multiplier * self.width_padding,
                            frame_height - self.height_padding),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.text_scale,
                        (255, 255, 255),
                        1)
