from dataclasses import dataclass
import cv2
from enum import Enum


@dataclass(frozen=True)
class Highlight:
    connections: list[tuple[int, int]]
    joints: list[int]


class HighlightType(str, Enum):
    ARMS = "ARMS"


highlights = {
    HighlightType.ARMS: Highlight(
        connections=[
            (5, 7), (7, 9),
            (6, 8), (8, 10)
        ],
        joints=[
            5, 6, 7, 8, 9, 10
        ]
    )
}


class ExerciseHighlightVisualizer:
    def __init__(self, highlight_list: list[HighlightType]):
        self.connections = []
        self.joints = set()
        for highlight in highlight_list:
            self.connections.extend(highlights[highlight].connections)
            self.joints.update(highlights[highlight].joints)

    def draw_highlights(self, person_kpts, frame, confidence_threshold=0.5):
        for start_idx, end_idx in self.connections:
            pt1 = person_kpts[start_idx]
            pt2 = person_kpts[end_idx]

            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                pos1 = (int(pt1[0]), int(pt1[1]))
                pos2 = (int(pt2[0]), int(pt2[1]))
                cv2.line(frame, pos1, pos2, (0, 0, 255), 4)

        for idx in self.joints:
            pt = person_kpts[idx]
            if pt[2] > confidence_threshold:
                pos = (int(pt[0]), int(pt[1]))
                cv2.circle(frame, pos, 8, (0, 0, 255), -1)
                cv2.circle(frame, pos, 9, (255, 255, 255), 1)
