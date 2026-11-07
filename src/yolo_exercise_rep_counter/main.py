import cv2
from visualization.real_time_confidence_bars import RealTimeBarVisualizer
from visualization.exercise_highlight import ExerciseHighlightVisualizer, HighlightType
from visualization.exercise_info import ExerciseInfoVisualizer
from validators.validators import BicepCurlValidator
from models.models import YOLOv8n_pretrained_model

model = YOLOv8n_pretrained_model

stream_url = "http://localhost:4747/video"
video_capture = cv2.VideoCapture(0, cv2.CAP_MSMF)

bar_visualizer = RealTimeBarVisualizer(list(range(5, 13)), bar_width=50, text_scale=0.4, width_padding=10)
exercise_highlight_visualizer = ExerciseHighlightVisualizer([HighlightType.ARMS])

exercise_info_visualizer_left = ExerciseInfoVisualizer()
exercise_info_visualizer_right = ExerciseInfoVisualizer(phase_position=(400, 100), rep_counter_position=(400, 150))

bicep_curl_validator_left = BicepCurlValidator(required_confidence=0.5, min_phase_length=2, arm="left")
bicep_curl_validator_right = BicepCurlValidator(required_confidence=0.5, min_phase_length=2, arm="right")

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        print("ERROR: Could not read frame from stream")
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, imgsz=320, stream=True, verbose=False)

    for result in results:
        key_points = []
        if result.keypoints is not None:
            key_points = result.keypoints.data.cpu().numpy()

        annotated_frame = frame.copy()

        if len(key_points) > 0:
            person_key_points = key_points[0]
            exercise_highlight_visualizer.draw_highlights(person_key_points, annotated_frame)

        constraints_passed_left, details_left = bicep_curl_validator_left.check_constraints(key_points)
        constraints_passed_right, details_right = bicep_curl_validator_right.check_constraints(key_points)

        exercise_info_visualizer_left.visualize_name(annotated_frame, "Bicep Curl")

        if details_left.get("average_confidence", 0.0) > details_right.get("average_confidence", 0.0):
            state, rep_count = bicep_curl_validator_left.update_state(constraints_passed_left, details_left)
            exercise_info_visualizer_left.visualize_phase(annotated_frame, state.value.replace("_", " "))
            exercise_info_visualizer_left.visualize_rep_count(annotated_frame, rep_count)
        else:
            state, rep_count = bicep_curl_validator_right.update_state(constraints_passed_right, details_right)
            exercise_info_visualizer_right.visualize_phase(annotated_frame, state.value.replace("_", " "))
            exercise_info_visualizer_right.visualize_rep_count(annotated_frame, rep_count)

        cv2.imshow("YOLO-Pose Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
