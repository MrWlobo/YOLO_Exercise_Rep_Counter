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
exercise_info_visualizer = ExerciseInfoVisualizer()

bicep_curl_validator = BicepCurlValidator(required_confidence=0.5, min_phase_length=2)

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

            exercise_info_visualizer.visualize_name(annotated_frame, "Bicep Curl")
            exercise_highlight_visualizer.draw_highlights(person_key_points, annotated_frame)

        # visualizer.visualize_bars(key_points, annotated_frame)
        constraints_passed, details = bicep_curl_validator.check_constraints(key_points)
        state = bicep_curl_validator.update_state(constraints_passed, details)
        if state:
            exercise_info_visualizer.visualize_phase(annotated_frame, state.value.replace("_", " "))

        cv2.imshow("YOLO-Pose Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
