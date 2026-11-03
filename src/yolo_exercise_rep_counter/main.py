import cv2
from data_visualization.real_time_confidence_bars import RealTimeBarVisualizer
from data_visualization.exercise_highlight import ExerciseHighlightVisualizer
from validators.validators import BicepCurlValidator
from models.models import YOLOv8n_pretrained_model

model = YOLOv8n_pretrained_model

stream_url = "http://localhost:4747/video"
video_capture = cv2.VideoCapture(0, cv2.CAP_MSMF)

bar_visualizer = RealTimeBarVisualizer(list(range(5, 13)), bar_width=50, text_scale=0.4, width_padding=10)
exercise_highlight_visualizer = ExerciseHighlightVisualizer("ARMS")

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
            person_kpts = key_points[0]

            cv2.putText(
                annotated_frame,
                "BICEP CURL",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            exercise_highlight_visualizer.highlight_arms(person_kpts, annotated_frame)

        #visualizer.visualize_bars(key_points, annotated_frame)
        constraints_passed, details = bicep_curl_validator.check_constraints(key_points)
        bicep_curl_validator.update_state(constraints_passed, details)

        cv2.imshow("YOLO-Pose Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
