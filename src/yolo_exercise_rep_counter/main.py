import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

stream_url = "http://localhost:4747/video"
video_capture = cv2.VideoCapture(0, cv2.CAP_MSMF)

bar_width = 75
bar_padding = 10
width_padding, height_padding = 15, 10


def format_keypoint_output(keypoint):
    return f"X = {keypoint[0]} | Y = {keypoint[1]} | Confidence = {keypoint[2]:.2f}"


while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        print("ERROR: Could not read frame from stream")
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, stream=True)

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

            left_shoulder = None
            left_elbow = None
            left_wrist = None

            right_shoulder = None
            right_elbow = None
            right_wrist = None

            if len(keypoints) > 0:
                person_kpts = keypoints[0]

                left_shoulder = person_kpts[5][:3]
                left_elbow = person_kpts[7][:3]
                left_wrist = person_kpts[9][:3]

                right_shoulder = person_kpts[6][:3]
                right_elbow = person_kpts[8][:3]
                right_wrist = person_kpts[10][:3]

                print(f"LEFT:\nShoulder: {format_keypoint_output(left_shoulder)}\n"
                      f"Elbow: {format_keypoint_output(left_elbow)}\n"
                      f"Wrist: {format_keypoint_output(left_wrist)}")

                print(f"RIGHT:\nShoulder: {format_keypoint_output(right_shoulder)}\n"
                      f"Elbow: {format_keypoint_output(right_elbow)}\n"
                      f"Wrist: {format_keypoint_output(right_wrist)}")

        annotated_frame = result.plot()

        left_keypoints = {"Shoulder": left_shoulder,
                          "Elbow": left_elbow,
                          "Wrist": left_wrist}
        right_keypoints = {"Shoulder": right_shoulder,
                           "Elbow": right_elbow,
                           "Wrist": right_wrist}

        for i, (name, keypoint) in enumerate(left_keypoints.items()):
            frame_height = annotated_frame.data.shape[0]
            confidence_score = int(keypoint[2] * 200) if keypoint is not None else 0
            bar = cv2.rectangle(annotated_frame,
                          (i * (bar_width + bar_padding), frame_height - confidence_score),
                          (i * (bar_width + bar_padding) + bar_width, frame_height),
                          (255, 0, 0),
                          -1)
            cv2.putText(annotated_frame,
                        name,
                        (i * (bar_width + bar_padding) + width_padding, frame_height - height_padding),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        2)

        for i, (name, keypoint) in enumerate(right_keypoints.items()):
            frame_height = annotated_frame.data.shape[0]
            frame_widtht = annotated_frame.data.shape[1]
            confidence_score = int(keypoint[2] * 200) if keypoint is not None else 0
            cv2.rectangle(annotated_frame,
                          (frame_widtht - i * (bar_width + bar_padding), frame_height - confidence_score),
                          (frame_widtht - i * (bar_width + bar_padding) - bar_width, frame_height),
                          (0, 0, 255),
                          -1)
            cv2.putText(annotated_frame,
                        name,
                        (frame_widtht - i * (bar_width + bar_padding) - 4 * width_padding, frame_height - height_padding),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        2)

        cv2.imshow("YOLO-Pose Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
