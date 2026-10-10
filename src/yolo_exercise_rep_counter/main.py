import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

stream_url = "http://localhost:4747/video"
video_capture = cv2.VideoCapture(0, cv2.CAP_MSMF)


def format_keypoint_output(keypoint):
    return f"X = {keypoint[0]} | Y = {keypoint[1]} | Confidence = {keypoint[2]:.2f}"


while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        print("ERROR: Could not read frame from stream")
        break

    results = model(frame, stream=True)

    for result in results:
        annotated_frame = result.plot()
        cv2.imshow("YOLO-Pose Live Detection", annotated_frame)

        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
