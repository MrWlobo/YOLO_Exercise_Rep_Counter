import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

stream_url = "http://localhost:4747/video"
video_capture = cv2.VideoCapture(0, cv2.CAP_MSMF)

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        print("ERROR: Could not read frame from stream")
        break

    results = model(frame, stream=True)

    for result in results:
        annotated_frame = result.plot()
        cv2.imshow("YOLO-Pose Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()