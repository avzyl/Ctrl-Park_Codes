import cv2
import numpy as np

# RTSP stream source
cap = cv2.VideoCapture("rtsp://ctrlpark_admin:mingae123@192.168.100.168:554/stream1")
if not cap.isOpened():
    print("Error opening RTSP stream.")
    exit()

# define multiple slot polygons (coordinates should be adjusted to match your stream's frame size)
slot_1 = np.array([[455, 213], [367, 274], [631, 385], [744, 269]], np.int32).reshape((-1, 1, 2))
slot_2 = np.array([[327, 302], [548, 453], [337, 534], [162, 383]], np.int32).reshape((-1, 1, 2))

slots = [slot_1, slot_2]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize if needed (optional)
    frame = cv2.resize(frame, (960, 540))

    frame_copy = frame.copy()

    # Draw polygons on the frame
    for slot in slots:
        cv2.polylines(frame_copy, [slot], isClosed=True, color=(255, 0, 255), thickness=2)

    cv2.imshow("Parking Slots Live", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
