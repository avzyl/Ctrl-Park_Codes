import cv2
import pickle
import threading

slots = []
frame = None
lock = threading.Lock()
running = True

def mouse_click(event, x, y, flags, param):
    global slots
    if event == cv2.EVENT_LBUTTONDOWN:
        with lock:
            slots.append((x, y))
        print(f"Point added: ({x}, {y})")

def rtsp_capture():
    global frame, running
    cap = cv2.VideoCapture("rtsp://ctrlpark_admin:mingae123@192.168.100.168:554/stream1")
    if not cap.isOpened():
        print("Error opening RTSP stream")
        running = False
        return

    while running:
        ret, img = cap.read()
        if not ret:
            continue  # try again quickly
        # Resize or keep original depending on your need:
        img = cv2.resize(img, (960, 540))

        with lock:
            frame = img.copy()
    cap.release()

cv2.namedWindow("Define Slots")
cv2.setMouseCallback("Define Slots", mouse_click)

# Start RTSP capture in separate thread to keep latest frame fresh
thread = threading.Thread(target=rtsp_capture, daemon=True)
thread.start()

while running:
    with lock:
        if frame is None:
            continue
        img_display = frame.copy()

    # Draw saved points
    with lock:
        for point in slots:
            cv2.circle(img_display, point, 5, (0, 255, 0), -1)

    cv2.imshow("Define Slots", img_display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        running = False
        break
    elif key == ord('q'):
        running = False
        slots = []  # discard
        break

cv2.destroyAllWindows()

# Save points if not empty
if slots:
    with open('slots.pkl', 'wb') as f:
        pickle.dump(slots, f)
    print(f"Saved {len(slots)} points to slots.pkl")
else:
    print("No points to save.")
