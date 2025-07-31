# License Plate Detection with YOLOv8 and EasyOCR (Real-Time via Webcam)

# This script performs real-time license plate detection and OCR using YOLOv8 and EasyOCR.
# Key Features:

# 1. Loads a trained YOLOv8 model for detecting license plates.
# 2. Captures live video from the webcam.
# 3. Draws a visual "scan zone" where detection is considered valid to reduce false positives.
# 4. Smooths bounding box movement using a history of coordinates to reduce shakiness.
# 5. Uses EasyOCR to extract text from the cropped license plate region.
# 6. Normalizes and compares OCR output to previous detections to avoid repeated saves.
# 7. Saves only unique/stable license plate images and numbers to a folder.
# 8. Draws detection boxes and plate text (with white background and black text) on screen.
# 9. Includes a cooldown/reset mechanism to clear memory if a plate disappears from the frame.
# 10. Reduces system flooding by allowing detection only inside the defined scan zone.

# Press 'q' to exit the webcam viewer.

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import deque
from ultralytics.utils import LOGGER
import os
import time
import re
from difflib import SequenceMatcher

# Suppress YOLO log messages
LOGGER.setLevel("ERROR")

# Load YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")

# Create OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Create folder for saving plates
save_dir = "detected_plates"
os.makedirs(save_dir, exist_ok=True)

# Capture from webcam
cap = cv2.VideoCapture(0)

# Smoothing and detection tracking
box_history = {}
smoothing_window = 5

# Stability tracking
plate_stability = {}
STABILITY_THRESHOLD = 5  # How many consistent frames before saving
last_plate_text = None
disappeared_frames = 0
PLATE_RESET_FRAMES = 30

# Define scanning zone (you can adjust these values)
SCAN_ZONE = {
    "x_min": 100,
    "y_min": 100,
    "x_max": 550,
    "y_max": 400
}

# Helpers
def normalize_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_similar(a, b, threshold=0.9):
    return SequenceMatcher(None, a, b).ratio() >= threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_plate_text = None
    results = model(frame, conf=0.5)

    for result in results:
        for j, box in enumerate(result.boxes.xyxy):
            conf = result.boxes.conf[j].item()
            x1, y1, x2, y2 = map(int, box)

            # Skip if bounding box is outside scanning zone
            if not (SCAN_ZONE["x_min"] <= x1 <= SCAN_ZONE["x_max"] and
                    SCAN_ZONE["y_min"] <= y1 <= SCAN_ZONE["y_max"] and
                    SCAN_ZONE["x_min"] <= x2 <= SCAN_ZONE["x_max"] and
                    SCAN_ZONE["y_min"] <= y2 <= SCAN_ZONE["y_max"]):
                continue

            # Smooth bounding box
            center_id = f"{x1}_{y1}_{x2}_{y2}"
            if center_id not in box_history:
                box_history[center_id] = deque(maxlen=smoothing_window)
            box_history[center_id].append([x1, y1, x2, y2])
            x1, y1, x2, y2 = np.mean(box_history[center_id], axis=0).astype(int)

            # Crop plate
            plate_img = frame[y1:y2, x1:x2]

            # OCR
            ocr_result = reader.readtext(plate_img, detail=0)
            if ocr_result:
                normalized = normalize_text(''.join(ocr_result).strip())
                if normalized:
                    current_plate_text = normalized

                    # Stability tracking
                    plate_stability[normalized] = plate_stability.get(normalized, 0) + 1

                    if plate_stability[normalized] == STABILITY_THRESHOLD:
                        if last_plate_text is None or not is_similar(current_plate_text, last_plate_text):
                            filename = f"{save_dir}/{current_plate_text}_{int(time.time())}.jpg"
                            cv2.imwrite(filename, plate_img)
                            print(f"[SAVED STABLE] {current_plate_text}")
                            last_plate_text = current_plate_text
                            disappeared_frames = 0
                    # else:
                    #     print(f"[WAITING] Stability {plate_stability[normalized]}/{STABILITY_THRESHOLD} for {normalized}")

                    # Draw label
                    cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), (255, 255, 255), -1)
                    cv2.putText(frame, current_plate_text, (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Reset stability for non-visible plates
    if current_plate_text is None and last_plate_text is not None:
        disappeared_frames += 1
        if disappeared_frames > PLATE_RESET_FRAMES:
            print("[RESET] Plate session cleared.")
            last_plate_text = None
            disappeared_frames = 0

    # Gradually decay old plate stability to avoid memory flooding
    for plate in list(plate_stability.keys()):
        if plate != current_plate_text:
            plate_stability[plate] = max(0, plate_stability[plate] - 1)
            if plate_stability[plate] == 0:
                del plate_stability[plate]

    # Draw scanning zone on screen
    cv2.rectangle(frame,
                (SCAN_ZONE["x_min"], SCAN_ZONE["y_min"]),
                (SCAN_ZONE["x_max"], SCAN_ZONE["y_max"]),
                (255, 255, 0), 2)

    cv2.putText(frame, "SCAN ZONE", 
                (SCAN_ZONE["x_min"], SCAN_ZONE["y_min"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
