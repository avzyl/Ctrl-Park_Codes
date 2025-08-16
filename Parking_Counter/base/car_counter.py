import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load your image
img = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/parking_slant.jpg')
if img is None:
    print("Error loading image.")
    exit()

# Define parking slot polygons
slot_1 = np.array([[187, 206], [202, 237], [380, 233], [360, 202]], np.int32)
slot_2 = np.array([[202, 238], [220, 271], [406, 263], [382, 231]], np.int32)
slot_3 = np.array([[221, 269], [407, 263], [425, 290], [242, 299]], np.int32)
slots = [slot_1, slot_2, slot_3]

# Run YOLO detection
results = model(img)

# Overlay for semi-transparent drawings
overlay = img.copy()
cars = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in ["car", "truck", "bus"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bottom-center point
            cx = int((x1 + x2) / 2)
            cy = y2
            cars.append((cx, cy))

            # --- Semi-transparent blue box lines ---
            box_color = (255, 0, 0)  # blue
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

            # --- Semi-transparent red dot ---
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)

# Blend overlay with image (adjust alpha for transparency)
alpha = 0.5
img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# Slot overlay for availability
overlay = img.copy()
for i, slot in enumerate(slots, start=1):
    slot_status = "Available"
    color = (0, 255, 0)  # green by default

    for (cx, cy) in cars:
        inside = cv2.pointPolygonTest(slot, (cx, cy), False)
        if inside >= 0:
            slot_status = "Occupied"
            color = (0, 0, 255)  # red
            break

    cv2.fillPoly(overlay, [slot], color)
    text_pos = tuple(slot[0])
    cv2.putText(img, f"Slot {i}: {slot_status}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Blend slot overlay
alpha = 0.4
img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# Show result
cv2.imshow("Parking Slots", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
