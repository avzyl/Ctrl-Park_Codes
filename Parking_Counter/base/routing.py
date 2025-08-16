import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

img = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/parking_slant.jpg')
if img is None:
    print("Error loading image.")
    exit()

# define parking slot polygons
slot_1 = np.array([[187, 206], [202, 237], [380, 233], [360, 202]], np.int32)
slot_2 = np.array([[202, 238], [220, 271], [406, 263], [382, 231]], np.int32)
slot_3 = np.array([[221, 269], [407, 263], [425, 290], [242, 299]], np.int32)
slot_4 = np.array([[438, 628], [663, 616], [626, 557], [407, 579]], np.int32)
slot_6 = np.array([[381, 527], [589, 498], [557, 459], [345, 477]], np.int32)
slot_7 = np.array([[604, 243], [633, 271], [811, 269], [784, 234]], np.int32)
slot_8 = np.array([[635, 270], [809, 269], [849, 297], [663, 303]], np.int32)
slot_9 = np.array([[666, 303], [839, 294], [866, 329], [699, 336]], np.int32)
slot_10 = np.array([[173, 820], [429, 804], [468, 872], [201, 891]], np.int32)

slots = [slot_1, slot_2, slot_3, slot_4, slot_6, slot_7, slot_8, slot_9, slot_10]


# define gate location
gate_point = (50, img.shape[0] - 50)

results = model(img)

# detect cars
cars = []
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in ["car", "truck", "bus"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cx = int((x1 + x2) / 2)
            cy = y2
            cars.append((cx, cy))
            
            # draw car box in RED
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

overlay = img.copy()

# counter for slots
available_count = 0
available_slots = []

# check each slot
for i, slot in enumerate(slots, start=1):
    slot_status = "Available"
    color = (0, 255, 0)  # green by default

    for (cx, cy) in cars:
        inside = cv2.pointPolygonTest(slot, (cx, cy), False)
        if inside >= 0:
            slot_status = "Occupied"
            color = (0, 0, 255)  # red
            break

    if slot_status == "Available":
        available_count += 1
        # get slot center for distance calculation
        M = cv2.moments(slot)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            available_slots.append(((cX, cY), i, slot))

    cv2.fillPoly(overlay, [slot], color)

alpha = 0.4
img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# --- find nearest available slot ---
if available_slots:
    nearest_slot = min(available_slots, key=lambda s: np.linalg.norm(np.array(s[0]) - np.array(gate_point)))
    (nearest_cx, nearest_cy), nearest_idx, nearest_poly = nearest_slot

    cv2.polylines(img, [nearest_poly], isClosed=True, color=(0, 255, 255), thickness=3)

    # draw route line from gate to slot
    cv2.line(img, gate_point, (nearest_cx, nearest_cy), (0, 255, 255), 3)
    cv2.circle(img, gate_point, 8, (0, 255, 255), -1)
    cv2.putText(img, f"Suggested: Slot {nearest_idx}", (nearest_cx - 50, nearest_cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

total_slots = len(slots)
cv2.putText(img, f"Available: {available_count}/{total_slots}", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

# show result
cv2.imshow("Parking Slots with Routing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
