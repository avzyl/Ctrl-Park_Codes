# run to define slots using mouse interaction

import cv2
import pickle

slots = []
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        slots.append((x, y))
        print(f"Point added: ({x}, {y})")

# load image
img = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/parkinglot.jpg')

if img is None:
    print("Error loading image.")
    exit()

cv2.namedWindow("Define Slots")
cv2.setMouseCallback("Define Slots", mouse_click)

while True:
    img_copy = img.copy()
    for point in slots:
        cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
    cv2.imshow("Define Slots", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cv2.destroyAllWindows('q')

# save points
with open('slots.pkl', 'wb') as f:
    pickle.dump(slots, f)
