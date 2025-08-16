# use to test the polygons made using tool.py

import cv2
import numpy as np

# Load your image
img = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/cars.jpg')
if img is None:
    print("Error loading image.")
    exit()

# Define multiple slot polygons
slot_1 = np.array([[8, 33], [10, 221], [98, 218], [101, 31]], np.int32)
slot_2 = np.array([[100, 30], [96, 218], [185, 216], [187, 34]], np.int32)
# Group them into a list
slots = [slot_1, slot_2]

while True:
    img_copy = img.copy()

    # Draw each slanted box
    for slot in slots:
        cv2.polylines(img_copy, [slot], isClosed=True, color=(255, 0, 255), thickness=2)

    # Show the image
    cv2.imshow("Parking Slots", img_copy)

    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

cv2.destroyAllWindows()
