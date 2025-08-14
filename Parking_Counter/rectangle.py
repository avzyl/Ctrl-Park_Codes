# run to show polygons

import cv2
import numpy as np

# load image
img = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/parkinglot.jpg')
if img is None:
    print("Error loading image.")
    exit()

# define multiple slot polygons
slot_1 = np.array([[8, 681], [145, 598], [191, 716], [33, 778]], np.int32).reshape((-1, 1, 2))
slot_2 = np.array([[347, 688], [192, 717], [145, 597], [275, 523]], np.int32).reshape((-1, 1, 2))

# group into a list
slots = [slot_1, slot_2]

while True:
    img_copy = img.copy()

    # draw each slanted box
    for slot in slots:
        cv2.polylines(img_copy, [slot], isClosed=True, color=(255, 0, 255), thickness=2)

    cv2.imshow("Parking Slots", img_copy)

    # press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

cv2.destroyAllWindows()
