import cv2
import numpy as np
from ultralytics import YOLO
import heapq

# ------------------------------
# 1) Load model and image
# ------------------------------
model = YOLO("yolov8n.pt")
img = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/parking_slant.jpg')
if img is None:
    print("Error loading image."); exit()

# ------------------------------
# 2) Define parking slots (your polygons)
# ------------------------------
slot_1 = np.array([[187,206],[202,237],[380,233],[360,202]], np.int32)
slot_2 = np.array([[202,238],[220,271],[406,263],[382,231]], np.int32)
slot_3 = np.array([[221,269],[407,263],[425,290],[242,299]], np.int32)
slot_4 = np.array([[438,628],[663,616],[626,557],[407,579]], np.int32)
slot_5 = np.array([[4,134],[103,132],[77,93],[0,92]], np.int32)
slot_6 = np.array([[381,527],[589,498],[557,459],[345,477]], np.int32)
slot_7 = np.array([[604,243],[633,271],[811,269],[784,234]], np.int32)
slot_8 = np.array([[635,270],[809,269],[849,297],[663,303]], np.int32)
slot_9 = np.array([[666,303],[839,294],[866,329],[699,336]], np.int32)
slot_10 = np.array([[176,822],[200,892],[469,872],[434,803]], np.int32)
slot_11 = np.array([[126,700],[148,758],[398,745],[369,682]], np.int32)
slot_12 = np.array([[65,541],[86,591],[314,581],[283,525]], np.int32)
slot_13 = np.array([[65,544],[282,526],[263,487],[47,496]], np.int32)
slot_14 = np.array([[46,496],[29,452],[237,445],[262,488]], np.int32)
slots = [slot_1,slot_2,slot_3,slot_4,slot_5,slot_6,slot_7,slot_8,slot_9,slot_11, slot_12, slot_13, slot_14]

# ------------------------------
# 3) Lane graph (roadway)
#     Tip: add more nodes to trace the lane like a polyline
# ------------------------------
gate_point = (50, img.shape[0]-50)
nodes = {
    "gate": gate_point,
    "node0": (4,157),
    "node1": (123,156),
    "node2": (433,174),
    "node3": (863,773),
    "node4": (98,872),
    "node5": (6,443),
    "node6": (803,162),
}
edges = {
    "gate": ["node4"],
    "node4": ["node5"],
    "node5": ["node0"],
    "node0": ["node1"],
    "node1": ["node2"],
    "node2": ["node6"],
    "node6": ["node3"],
    "node3": [],
}

# ------------------------------
# 4) Dijkstra helpers
# ------------------------------
def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def shortest_path(start, goal):
    pq = [(0.0, start, [])]
    seen = set()
    while pq:
        cost, u, path = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        path = path + [u]
        if u == goal: return path, cost
        for v in edges.get(u, []):
            heapq.heappush(pq, (cost + euclidean(nodes[u], nodes[v]), v, path))
    return None, float("inf")

def path_length(path):
    if not path or len(path) < 2: return float("inf")
    return sum(euclidean(nodes[path[i]], nodes[path[i+1]]) for i in range(len(path)-1))

# ------------------------------
# 5) YOLO: detect vehicles
# ------------------------------
results = model(img)
cars = []
for r in results:
    for box in r.boxes:
        label = model.names[int(box.cls[0])]
        if label in ["car", "truck", "bus"]:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, y2
            cars.append((cx,cy))
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.circle(img,(cx,cy),5,(0,0,255),-1)

# ------------------------------
# 6) Determine available slots
# ------------------------------
overlay = img.copy()
available_slots = []   # list of ((cX,cY), slot_id, poly)
available_count = 0

for i, slot in enumerate(slots, start=1):
    occupied = any(cv2.pointPolygonTest(slot, (cx,cy), False) >= 0 for (cx,cy) in cars)
    color = (0,255,0) if not occupied else (0,0,255)
    if not occupied:
        M = cv2.moments(slot)
        if M["m00"] != 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            available_slots.append(((cX,cY), i, slot))
            available_count += 1
    cv2.polylines(overlay, [slot], True, color, 2)

img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

# ------------------------------
# 7) Add available slots to graph, connect to nearest lane node
# ------------------------------
if available_slots:
    road_nodes = [k for k in nodes.keys() if (k != "gate" and not k.startswith("slot"))]

    # create slot nodes + connect them
    for (cxy, sid, _) in available_slots:
        sname = f"slot{sid}"
        nodes[sname] = cxy
        edges.setdefault(sname, [])  # slots have no outgoing edges
        # connect from nearest road node to this slot
        nearest = min(road_nodes, key=lambda n: euclidean(cxy, nodes[n]))
        edges.setdefault(nearest, []).append(sname)

    # choose best slot by shortest path from gate
    best = None
    best_len = float("inf")
    best_path = None
    for (_, sid, _) in available_slots:
        sname = f"slot{sid}"
        path, _ = shortest_path("gate", sname)
        L = path_length(path)
        if L < best_len:
            best_len, best, best_path = L, sid, path

    # draw final path and highlight slot
    if best_path:
        for i in range(len(best_path)-1):
            p1, p2 = nodes[best_path[i]], nodes[best_path[i+1]]
            cv2.line(img, p1, p2, (0,255,255), 3)

        # highlight chosen slot polygon
        chosen_poly = next(poly for (cxy, sid, poly) in available_slots if sid == best)
        cv2.polylines(img, [chosen_poly], True, (0,255,255), 3)

        # label
        cX, cY = nodes[f"slot{best}"]
        cv2.putText(img, f"Suggested Slot {best}", (cX-60, cY-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

# ------------------------------
# 8) Top banner: available count
# ------------------------------
text = f"Available: {available_count}/{len(slots)}"
(text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
x = (img.shape[1]-text_w)//2
cv2.putText(img, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

# ------------------------------
# 9) Show result
# ------------------------------
cv2.imshow("Parking with Routing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
