# manually input the status of each slot (instead of relying on YOLO detection)

import cv2
import numpy as np
import heapq

# ------------------------------
# 1) Load image
# ------------------------------
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

slots = [slot_1,slot_2,slot_3,slot_4,slot_5,slot_6,slot_7,
         slot_8,slot_9,slot_10,slot_11,slot_12,slot_13,slot_14]

# ------------------------------
# 3) Lane graph (roadway)
# ------------------------------
gate_point = (50, img.shape[0]-50)
base_nodes = {
    "gate": gate_point,
    "node0": (4,157),
    "node1": (123,156),
    "node2": (433,174),
    "node3": (863,773),
    "node4": (98,872),
    "node5": (6,443),
    "node6": (803,162),
}
base_edges = {
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
# 4) Dijkstra helpers (now parameterized!)
# ------------------------------
def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def shortest_path(start, goal, nodes, edges):
    pq = [(0.0, start, [])]
    seen = set()
    while pq:
        cost, u, path = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        path = path + [u]
        if u == goal: 
            return path, cost
        for v in edges.get(u, []):
            heapq.heappush(pq, (cost + euclidean(nodes[u], nodes[v]), v, path))
    return None, float("inf")

def path_length(path, nodes):
    if not path or len(path) < 2: return float("inf")
    return sum(euclidean(nodes[path[i]], nodes[path[i+1]]) for i in range(len(path)-1))

# ------------------------------
# 5) Initialize all slots as occupied
# ------------------------------
status_list = {i+1: "occupied" for i in range(len(slots))}

# ------------------------------
# 6) Draw + Suggest Best Slot
# ------------------------------
def update_display():
    display = img.copy()
    overlay = display.copy()
    available_slots = []

    # draw slots
    for i, slot in enumerate(slots, start=1):
        if status_list[i] == "available":
            color = (0,255,0)  # green
            M = cv2.moments(slot)
            if M["m00"] != 0:
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                available_slots.append(((cX,cY), i, slot))
        else:
            color = (0,0,255)  # red
        cv2.polylines(overlay, [slot], True, color, 2)

    display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)

    # routing
    if available_slots:
        # fresh copy of graph
        nodes = base_nodes.copy()
        edges = {k: v.copy() for k,v in base_edges.items()}
        road_nodes = [k for k in nodes.keys() if (k != "gate" and not k.startswith("slot"))]

        # add slots as nodes
        for (cxy, sid, _) in available_slots:
            sname = f"slot{sid}"
            nodes[sname] = cxy
            edges.setdefault(sname, [])
            nearest = min(road_nodes, key=lambda n: euclidean(cxy, nodes[n]))
            edges.setdefault(nearest, []).append(sname)

        # choose best slot
        best = None
        best_len = float("inf")
        best_path = None
        for (_, sid, _) in available_slots:
            sname = f"slot{sid}"
            path, _ = shortest_path("gate", sname, nodes, edges)
            L = path_length(path, nodes)
            if L < best_len:
                best_len, best, best_path = L, sid, path

        # draw path
        if best_path:
            for i in range(len(best_path)-1):
                p1, p2 = nodes[best_path[i]], nodes[best_path[i+1]]
                cv2.line(display, p1, p2, (0,255,255), 3)

            chosen_poly = next(poly for (cxy, sid, poly) in available_slots if sid == best)
            cv2.polylines(display, [chosen_poly], True, (0,255,255), 3)

            cX, cY = nodes[f"slot{best}"]
            cv2.putText(display, f"Suggested Slot {best}", (cX-60, cY-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    return display

# ------------------------------
# 7) Interactive loop
# ------------------------------
print("Controls: Press 1-9 for slots 1-9, A-E for slots 10-14. Q to quit.")
while True:
    disp = update_display()
    cv2.imshow("Real-Time Parking Slots", disp)
    key = cv2.waitKey(100) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(1,10)]:
        sid = int(chr(key))
        status_list[sid] = "available" if status_list[sid]=="occupied" else "occupied"
    elif key in [ord('a'), ord('b'), ord('c'), ord('d'), ord('e')]:
        sid = 10 + (key - ord('a'))
        status_list[sid] = "available" if status_list[sid]=="occupied" else "occupied"

cv2.destroyAllWindows()
