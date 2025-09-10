import cv2
import numpy as np
import heapq
import math
from ultralytics import YOLO

# ------------------------------
# 1) Load image (fallback) and attempt camera
# ------------------------------
img_src = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/parking_slant.jpg')
if img_src is None:
    print("Error loading image."); exit()
H, W = img_src.shape[:2]

# Try opening webcam (0) — change to RTSP URL if needed
cap = cv2.VideoCapture(0)
use_video = cap.isOpened()
if use_video:
    print("Using webcam stream for live detection.")
else:
    print("Webcam not found — using static image for detection/display.")
    cap.release()
    cap = None

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
# 3) Lane graph (roadway) - your hardcoded path
# ------------------------------
gate_point = (50, H-50)
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
# 4) Graph helpers (Dijkstra)
# ------------------------------
def euclidean(p1, p2):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))

def shortest_path(start, goal, nodes, edges):
    pq = [(0.0, start, [])]
    seen = set()
    while pq:
        cost, u, path = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        path = path + [u]
        if u == goal:
            return path, cost
        for v in edges.get(u, []):
            heapq.heappush(pq, (cost + euclidean(nodes[u], nodes[v]), v, path))
    return None, float("inf")

# ------------------------------
# 5) Turn-by-turn instruction helpers
# ------------------------------
def bearing(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    ang = math.degrees(math.atan2(dy, dx))  # -180..180
    return ang

def turn_text(delta_deg, straight_thresh=15):
    if abs(delta_deg) < straight_thresh:
        return "Go straight ahead"
    return "Turn left" if delta_deg > 0 else "Turn right"

def steps_from_points(polyline_pts, final_label):
    steps = []
    if len(polyline_pts) < 2:
        return [f"Arrived at {final_label}"]
    steps.append("Go straight ahead")
    for i in range(1, len(polyline_pts) - 1):
        a = polyline_pts[i-1]
        b = polyline_pts[i]
        c = polyline_pts[i+1]
        ang1 = bearing(a, b)
        ang2 = bearing(b, c)
        delta = (ang2 - ang1)
        while delta > 180: delta -= 360
        while delta < -180: delta += 360
        if euclidean(a, b) > 5:
            steps.append(turn_text(delta))
    steps.append(f"Arrived at {final_label}")
    return steps

def interpolate_segment(a, b, step_px):
    dist = euclidean(a, b)
    if dist == 0:
        return [a]
    n = max(1, int(dist // step_px))
    pts = [ (int(a[0] + (b[0]-a[0]) * t / n), int(a[1] + (b[1]-a[1]) * t / n)) for t in range(n) ]
    pts.append((int(b[0]), int(b[1])))
    return pts

def path_to_points(node_path, nodes, step_px=6):
    pts = []
    for i in range(len(node_path)-1):
        p1 = nodes[node_path[i]]
        p2 = nodes[node_path[i+1]]
        seg = interpolate_segment(p1, p2, step_px)
        if i > 0 and pts:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts

# ------------------------------
# 6) Detection + slot occupancy helpers (YOLO)
# ------------------------------
model = YOLO("yolov8n.pt", verbose=False)

def detect_vehicles(frame):
    """Return list of bboxes (x,y,w,h) for vehicles detected in frame."""
    detected = []
    # small resize for faster inference (optional)
    # img_small = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # COCO classes considered as vehicles: car(2), motorcycle(3), bus(5), truck(7)
            if cls in (2,3,5,7):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected.append((x1, y1, x2-x1, y2-y1))
    return detected

def car_center_in_slot(car_bbox, slot_poly):
    x,y,w,h = car_bbox
    cx, cy = x + w//2, y + h//2
    return cv2.pointPolygonTest(slot_poly, (cx,cy), False) >= 0

# ------------------------------
# 7) Routing computation (based on status_list)
# ------------------------------
status_list = {i+1: "occupied" for i in range(len(slots))}  # default
SPEED_PX = 6
RUN_SIM = False
progress_idx = 0
current_points = []
current_steps = []
chosen_slot_id = None

def compute_best_route():
    # Build list of available slots with centroid
    available = []
    for i, slot in enumerate(slots, start=1):
        if status_list[i] == "available":
            M = cv2.moments(slot)
            if M["m00"] != 0:
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                available.append((i, (cX, cY), slot))

    if not available:
        return None, None, None, None

    nodes = dict(base_nodes)
    edges = {k: v[:] for k,v in base_edges.items()}
    road_nodes = [k for k in nodes.keys() if k != "gate" and not k.startswith("slot")]

    for sid, cxy, _ in available:
        sname = f"slot{sid}"
        nodes[sname] = cxy
        edges.setdefault(sname, [])
        nearest = min(road_nodes, key=lambda n: euclidean(cxy, nodes[n]))
        edges.setdefault(nearest, []).append(sname)

    # find best slot by path length
    best_sid = None
    best_len = float('inf')
    best_path = None
    for sid, _, _ in available:
        sname = f"slot{sid}"
        path, _ = shortest_path("gate", sname, nodes, edges)
        if not path:
            continue
        L = sum(euclidean(nodes[path[i]], nodes[path[i+1]]) for i in range(len(path)-1))
        if L < best_len:
            best_len, best_sid, best_path = L, sid, path

    return best_sid, best_path, nodes, edges

def update_route_and_steps():
    global current_points, current_steps, chosen_slot_id, progress_idx
    chosen_slot_id, node_path, nodes, edges = compute_best_route()
    current_points = []
    current_steps = []
    progress_idx = 0
    if node_path:
        current_points = path_to_points(node_path, nodes, step_px=6)
        final_label = f"Slot {chosen_slot_id}" if chosen_slot_id is not None else "destination"
        current_steps = steps_from_points(current_points, f"{final_label}")

# ------------------------------
# 8) Drawing + simulation helpers
# ------------------------------
def draw_scene(base_img, node_path_pts):
    canvas = base_img.copy()
    overlay = canvas.copy()
    for i, slot in enumerate(slots, start=1):
        color = (0,255,0) if status_list[i] == "available" else (0,0,255)
        cv2.polylines(overlay, [slot], True, color, 2)
        # put slot id label
        M = cv2.moments(slot)
        if M["m00"] != 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(overlay, f"{i}", (cX-10, cY+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

    if node_path_pts and len(node_path_pts) > 1:
        if progress_idx > 1:
            cv2.polylines(canvas, [np.array(node_path_pts[:progress_idx], np.int32)], False, (0,0,255), 3)
        if progress_idx < len(node_path_pts)-1:
            seg = np.array([node_path_pts[progress_idx], node_path_pts[progress_idx+1]], np.int32)
            cv2.polylines(canvas, [seg], False, (0,255,255), 4)
        if progress_idx+1 < len(node_path_pts):
            cv2.polylines(canvas, [np.array(node_path_pts[progress_idx+1:], np.int32)], False, (0,200,0), 3)

        cx, cy = node_path_pts[min(progress_idx, len(node_path_pts)-1)]
        cv2.circle(canvas, (int(cx), int(cy)), 7, (0,255,255), -1)

    return canvas

def tick_simulation():
    global progress_idx, RUN_SIM
    if not RUN_SIM or not current_points:
        return
    if progress_idx >= len(current_points)-1:
        RUN_SIM = False
        return
    progress_idx = min(progress_idx + SPEED_PX, len(current_points)-1)

def current_instruction():
    if not current_points:
        return "No available slots"
    if progress_idx >= len(current_points)-5:
        return f"Arrived at Slot {chosen_slot_id}"
    i = max(1, min(progress_idx, len(current_points)-2))
    a = current_points[i-1]; b = current_points[i]; c = current_points[i+1]
    ang1 = bearing(a, b); ang2 = bearing(b, c)
    delta = ang2 - ang1
    while delta > 180: delta -= 360
    while delta < -180: delta += 360
    return turn_text(delta)

# ------------------------------
# 9) Main loop: detect -> update status -> route -> draw
# ------------------------------
print("Controls: S start/pause sim | R reset | + / - speed | Q quit")
# If using static image we will process once and display; if using video we loop
first_frame = True
while True:
    if use_video:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended")
            break
        frame_for_detection = frame
    else:
        # static image path: operate on img_src each loop for UI updates
        frame_for_detection = img_src.copy()

    # 1) Detect vehicles (YOLO)
    detected_cars = detect_vehicles(frame_for_detection)  # list of (x,y,w,h)

    # 2) Update slot occupancy based on detection
    for i, slot in enumerate(slots, start=1):
        occupied = False
        for car in detected_cars:
            if car_center_in_slot(car, slot):
                occupied = True
                break
        status_list[i] = "occupied" if occupied else "available"

    # 3) Recompute route suggestion if availability changed (or every frame)
    update_route_and_steps()

    # 4) Advance simulation if running
    tick_simulation()

    # 5) Draw scene (draw on the display frame so rectangles show too)
    display_base = frame_for_detection if use_video else img_src
    frame_out = draw_scene(display_base, current_points)

    # Draw detected boxes on top for debugging/visibility
    for (x,y,w,h) in detected_cars:
        cv2.rectangle(frame_out, (x,y), (x+w,y+h), (255,200,0), 2)

    # UI text
    available_count = sum(1 for s in status_list.values() if s == "available")
    top_text = f"Available: {available_count}/{len(slots)}"
    cv2.putText(frame_out, top_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    if chosen_slot_id is not None:
        cv2.putText(frame_out, f"Suggested: Slot {chosen_slot_id}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    instr = current_instruction()
    cv2.putText(frame_out, f"Next: {instr}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,230,50), 2)
    cv2.putText(frame_out, "Legend: Green=Remaining  Yellow=Current  Red=Past", (20, H-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Parking Routing (Auto Detection)", frame_out)
    key = cv2.waitKey(1 if use_video else 30) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        RUN_SIM = not RUN_SIM
    elif key == ord('r'):
        RUN_SIM = False
        progress_idx = 0
    elif key == ord('+'):
        SPEED_PX = min(25, SPEED_PX + 1)
    elif key == ord('-'):
        SPEED_PX = max(1, SPEED_PX - 1)

    # If static image and we've displayed once and simulation not running, exit loop if user pressed q
    if not use_video and first_frame:
        # keep showing static image until user exits — don't re-run detection repeatedly unless desired
        first_frame = False

# cleanup
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
