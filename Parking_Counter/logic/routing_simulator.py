import cv2
import numpy as np
import heapq
import math
from collections import defaultdict

# ------------------------------
# 1) Load image
# ------------------------------
img_src = cv2.imread('C:/Users/lyzza/UR_SY2526/ANPR1/parking_slant.jpg')
if img_src is None:
    print("Error loading image."); exit()

H, W = img_src.shape[:2]

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
    # bearing in degrees: 0 = east (right), 90 = south (down) in image coords if y increases downward
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    ang = math.degrees(math.atan2(dy, dx))  # -180..180
    return ang

def turn_text(delta_deg, straight_thresh=15):
    if abs(delta_deg) < straight_thresh:
        return "Go straight ahead"
    return "Turn left" if delta_deg > 0 else "Turn right"

def steps_from_points(polyline_pts, final_label):
    """
    Build steps based on change in heading between consecutive segments.
    polyline_pts: list of (x,y)
    """
    steps = []
    if len(polyline_pts) < 2:
        return ["Arrived at destination"]

    # Initial direction
    ang_prev = bearing(polyline_pts[0], polyline_pts[1])
    steps.append("Go straight ahead")

    for i in range(1, len(polyline_pts) - 1):
        a = polyline_pts[i-1]
        b = polyline_pts[i]
        c = polyline_pts[i+1]
        ang1 = bearing(a, b)
        ang2 = bearing(b, c)
        delta = (ang2 - ang1)
        # Normalize to -180..180
        while delta > 180: delta -= 360
        while delta < -180: delta += 360

        text = turn_text(delta)
        # Avoid duplicate "Go straight ahead" spam if segment is tiny
        if euclidean(a, b) > 5:
            steps.append(text)

    steps.append(f"Arrived at {final_label}")
    return steps

def interpolate_segment(a, b, step_px):
    """
    Yield intermediate points from a to b every ~step_px.
    """
    dist = euclidean(a, b)
    if dist == 0:
        return [a]
    n = max(1, int(dist // step_px))
    pts = [ (int(a[0] + (b[0]-a[0]) * t / n), int(a[1] + (b[1]-a[1]) * t / n)) for t in range(n) ]
    pts.append((int(b[0]), int(b[1])))
    return pts

def path_to_points(node_path, nodes, step_px=6):
    """
    Convert node names -> dense list of pixel points along the path.
    """
    pts = []
    for i in range(len(node_path)-1):
        p1 = nodes[node_path[i]]
        p2 = nodes[node_path[i+1]]
        seg = interpolate_segment(p1, p2, step_px)
        if i > 0 and pts:
            # Avoid duplicate point at join
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts

# ------------------------------
# 6) Interactive slot status + routing + live animation
# ------------------------------
status_list = {i+1: "occupied" for i in range(len(slots))}
SPEED_PX = 6          # pixels per tick
RUN_SIM = False
progress_idx = 0
current_points = []   # dense path points
current_steps = []    # instruction list
chosen_slot_id = None

def compute_best_route():
    """
    Build temporary graph with available slots attached to nearest road node.
    Returns: (best_slot_id, node_path, nodes_local, edges_local)
    """
    # Collect available slots with centroid
    available = []
    for i, slot in enumerate(slots, start=1):
        if status_list[i] == "available":
            M = cv2.moments(slot)
            if M["m00"] != 0:
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                available.append((i, (cX, cY), slot))

    if not available:
        return None, None, None, None

    # fresh copy of base graph
    nodes = dict(base_nodes)
    edges = {k: v[:] for k, v in base_edges.items()}

    # road node names to connect from
    road_nodes = [k for k in nodes.keys() if k != "gate" and not k.startswith("slot")]

    # connect each available slot to nearest road node
    for sid, cxy, _ in available:
        sname = f"slot{sid}"
        nodes[sname] = cxy
        edges.setdefault(sname, [])  # no outgoing
        nearest = min(road_nodes, key=lambda n: euclidean(cxy, nodes[n]))
        edges.setdefault(nearest, []).append(sname)

    # find best by path length
    best_sid = None
    best_len = float('inf')
    best_path = None
    for sid, _, _ in available:
        sname = f"slot{sid}"
        path, _ = shortest_path("gate", sname, nodes, edges)
        if not path:
            continue
        # compute length
        L = 0.0
        for i in range(len(path)-1):
            L += euclidean(nodes[path[i]], nodes[path[i+1]])
        if L < best_len:
            best_len, best_sid, best_path = L, sid, path

    return best_sid, best_path, nodes, edges

def draw_scene(base_img, node_path_pts):
    """
    Draw slots and path layers on a copy of the image.
    """
    canvas = base_img.copy()
    overlay = canvas.copy()

    # draw slots
    for i, slot in enumerate(slots, start=1):
        color = (0,255,0) if status_list[i] == "available" else (0,0,255)
        cv2.polylines(overlay, [slot], True, color, 2)

    # blend
    canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

    # draw path layers if exists
    if node_path_pts and len(node_path_pts) > 1:
        # past = red, current segment = yellow, remaining = green
        if progress_idx > 1:
            cv2.polylines(canvas, [np.array(node_path_pts[:progress_idx], np.int32)], False, (0,0,255), 3)
        # current small segment
        if progress_idx < len(node_path_pts)-1:
            seg = np.array([node_path_pts[progress_idx], node_path_pts[progress_idx+1]], np.int32)
            cv2.polylines(canvas, [seg], False, (0,255,255), 4)
        # remaining
        if progress_idx+1 < len(node_path_pts):
            cv2.polylines(canvas, [np.array(node_path_pts[progress_idx+1:], np.int32)], False, (0,200,0), 3)

        # "car" marker
        cx, cy = node_path_pts[min(progress_idx, len(node_path_pts)-1)]
        cv2.circle(canvas, (int(cx), int(cy)), 7, (0,255,255), -1)

    return canvas

def update_route_and_steps():
    global current_points, current_steps, chosen_slot_id, progress_idx
    chosen_slot_id, node_path, nodes, edges = compute_best_route()
    current_points = []
    current_steps = []
    progress_idx = 0

    if node_path:
        # convert node path to pixel polyline
        current_points = path_to_points(node_path, nodes, step_px=6)
        # build human-readable steps
        final_label = f"Slot {chosen_slot_id}" if chosen_slot_id is not None else "destination"
        current_steps = steps_from_points(current_points, f"{final_label}")

def tick_simulation():
    global progress_idx, RUN_SIM
    if not RUN_SIM or not current_points:
        return
    if progress_idx >= len(current_points)-1:
        RUN_SIM = False
        return
    # advance by SPEED_PX pixels along the dense points
    progress_idx = min(progress_idx + SPEED_PX, len(current_points)-1)

def current_instruction():
    if not current_points:
        return "No available slots"
    # Heuristic: recompute step at key indices by heading change near progress
    # We re-use steps_from_points logic but pick the next relevant step.
    # Simpler: when near end -> arrived
    if progress_idx >= len(current_points)-5:
        return f"Arrived at Slot {chosen_slot_id}"
    # Compute local turn between prev and next few points
    i = max(1, min(progress_idx, len(current_points)-2))
    a = current_points[i-1]; b = current_points[i]; c = current_points[i+1]
    ang1 = bearing(a, b); ang2 = bearing(b, c)
    delta = ang2 - ang1
    while delta > 180: delta -= 360
    while delta < -180: delta += 360
    text = turn_text(delta)
    return text

# Initial compute
update_route_and_steps()

print("Controls:")
print("  1-9 toggle slot 1-9 | A-E toggle slot 10-14")
print("  S start/pause sim    | R reset to start")
print("  + / - change speed   | Q quit")

while True:
    # run simulation tick
    tick_simulation()

    # draw scene
    frame = draw_scene(img_src, current_points)

    # UI banners
    # Available count
    available_count = sum(1 for s in status_list.values() if s == "available")
    top_text = f"Available: {available_count}/{len(slots)}"
    cv2.putText(frame, top_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Suggested slot label
    if chosen_slot_id is not None:
        cv2.putText(frame, f"Suggested: Slot {chosen_slot_id}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # Current instruction
    instr = current_instruction()
    cv2.putText(frame, f"Next: {instr}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 230, 50), 2)

    # Legend
    cv2.putText(frame, "Legend:  Green=Remaining  Yellow=Current  Red=Past",
                (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Parking Routing with Turn-by-Turn + Erasing Path", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(1,10)]:
        sid = int(chr(key))
        status_list[sid] = "available" if status_list[sid]=="occupied" else "occupied"
        update_route_and_steps()
    elif key in [ord('a'), ord('b'), ord('c'), ord('d'), ord('e')]:
        sid = 10 + (key - ord('a'))
        status_list[sid] = "available" if status_list[sid]=="occupied" else "occupied"
        update_route_and_steps()
    elif key == ord('s'):
        RUN_SIM = not RUN_SIM
    elif key == ord('r'):
        RUN_SIM = False
        progress_idx = 0
    elif key == ord('+'):
        SPEED_PX = min(25, SPEED_PX + 1)
    elif key == ord('-'):
        SPEED_PX = max(1, SPEED_PX - 1)

cv2.destroyAllWindows()
