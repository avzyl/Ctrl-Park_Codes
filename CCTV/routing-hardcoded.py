# save as app_server.py
from flask import Flask, Response, render_template_string, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import heapq
import math
import threading
import time

app = Flask(__name__)
CORS(app)

# ---------------- Config ---------------- #
VIDEO_PATH = "record_clear.mp4"
FRAME_WIDTH = 980
FRAME_HEIGHT = 540
FPS = 30

# ---------------- Load YOLO model ---------------- #
model = YOLO("yolov8n.pt")

# ---------------- Parking Slots ---------------- #

#parking2
slot_18 =np.array([[585, 352], [844, 328], [750, 279], [484, 287]], np.int32).reshape((-1, 1, 2))
slot_19 = np.array([[484, 287], [743, 278], [651, 240], [414, 241]], np.int32).reshape((-1, 1, 2))
slot_20 = np.array([[414, 241], [650, 240], [581, 210], [366, 208]], np.int32).reshape((-1, 1, 2))
slot_21 = np.array([[364, 206], [579, 212], [516, 188], [326, 185]], np.int32).reshape((-1, 1, 2))
slot_22 = np.array([[326, 185], [517, 188], [465, 161], [296, 165]], np.int32).reshape((-1, 1, 2))
slot_23 = np.array([[296, 165], [470, 161], [427, 148], [267, 146]], np.int32).reshape((-1, 1, 2))
slot_24 = np.array([[267, 146], [434, 149], [385, 136], [250, 134]], np.int32).reshape((-1, 1, 2))
slot_25 = np.array([[256, 133], [395, 134], [368, 126], [238, 127]], np.int32).reshape((-1, 1, 2))
slot_26 = np.array([[238, 127], [369, 125], [345, 116], [223, 118]], np.int32).reshape((-1, 1, 2))
slot_27 = np.array([[223, 118], [343, 115], [325, 106], [211, 112]], np.int32).reshape((-1, 1, 2))
slot_28 = np.array([[211, 112], [326, 106], [308, 100], [201, 103]], np.int32).reshape((-1, 1, 2))
slot_29 = np.array([[201, 103], [307, 99], [292, 97], [193, 97]], np.int32).reshape((-1, 1, 2))
slot_30 = np.array([[193, 97], [291, 93], [280, 91], [187, 93]], np.int32).reshape((-1, 1, 2))
slot_31 = np.array([[187, 93], [281, 90], [268, 88], [181, 90]], np.int32).reshape((-1, 1, 2))
slot_32 = np.array([[181, 90], [268, 89], [261, 82], [176, 84]], np.int32).reshape((-1, 1, 2))

#parking 3
slot_34 = np.array([[916, 306], [936, 257], [869, 233], [838, 263]], np.int32).reshape((-1, 1, 2))

# Add slot_35 since it's in your hardcoded paths
slot_35 = np.array([[837, 263], [868, 233], [804, 210], [758, 228]], np.int32).reshape((-1, 1, 2))

# Create a mapping from actual slot numbers to their positions in the list
slot_mapping = {
    18: slot_18, 19: slot_19, 20: slot_20, 21: slot_21, 22: slot_22, 23: slot_23,
    24: slot_24, 25: slot_25, 26: slot_26, 27: slot_27, 28: slot_28, 29: slot_29,
    30: slot_30, 31: slot_31, 32: slot_32, 34: slot_34, 35: slot_35
}

# Create slots list in order
slots = [slot_mapping[sid] for sid in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35]]

def get_slot_center(slot):
    M = cv2.moments(slot)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

# ---------------- Graph ---------------- #
nodes_base = {
    "gate": (52, 66),
    "node1": (91, 66),
    "node2": (103, 78),
    "node3": (249, 206),
    "node4": (276, 237),
    "node5": (306, 282),
    "node6": (361, 348),
    "node7": (506, 467),
    "node8": (968, 373),
    "node9": (931, 334),
    "node10": (873, 300),
    "node11": (784, 263),
    "node12": (700, 227),
    "node13": (636, 199),
    "node14": (571, 177),
    "node15": (524, 159),
    "node16": (411, 110),
    "node17": (359, 93),
    "node18": (314, 79),
}

edges_base = {
    "gate": ["node1"],
    "node1": ["node2"],
    "node2": ["node3"],
    "node3": ["node4"],
    "node4": ["node5"],
    "node5": ["node6"],
    "node6": ["node7"],
    "node7": ["node8"],
    "node8": ["node9"],
    "node9": ["node10"],
    "node10": ["node11"],
    "node11": ["node12"],
    "node12": ["node13"],
    "node13": ["node14"],
    "node14": ["node15"],
    "node15": ["node16"],
    "node16": ["node17"],
    "node17": ["node18"],
    "node18": []
}

# ---------------- Path Helpers ---------------- #
def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

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

def attach_slot_to_nearest_node(slot_center, sname, nodes, edges):
    nearest = min(nodes.keys(), key=lambda n: euclidean(slot_center, nodes[n]))
    nodes[sname] = slot_center
    edges[sname] = []  # slot has no outgoing edges
    edges[nearest] = [sname]  # stop traversal at slot
    return nearest

def get_turn_directions(path, nodes):
    directions = []
    for i in range(1, len(path) - 1):
        p0, p1, p2 = np.array(nodes[path[i - 1]]), np.array(nodes[path[i]]), np.array(nodes[path[i + 1]])
        v1, v2 = p1 - p0, p2 - p1
        angle = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        angle = (angle + 360) % 360
        if 45 < angle < 135:
            directions.append("Turn Left")
        elif 225 < angle < 315:
            directions.append("Turn Right")
        else:
            directions.append("Go Straight")
    return directions

# ---------------- Hardcoded Paths ---------------- #
hardcoded_paths = {
    "slot_32": ["gate", "node1", "node2", "slot_32"],
    "slot_34": ["gate", "node1", "node2", "node3", "node4", "node5", "node6", "node7", "node8", "node9", "slot_34"],
    "slot_35": ["gate", "node1", "node2", "node3", "node4", "node5", "node6", "node7", "node8", "node9", "slot_35"],
}

# ---------------- Slot Priority ---------------- #
# Define the priority order: farthest from gate first
slot_priority = [35, 34, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]

# ---------------- Globals ---------------- #
latest_directions = []
latest_path = []
latest_jpeg = None
jpeg_lock = threading.Lock()
stop_event = threading.Event()

# ---------------- Video Producer ---------------- #
def frame_producer():
    global latest_jpeg, latest_directions, latest_path

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    while not stop_event.is_set():
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        results = model(img, verbose=False)
        cars = []

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in ["car", "truck", "bus"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, y2 - 10
                    cars.append((cx, cy))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        overlay = img.copy()
        available_slots = []
        available_count = 0

        # Iterate through actual slot numbers (18-32, 34, 35)
        for slot_number in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35]:
            slot = slot_mapping[slot_number]
            occupied = any(cv2.pointPolygonTest(slot, (cx, cy), False) >= 0 for (cx, cy) in cars)
            color = (0, 255, 0) if not occupied else (0, 0, 255)
            if not occupied:
                cX, cY = get_slot_center(slot)
                if cX and cY:
                    available_slots.append(((cX, cY), slot_number, slot))
                    available_count += 1
            cv2.polylines(overlay, [slot], True, color, 2)

        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        text = f"Available: {available_count}/{len(slots)}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = (img.shape[1] - tw) // 2
        cv2.putText(img, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if available_slots:
            # print(f"DEBUG: Available slots: {[sid for (_, sid, _) in available_slots]}")  # Debug line
            
            # Prioritized slot selection: follow slot_priority order
            selected_slot = None
            
            # Check available slots in priority order (farthest first)
            for slot_id in slot_priority:
                slot_available = any(sid == slot_id for (_, sid, _) in available_slots)
                if slot_available:
                    selected_slot = slot_id
                    # print(f"DEBUG: Selected slot {selected_slot}")  # Debug line
                    break
            
            # If no priority slots available, use the first available slot
            if not selected_slot:
                selected_slot = available_slots[0][1]
                # print(f"DEBUG: Using first available slot {selected_slot}")  # Debug line
            
            sname = f"slot_{selected_slot}"
            # print(f"DEBUG: Slot name: {sname}")  # Debug line

            # Use hardcoded path if available, else shortest path
            if sname in hardcoded_paths:
                path = hardcoded_paths[sname]
                # print(f"DEBUG: Using hardcoded path for {sname}: {path}")  # Debug line
                # For hardcoded paths, we don't need to modify the graph
                nodes = dict(nodes_base)
                # Add the slot center to nodes for drawing
                slot_center = get_slot_center(slot_mapping[selected_slot])
                if slot_center:
                    nodes[sname] = slot_center
            else:
                # print(f"DEBUG: Using graph-based pathfinding for {sname}")  # Debug line
                # Fallback to graph-based pathfinding
                nodes = dict(nodes_base)
                edges = {k: list(v) for k, v in edges_base.items()}
                for (cxy, sid, _) in available_slots:
                    sname_temp = f"slot_{sid}"
                    attach_slot_to_nearest_node(cxy, sname_temp, nodes, edges)
                path, _ = shortest_path("gate", sname, nodes, edges)
                # print(f"DEBUG: Graph path result: {path}")  # Debug line

            # FIX: Check if path is not None before using it
            if path is not None:
                # Compute directions safely
                latest_path = path
                latest_directions = get_turn_directions(latest_path, nodes) if len(path) > 2 else []
                # print(f"DEBUG: Final path: {latest_path}")  # Debug line
                # print(f"DEBUG: Directions: {latest_directions}")  # Debug line

                # Draw path
                for i in range(len(latest_path) - 1):
                    if latest_path[i] in nodes and latest_path[i + 1] in nodes:
                        p1, p2 = nodes[latest_path[i]], nodes[latest_path[i + 1]]
                        cv2.line(img, p1, p2, (0, 255, 255), 3)
            else:
                # If no path found, clear directions and path
                # print("DEBUG: No path found!")  # Debug line
                latest_path = []
                latest_directions = []
        else:
            # No available slots
            latest_path = []
            latest_directions = []

        # Encode frame
        ret_enc, buffer = cv2.imencode('.jpg', img)
        if ret_enc:
            with jpeg_lock:
                latest_jpeg = buffer.tobytes()

        time.sleep(1.0 / FPS)

# ---------------- Flask Routes ---------------- #
def mjpeg_generator():
    while not stop_event.is_set():
        with jpeg_lock:
            frame = latest_jpeg
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/directions')
def directions():
    return jsonify({"directions": latest_directions, "path": latest_path})

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head><title>Parking Assistant</title></head>
    <body>
        <h1>Live Parking Detection (MJPEG)</h1>
        <img src="{{ url_for('video_feed') }}" width="960" height="540" />
        <h3><a href="{{ url_for('directions') }}" target="_blank">View Directions JSON</a></h3>
    </body>
    </html>
    ''')

# ---------------- Main ---------------- #
if __name__ == '__main__':
    producer = threading.Thread(target=frame_producer, daemon=True)
    producer.start()

    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        stop_event.set()
        producer.join(timeout=2)
