from flask import Flask, Response, render_template_string, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import heapq
import math

app = Flask(__name__)

# ---------------- Load YOLO model ---------------- #
model = YOLO("yolov8n.pt")

# Video capture setup
video_path = "record_clear.mp4"
cap = cv2.VideoCapture(video_path)

FRAME_WIDTH = 980
FRAME_HEIGHT = 540
cap.set(cv2.CAP_PROP_FPS, 30)

# ---------------- Parking Slots (polygons) ---------------- #
slot_34 = np.array([[916, 306], [936, 257], [869, 233], [838, 263]], np.int32).reshape((-1, 1, 2))
slot_52 = np.array([[281, 74], [339, 71], [272, 69], [330, 68]], np.int32).reshape((-1, 1, 2))

slots = [slot_34, slot_52]

def get_slot_center(slot):
    M = cv2.moments(slot)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

# ---------------- Graph Nodes & Edges ---------------- #
nodes_base = {
    "gate": (52, 66),
    "node1": (91, 66),
    "node2": (103, 78),
    "node3": (126, 81),
    "node4": (536, 497),
    "node5": (962, 342),
    "node6": (289, 79),
}
edges_base = {
    "gate": ["node1"],
    "node1": ["node2"],
    "node2": ["node3"],
    "node3": ["node4"],
    "node4": ["node5"],
    "node5": ["node6"],
    "node6": [],
}

# ---------------- Graph Utilities ---------------- #
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

def path_length(path, nodes):
    if not path or len(path) < 2:
        return float("inf")
    return sum(euclidean(nodes[path[i]], nodes[path[i + 1]]) for i in range(len(path) - 1))

def attach_slot_to_nearest_node(slot_center, sname, nodes, edges):
    nearest = min(nodes.keys(), key=lambda n: euclidean(slot_center, nodes[n]))
    nodes[sname] = slot_center
    edges[sname] = []
    edges[nearest].append(sname)

# ---------------- Directions Utility ---------------- #
def get_turn_directions(path, nodes):
    directions = []
    for i in range(1, len(path) - 1):
        p0 = np.array(nodes[path[i-1]])
        p1 = np.array(nodes[path[i]])
        p2 = np.array(nodes[path[i+1]])

        v1 = p1 - p0
        v2 = p2 - p1

        angle = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        angle = (angle + 360) % 360  # Normalize

        if 45 < angle < 135:
            directions.append("Turn Left")
        elif 225 < angle < 315:
            directions.append("Turn Right")
        else:
            directions.append("Go Straight")
    return directions

# ---------------- Globals ---------------- #
latest_directions = []
latest_path = []

# ---------------- Frame Generator ---------------- #
def generate_frames():
    global cap, latest_directions, latest_path
    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)

        ret, img = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            continue

        if FRAME_WIDTH and FRAME_HEIGHT:
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

        results = model(img, verbose=False)
        cars = []

        # Detect cars/trucks/buses
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in ["car", "truck", "bus"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = y2 - 10
                    cars.append((cx, cy))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        overlay = img.copy()
        available_slots = []
        available_count = 0

        # Slot occupancy check
        for i, slot in enumerate(slots, start=1):
            occupied = any(cv2.pointPolygonTest(slot, (cx, cy), False) >= 0 for (cx, cy) in cars)
            color = (0, 255, 0) if not occupied else (0, 0, 255)
            if not occupied:
                cX, cY = get_slot_center(slot)
                if cX is not None and cY is not None:
                    available_slots.append(((cX, cY), i, slot))
                    available_count += 1
            cv2.polylines(overlay, [slot], True, color, 2)

        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Suggest closest available slot
        if available_slots:
            nodes = dict(nodes_base)
            edges = {k: list(v) for k, v in edges_base.items()}

            for (cxy, sid, _) in available_slots:
                sname = f"slot{sid}"
                attach_slot_to_nearest_node(cxy, sname, nodes, edges)

            best = None
            best_len = float("inf")
            best_path = None
            for (_, sid, _) in available_slots:
                sname = f"slot{sid}"
                path, _ = shortest_path("gate", sname, nodes, edges)
                L = path_length(path, nodes)
                if L < best_len:
                    best_len, best, best_path = L, sid, path

            if best_path:
                latest_path = best_path
                latest_directions = get_turn_directions(best_path, nodes)

                # Draw path on video
                for i in range(len(best_path) - 1):
                    p1, p2 = nodes[best_path[i]], nodes[best_path[i + 1]]
                    cv2.line(img, p1, p2, (0, 255, 255), 3)

        # Show availability count
        text = f"Available: {available_count}/{len(slots)}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = (img.shape[1] - text_w) // 2
        cv2.putText(img, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode and stream
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- Flask Routes ---------------- #
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/directions')
def directions():
    return jsonify({"directions": latest_directions, "path": latest_path})

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head><title>Parking Assistant</title></head>
    <body>
        <h1>Parking Slot Detection</h1>
        <div style="display:flex; gap:20px;">
            
            <!-- Left: Video feed -->
            <div>
                <h2>Live Camera (Zoom & Pan)</h2>
                <div id="videoContainer" style="width:640px; height:360px; overflow:hidden; border:1px solid black; position:relative; cursor:grab;">
                    <img id="videoFeed" src="{{ url_for('video_feed') }}" style="transform-origin: center center; position:absolute; left:0; top:0;">
                </div>
            </div>

            <!-- Right: Map -->
            <div>
                <h2>2D Map (Pan, Zoom, Rotate)</h2>
                <canvas id="mapCanvas" width="640" height="360" style="border:1px solid black; cursor:grab;"></canvas>
            </div>
        </div>

        <script>
        // ---------- VIDEO ZOOM & PAN ----------
        const video = document.getElementById("videoFeed");
        const videoContainer = document.getElementById("videoContainer");

        let vScale = 1.0, vOffsetX = 0, vOffsetY = 0;
        let vDragging = false, vLastX = 0, vLastY = 0;

        videoContainer.addEventListener("mousedown", e => {
            vDragging = true;
            vLastX = e.clientX;
            vLastY = e.clientY;
            videoContainer.style.cursor = "grabbing";
        });

        window.addEventListener("mouseup", () => {
            vDragging = false;
            videoContainer.style.cursor = "grab";
        });

        window.addEventListener("mousemove", e => {
            if (vDragging) {
                let dx = e.clientX - vLastX;
                let dy = e.clientY - vLastY;
                vOffsetX += dx;
                vOffsetY += dy;
                vLastX = e.clientX;
                vLastY = e.clientY;
                updateVideoTransform();
            }
        });

        videoContainer.addEventListener("wheel", e => {
            e.preventDefault();
            let zoomFactor = 1.1;
            if (e.deltaY < 0) vScale *= zoomFactor;
            else vScale /= zoomFactor;
            updateVideoTransform();
        });

        function updateVideoTransform() {
            video.style.transform = `translate(${vOffsetX}px, ${vOffsetY}px) scale(${vScale})`;
        }

        // ---------- MAP (Pan, Zoom, Rotate w/ Shift) ----------
        let canvas = document.getElementById("mapCanvas");
        let ctx = canvas.getContext("2d");

        let nodes = {
            "gate": [52, 66],
            "node1": [91, 66],
            "node2": [103, 78],
            "node3": [126, 81],
            "node4": [536, 300],
            "node5": [550, 200],
            "node6": [289, 79]
        };
        let edges = {
            "gate": ["node1"],
            "node1": ["node2"],
            "node2": ["node3"],
            "node3": ["node4"],
            "node4": ["node5"],
            "node5": ["node6"],
            "node6": []
        };

        let rotation = 0, scale = 1.0, offsetX = 0, offsetY = 0;
        let isDragging = false, lastX = 0, lastY = 0;

        canvas.addEventListener("mousedown", e => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            canvas.style.cursor = "grabbing";
        });

        window.addEventListener("mouseup", () => {
            isDragging = false;
            canvas.style.cursor = "grab";
        });

        window.addEventListener("mousemove", e => {
            if (isDragging) {
                let dx = e.clientX - lastX;
                let dy = e.clientY - lastY;
                if (e.shiftKey) rotation += dx * 0.01;
                else { offsetX += dx; offsetY += dy; }
                lastX = e.clientX;
                lastY = e.clientY;
                drawMap();
            }
        });

        canvas.addEventListener("wheel", e => {
            e.preventDefault();
            let zoomFactor = 1.1;
            if (e.deltaY < 0) scale *= zoomFactor;
            else scale /= zoomFactor;
            drawMap();
        });

        function drawMap() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            ctx.translate(canvas.width/2 + offsetX, canvas.height/2 + offsetY);
            ctx.scale(scale, scale);
            ctx.rotate(rotation);
            ctx.translate(-canvas.width/2, -canvas.height/2);

            ctx.strokeStyle = "black"; ctx.lineWidth = 2;
            for (let u in edges) edges[u].forEach(v => {
                ctx.beginPath();
                ctx.moveTo(nodes[u][0], nodes[u][1]);
                ctx.lineTo(nodes[v][0], nodes[v][1]);
                ctx.stroke();
            });

            ctx.fillStyle = "blue";
            for (let key in nodes) {
                ctx.beginPath();
                ctx.arc(nodes[key][0], nodes[key][1], 5, 0, Math.PI*2);
                ctx.fill();
                ctx.fillText(key, nodes[key][0]+8, nodes[key][1]);
            }
            ctx.restore();
        }

        setInterval(drawMap, 100);
        </script>
    </body>
    </html>


    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
