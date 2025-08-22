from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from ultralytics import YOLO
import heapq

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("rtsp://ctrlpark_admin:mingae123@192.168.100.168:554/stream1")
FRAME_WIDTH = 980
FRAME_HEIGHT = 540

# Define slots etc. (reuse your existing functions/definitions here)
slot_1 = np.array([[455, 213], [367, 274], [631, 385], [744, 269]], np.int32).reshape((-1, 1, 2))
slot_2 = np.array([[327, 302], [548, 453], [337, 534], [162, 383]], np.int32).reshape((-1, 1, 2))
slots = [slot_1, slot_2]

def get_slot_center(slot):
    M = cv2.moments(slot)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

nodes_base = {
    "gate": (1156, 505),
    "node1": (800, 550),
    "node2": (650, 500),
}
edges_base = {
    "gate": ["node1"],
    "node1": ["node2"],
    "node2": [],
}

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
    if not path or len(path) < 2:
        return float("inf")
    return sum(euclidean(nodes[path[i]], nodes[path[i+1]]) for i in range(len(path)-1))

def generate_frames():
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if FRAME_WIDTH and FRAME_HEIGHT:
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        results = model(img, verbose=False)
        cars = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in ["car", "truck", "bus"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, y2
                    cars.append((cx, cy))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        overlay = img.copy()
        available_slots = []
        available_count = 0
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

        if available_slots:
            nodes = dict(nodes_base)
            edges = {k: list(v) for k, v in edges_base.items()}
            for (cxy, sid, _) in available_slots:
                sname = f"slot{sid}"
                nodes[sname] = cxy
                edges[sname] = []
                edges["node2"].append(sname)

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
                for i in range(len(best_path) - 1):
                    p1, p2 = nodes[best_path[i]], nodes[best_path[i + 1]]
                    cv2.line(img, p1, p2, (0, 255, 255), 3)

                chosen_poly = next(poly for (cxy, sid, poly) in available_slots if sid == best)
                cv2.polylines(img, [chosen_poly], True, (0, 255, 255), 3)
                cX, cY = nodes[f"slot{best}"]
                cv2.putText(img, f"Suggested Slot {best}", (cX - 60, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        text = f"Available: {available_count}/{len(slots)}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = (img.shape[1] - text_w) // 2
        cv2.putText(img, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        # Yield frame as multipart MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Simple HTML page to show video feed
    return render_template_string('''
    <html>
        <head><title>Parking Counter Stream</title></head>
        <body>
            <h1>Parking Slot Detection Stream</h1>
            <img src="{{ url_for('video_feed') }}" width="960" height="540">
            <p>Press 'q' in terminal to quit.</p>
        </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
