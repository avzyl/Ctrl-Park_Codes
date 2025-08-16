from flask import Flask, Response, render_template_string
import cv2
import threading
import time

# === Configuration ===
RTSP_URL = 'rtsp://ctrlpark_admin:mingae123@192.168.100.168:554/stream1'
FRAME_WIDTH = 960   # Resize width (set to None to keep original)
FRAME_HEIGHT = 540  # Resize height

# === Flask Setup ===
app = Flask(__name__)
frame = None
lock = threading.Lock()

# === HTML Template ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RTSP Stream</title>
    <style>
        body { background: #111; text-align: center; color: white; font-family: sans-serif; }
        h1 { margin-top: 40px; }
        img { margin-top: 20px; border: 4px solid #fff; }
    </style>
</head>
<body>
    <h1>Live RTSP Camera Feed</h1>
    <img src="/video_feed" width="800">
</body>
</html>
"""

# === Frame Capture Thread ===
def capture_frames():
    global frame
    while True:
        print("Connecting to RTSP stream...")
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        if not cap.isOpened():
            print("❌ Cannot connect to camera. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        print("✅ RTSP stream connected.")
        while cap.isOpened():
            for _ in range(5):  # Flush old frames
                cap.grab()
            success, img = cap.retrieve()
            if not success:
                print("⚠️ Frame retrieval failed. Reconnecting...")
                break

            # Resize for speed (optional)
            if FRAME_WIDTH and FRAME_HEIGHT:
                img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

            with lock:
                frame = img

            time.sleep(0.01)  # ~100 FPS internal (stream limits browser)

        cap.release()
        print("🔁 Reconnecting in 3 seconds...")
        time.sleep(3)

# === MJPEG Stream Generator ===
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                jpg_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
            time.sleep(0.03)  # ~30 FPS for browser
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# === Home Page ===
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# === Main ===
if __name__ == '__main__':
    t = threading.Thread(target=capture_frames, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
