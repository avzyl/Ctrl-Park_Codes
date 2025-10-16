import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import deque
from ultralytics.utils import LOGGER
import time
import re
from difflib import SequenceMatcher
from datetime import datetime
from flask import Flask, Response
import threading
import logging
import firebase_admin
from firebase_admin import credentials, firestore
import pytz

# ------------------------ LOGGING ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlateStream")

# ------------------------ YOLO + OCR SETUP ------------------------
LOGGER.setLevel("ERROR")
model = YOLO("weight/best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# ------------------------ FIREBASE SETUP ------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------------ TIMEZONE ------------------------
PH_TZ = pytz.timezone("Asia/Manila")

# ------------------------ GLOBAL TIMERS ------------------------
last_seen_times = {}
EXIT_THRESHOLD = 60  # seconds before a reappearing plate is considered "exit"

# ------------------------ FIRESTORE HELPERS ------------------------
def find_registered_user(plate_number):
    docs = db.collection("users").where("plateNumber", "==", plate_number).get()
    if docs:
        return docs[0].to_dict()
    return None

def save_gate_log_to_firestore(plate_number, confidence, event_type="entry",
                               status="pending", remarks=None, match_id=None, duration=None):
    timestamp = datetime.now(PH_TZ)
    log_id = f"GL_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    data = {
        "log_id": log_id,
        "plate_number": plate_number,
        "timestamp": timestamp,
        "event_type": event_type,
        "camera_id": "gate_cam_1",
        "confidence": float(confidence),
        "status": status,
        "match_id": match_id,
        "duration": duration,
        "remarks": remarks or "",
    }
    db.collection("gate_logs").document(log_id).set(data)
    logger.info(f"🧭 [LOGGED] {plate_number} as {event_type.upper()} ({status}) @ {timestamp.strftime('%H:%M:%S')}")

def save_entry_or_exit_to_firestore(user_info, plate_number, event_type):
    data = {
        "fullName": user_info.get("fullName", "Unknown"),
        "plate_number": plate_number,
        "role": user_info.get("role", "Unknown"),
        "idNumber": user_info.get("idNumber", "N/A"),
        "timestamp": datetime.now(PH_TZ)
    }
    collection_name = "entries" if event_type == "entry" else "exits"
    db.collection(collection_name).add(data)
    logger.info(f"📥 [{event_type.upper()}] {user_info['fullName']} ({plate_number}) logged to {collection_name}.")

# ------------------------ ROUNDABOUT DETECTION ------------------------
def analyze_roundabout_behavior(plate_number):
    """
    Determine if a vehicle exited too soon after entering (roundabout detection).
    Save result to Firestore in separate collections.
    """
    try:
        entries = list(db.collection("entries")
                       .where("plate_number", "==", plate_number)
                       .order_by("timestamp", direction=firestore.Query.DESCENDING)
                       .limit(1)
                       .stream())

        exits = list(db.collection("exits")
                     .where("plate_number", "==", plate_number)
                     .order_by("timestamp", direction=firestore.Query.DESCENDING)
                     .limit(1)
                     .stream())

        if not entries or not exits:
            return None  # not enough data yet

        entry_time = entries[0].to_dict()["timestamp"]
        exit_time = exits[0].to_dict()["timestamp"]
        duration = (exit_time - entry_time).total_seconds() / 60  # minutes

        ROUNDABOUT_THRESHOLD = 3  # minutes

        if duration < ROUNDABOUT_THRESHOLD:
            remarks = f"Roundabout detected: exited after {duration:.1f} min"
            logger.info(f"🔁 {plate_number}: {remarks}")

            # 🔹 Save to roundabout collection
            db.collection("roundabout").add({
                "plate_number": plate_number,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration_min": duration,
                "remarks": remarks,
                "timestamp": exit_time,
                "status": "bypassed"
            })
            return "roundabout"
        else:
            remarks = f"Parked normally: stayed for {duration:.1f} min"
            logger.info(f"🅿️ {plate_number}: {remarks}")

            # 🔹 Save to parked collection
            db.collection("parked").add({
                "plate_number": plate_number,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration_min": duration,
                "remarks": remarks,
                "timestamp": exit_time,
                "status": "completed"
            })
            return "parked"

    except Exception as e:
        logger.error(f"[!] Roundabout check failed: {e}")
        return None


# ------------------------ FLASK SETUP ------------------------
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

@app.route("/scanner")
def index():
    return "<h1>Ctrl+Park Live Stream</h1><img src='/scanner/video_feed'/>"

@app.route("/scanner/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/scanner/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(PH_TZ).isoformat()}

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for webcam...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
            else:
                ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

# ------------------------ DETECTION HELPERS ------------------------
def normalize_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_similar(a, b, threshold=0.9):
    return SequenceMatcher(None, a, b).ratio() >= threshold

# ------------------------ DETECTION LOOP ------------------------
def detection_loop():
    global output_frame, lock, last_seen_times

    cap = cv2.VideoCapture(0)
    while not cap.isOpened():
        logger.warning("Webcam not accessible. Retrying...")
        time.sleep(2)
        cap = cv2.VideoCapture(0)

    box_history = {}
    smoothing_window = 5
    plate_stability = {}
    STABILITY_THRESHOLD = 5

    last_plate_text = None
    disappeared_frames = 0
    PLATE_RESET_FRAMES = 150

    SCAN_ZONE = {"x_min": 100, "y_min": 100, "x_max": 550, "y_max": 400}
    logger.info("🚗 Starting detection loop...")

    display_text = ""
    display_color = (0, 255, 0)
    display_timer = 0
    DISPLAY_DURATION = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam.")
            continue

        current_plate_text = None
        results = model(frame, conf=0.5)

        for result in results:
            for j, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                confidence = float(result.boxes.conf[j])

                if not (SCAN_ZONE["x_min"] <= x1 <= SCAN_ZONE["x_max"] and
                        SCAN_ZONE["y_min"] <= y1 <= SCAN_ZONE["y_max"] and
                        SCAN_ZONE["x_min"] <= x2 <= SCAN_ZONE["x_max"] and
                        SCAN_ZONE["y_min"] <= y2 <= SCAN_ZONE["y_max"]):
                    continue

                center_id = f"{x1}_{y1}_{x2}_{y2}"
                if center_id not in box_history:
                    box_history[center_id] = deque(maxlen=smoothing_window)
                box_history[center_id].append([x1, y1, x2, y2])
                x1, y1, x2, y2 = np.mean(box_history[center_id], axis=0).astype(int)

                plate_img = frame[y1:y2, x1:x2]
                ocr_result = reader.readtext(plate_img, detail=0)
                if ocr_result:
                    normalized = normalize_text(''.join(ocr_result).strip())
                    if normalized:
                        current_plate_text = normalized
                        plate_stability[normalized] = plate_stability.get(normalized, 0) + 1

                        if plate_stability[normalized] == STABILITY_THRESHOLD:
                            if last_plate_text is None or not is_similar(current_plate_text, last_plate_text):
                                current_time = time.time()
                                if normalized not in last_seen_times:
                                    event_type = "entry"
                                else:
                                    time_diff = current_time - last_seen_times[normalized]
                                    event_type = "exit" if time_diff > EXIT_THRESHOLD else None

                                last_seen_times[normalized] = current_time

                                if event_type:
                                    user_info = find_registered_user(current_plate_text)
                                    timestamp = datetime.now(PH_TZ)

                                    if user_info:
                                        display_text = f"{user_info['fullName']} - {current_plate_text} ({event_type.upper()})"
                                        display_color = (0, 255, 0)
                                        save_gate_log_to_firestore(
                                            current_plate_text,
                                            confidence,
                                            event_type=event_type,
                                            status="matched",
                                            remarks=f"Detected registered user ({event_type})",
                                            match_id=user_info.get("plateNumber")
                                        )
                                        save_entry_or_exit_to_firestore(user_info, current_plate_text, event_type)

                                        # 🧠 Check for roundabout when exiting
                                        if event_type == "exit":
                                            behavior = analyze_roundabout_behavior(current_plate_text)
                                            if behavior == "roundabout":
                                                display_text = f"{current_plate_text} - Roundabout detected"
                                                display_color = (0, 165, 255)
                                    else:
                                        display_text = f"Unauthorized: {current_plate_text} ({event_type.upper()})"
                                        display_color = (0, 0, 255)
                                        save_gate_log_to_firestore(
                                            current_plate_text,
                                            confidence,
                                            event_type=event_type,
                                            status="unverified",
                                            remarks=f"No match found ({event_type})"
                                        )

                                    display_timer = DISPLAY_DURATION
                                    last_plate_text = current_plate_text
                                    disappeared_frames = 0

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if current_plate_text is None and last_plate_text is not None:
            disappeared_frames += 1
            if disappeared_frames > PLATE_RESET_FRAMES:
                logger.info(f"🔄 Plate {last_plate_text} left frame — system ready for new scan.")
                last_plate_text = None
                disappeared_frames = 0
                plate_stability.clear()

        for plate in list(plate_stability.keys()):
            if plate != current_plate_text:
                plate_stability[plate] = max(0, plate_stability[plate] - 1)
                if plate_stability[plate] == 0:
                    del plate_stability[plate]

        cv2.rectangle(frame,
                      (SCAN_ZONE["x_min"], SCAN_ZONE["y_min"]),
                      (SCAN_ZONE["x_max"], SCAN_ZONE["y_max"]),
                      (255, 255, 0), 2)
        cv2.putText(frame, "SCAN ZONE",
                    (SCAN_ZONE["x_min"], SCAN_ZONE["y_min"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if display_timer > 0:
            text_x = SCAN_ZONE["x_min"]
            text_y = SCAN_ZONE["y_min"] - 50
            cv2.putText(frame, display_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 3)
            display_timer -= 1

        with lock:
            output_frame = frame.copy()

    cap.release()

# ------------------------ START EVERYTHING ------------------------
if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5002, debug=False)
