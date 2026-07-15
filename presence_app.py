from datetime import datetime, time
from functools import wraps
import base64
import csv
import os
import sys
import threading
import time as time_module

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from database import (get_db_connection, init_db, create_user, get_user_by_username,
                       get_admin_count, save_face_encoding, get_all_face_encodings,
                       get_all_students, save_student_face_encoding, get_all_student_face_encodings,
                       save_student_photo_path, get_active_schedule_for_student,
                       log_classroom_event, insert_students, insert_schedules, insert_enrollments,
                       get_all_courses, get_student_enrollments, enroll_student, unenroll_student,
                       add_course, delete_course)

app = Flask(__name__)
app.secret_key = 'super-secret-development-key-12345'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "presence_logs")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "frames")
PHOTO_DIR = os.path.join(BASE_DIR, "static", "student_photos")
CSV_LOG = os.path.join(LOG_DIR, "presence_events.csv")
HEARTBEAT_SECONDS = 180
PRESENCE_LOST_SECONDS = 10
DISPLAY_SIZE = (960, 540)
TARGET_STREAM_FPS = 30
DETECTION_EVERY_N_FRAMES = 5
JPEG_QUALITY = 75


DESIGNATED_START = time(14, 40)  # 2:40 PM
DESIGNATED_END   = time(16, 40)  # 4:40 PM


def is_designated_time(now):
    current = now.time()
    return DESIGNATED_START <= current <= DESIGNATED_END


DEFAULT_CONFIG = {
    "start_time": "08:00",
    "end_time": "17:00",
    "timezone_label": "Local computer time",
    "snapshot_cooldown_seconds": 60,
    "motion_sensitivity": 2500,
    "cameras": [
        {
            "id": "laptop",
            "name": "Laptop Camera",
            "source": "0",
            "enabled": True,
        },
        {
            "id": "phone",
            "name": "Android IP Camera",
            "source": "http://172.19.248.156:8080/video",
            "enabled": True,
        },
    ],
}

config_lock = threading.Lock()
config = DEFAULT_CONFIG.copy()
config["cameras"] = [camera.copy() for camera in DEFAULT_CONFIG["cameras"]]

camera_workers = {}
camera_workers_lock = threading.Lock()
recent_events = []
recent_events_lock = threading.Lock()


# ─── Helpers ────────────────────────────────────────────────────────────────

def ensure_log_files():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp", "camera_id", "camera_name", "status", "message", "snapshot"])


def parse_clock(value):
    hour, minute = value.split(":")
    return time(hour=int(hour), minute=int(minute))


def schedule_status(now, start_value, end_value):
    start = parse_clock(start_value)
    end = parse_clock(end_value)
    current = now.time()

    if start <= end:
        in_window = start <= current <= end
        overtime = current > end
    else:
        in_window = current >= start or current <= end
        overtime = end < current < start

    if overtime:
        return "overtime"
    if in_window:
        return "scheduled"
    return "outside_schedule"


def source_from_value(value):
    if str(value).strip().isdigit():
        return int(value)
    return str(value).strip()


def log_event(timestamp, camera_id, camera_name, status, message, snapshot):
    ensure_log_files()
    row = [timestamp.strftime("%Y-%m-%d %H:%M:%S"), camera_id, camera_name, status, message, snapshot]

    with open(CSV_LOG, "a", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(row)

    event = {
        "timestamp": row[0],
        "camera_id": camera_id,
        "camera_name": camera_name,
        "status": status,
        "message": message,
        "snapshot": snapshot,
    }

    with recent_events_lock:
        recent_events.insert(0, event)
        del recent_events[50:]


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ─── Camera Worker ──────────────────────────────────────────────────────────

class CameraWorker:
    def __init__(self, camera_config):
        self.camera_id = camera_config["id"]
        self.name = camera_config["name"]
        self.source = camera_config["source"]
        self.enabled = camera_config.get("enabled", True)
        self.capture = None
        self.previous_gray = None
        self.last_jpeg = None
        self.last_presence = False
        self.last_status = "starting"
        self.last_snapshot_at = 0
        self.room_occupied = False
        self.last_heartbeat = 0
        self.last_presence_seen_at = 0
        self.frame_index = 0
        self.cached_detections = []
        self.cached_motion_boxes = []
        self.cached_motion_detected = False
        self.cached_presence = False
        self.cached_status = "no_presence"
        self.cached_label = "No presence"
        self.cached_color = (180, 180, 180)
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)

        self.student_encodings = []
        self.encodings_loaded_at = 0
        self.id_cycle = 0
        self.cached_identifications = []  # list of (student_id, name)

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.upper_body_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_upperbody.xml"
        )

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()

    def open_capture(self):
        if self.capture is not None:
            self.capture.release()

        self.capture = cv2.VideoCapture(source_from_value(self.source))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
        self.capture.set(cv2.CAP_PROP_FPS, TARGET_STREAM_FPS)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def run(self):
        ensure_log_files()
        self.open_capture()

        while self.running:
            frame_started_at = time_module.time()

            if not self.enabled:
                time_module.sleep(0.5)
                continue

            success, frame = self.capture.read() if self.capture else (False, None)
            if not success:
                self.last_status = "camera_unavailable"
                self.open_capture()
                time_module.sleep(1)
                continue

            
            if self.camera_id == "phone":
                frame = cv2.resize(frame, (640, 360))
            else:
                frame = cv2.resize(frame, DISPLAY_SIZE)

            processed, presence_detected, status = self.process_frame(frame)

            with self.lock:
                self.last_jpeg = processed
                self.last_presence = presence_detected
                self.last_status = status

            elapsed = time_module.time() - frame_started_at
            delay = max(0, (1 / TARGET_STREAM_FPS) - elapsed)
            time_module.sleep(delay)

    def process_frame(self, frame):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        with config_lock:
            start_time = config["start_time"]
            end_time = config["end_time"]
            motion_sensitivity = int(config["motion_sensitivity"])
            snapshot_cooldown = int(config["snapshot_cooldown_seconds"])

        should_detect = (
            self.frame_index % DETECTION_EVERY_N_FRAMES == 0
            or self.previous_gray is None
        )
        self.frame_index += 1

        if should_detect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(equalized, (21, 21), 0)

            detections = []
            detections.extend(self.detect_with_cascade(self.face_detector, equalized, "face", 1.1, 5))
            detections.extend(self.detect_with_cascade(self.profile_detector, equalized, "profile", 1.1, 5))
            detections.extend(self.detect_with_cascade(self.upper_body_detector, equalized, "upper body", 1.05, 4))

            motion_detected = False
            motion_boxes = []

            if self.previous_gray is not None:
                delta = cv2.absdiff(self.previous_gray, blurred)
                threshold = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                threshold = cv2.dilate(threshold, None, iterations=2)
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < motion_sensitivity:
                        continue
                    motion_detected = True
                    motion_boxes.append(cv2.boundingRect(contour))

            self.previous_gray = blurred
            presence_detected = bool(detections) or motion_detected
            schedule = schedule_status(now, start_time, end_time)

            in_designated = is_designated_time(now)

            # Set status for logging purposes (keeps overtime/schedule info)
            if presence_detected and schedule == "overtime":
                status = "overtime"
            elif presence_detected and schedule == "scheduled":
                status = "scheduled_presence"
            elif presence_detected:
                status = "outside_schedule_presence"
            else:
                status = "no_presence"

            # Set label and color for display based on motion and designated time
            if presence_detected:
                label = "Motion detected"
                color = (0, 180, 0) if in_designated else (0, 0, 255)  # Green in designated, Red outside
            else:
                label = "Room Empty"
                color = (0, 255, 255) if in_designated else (0, 180, 0)  # Yellow in designated, Green outside

            # ── Student identification (every 15 detection cycles) ──────────
            self.id_cycle += 1
            if self.id_cycle % 15 == 1:
                self._reload_encodings_if_needed()
                if detections:
                    self.cached_identifications = self._identify_faces(frame, detections)

            if self.cached_identifications and presence_detected:
                student_id, student_name = self.cached_identifications[0]
                active_sched = get_active_schedule_for_student(student_id)
                if active_sched:
                    label  = f"{student_name} — In Class ({active_sched['course_code']})"
                    status = "scheduled_presence"
                    color  = (0, 180, 0) if in_designated else (0, 0, 255)
                else:
                    label  = f"{student_name} — No Class Scheduled"
                    status = "outside_schedule_presence"
                    color  = (0, 180, 0) if in_designated else (0, 0, 255)
            elif presence_detected and not self.cached_identifications:
                label = label + " (Unidentified)"
            # ────────────────────────────────────────────────────────────────

            self.cached_detections = detections
            self.cached_motion_boxes = motion_boxes
            self.cached_motion_detected = motion_detected
            self.cached_presence = presence_detected
            self.cached_status = status
            self.cached_label = label
            self.cached_color = color
        else:
            detections = self.cached_detections
            motion_boxes = self.cached_motion_boxes
            motion_detected = self.cached_motion_detected
            presence_detected = self.cached_presence
            status = self.cached_status
            label = self.cached_label
            color = self.cached_color

        for x, y, w, h, detector_name in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, detector_name, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not detections and motion_detected:
            for x, y, w, h in motion_boxes[:6]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 78), (20, 20, 20), -1)
        cv2.putText(frame, timestamp, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, label, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        jpeg = buffer.tobytes() if ret else None

        current_time = time_module.time()

        if not should_detect:
            return jpeg, presence_detected, status

        detected_sid = self.cached_identifications[0][0] if self.cached_identifications else None

        if presence_detected and not self.room_occupied:
            self.room_occupied = True
            self.last_heartbeat = current_time
            self.last_presence_seen_at = current_time
            snapshot = self.save_snapshot(frame, now, status)
            log_event(now, self.camera_id, self.name, status, label, snapshot)
            log_classroom_event(now, status, detected_sid, snapshot, label)

        elif presence_detected and self.room_occupied:
            heartbeat_interval = max(snapshot_cooldown, HEARTBEAT_SECONDS)
            self.last_presence_seen_at = current_time
            if current_time - self.last_heartbeat >= heartbeat_interval:
                self.last_heartbeat = current_time
                snapshot = self.save_snapshot(frame, now, status)
                log_event(now, self.camera_id, self.name, status, "Presence Still Active", snapshot)
                log_classroom_event(now, status, detected_sid, snapshot, "Presence Still Active")

        elif (
            not presence_detected
            and self.room_occupied
            and current_time - self.last_presence_seen_at >= PRESENCE_LOST_SECONDS
        ):
            self.room_occupied = False
            self.last_heartbeat = 0
            self.last_presence_seen_at = 0
            self.cached_identifications = []
            snapshot = self.save_snapshot(frame, now, "no_presence")
            log_event(now, self.camera_id, self.name, "no_presence", "Room Empty", snapshot)
            log_classroom_event(now, "no_presence", None, snapshot, "Room Empty")

        return jpeg, presence_detected, status

    def detect_with_cascade(self, detector, gray, detector_name, scale, neighbors):
        if detector.empty():
            return []
        results = detector.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(50, 50))
        return [(x, y, w, h, detector_name) for x, y, w, h in results]

    def save_snapshot(self, frame, now, status):
        ensure_log_files()
        filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{self.camera_id}_{status}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, frame)
        return os.path.join("presence_logs", "frames", filename)

    def _reload_encodings_if_needed(self):
        if time_module.time() - self.encodings_loaded_at > 300:
            self.student_encodings = get_all_student_face_encodings()
            self.encodings_loaded_at = time_module.time()

    def _identify_faces(self, frame, detections):
        """Try to match detected faces against stored student encodings."""
        identified = []
        if not self.student_encodings:
            return identified

        try:
            from deepface import DeepFace
        except ImportError:
            return identified

        for x, y, w, h, dname in detections:
            if dname != "face":
                continue
            face_roi = frame[max(0, y):y + h, max(0, x):x + w]
            if face_roi.size == 0:
                continue
            try:
                result = DeepFace.represent(
                    img_path=face_roi,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="skip"
                )
                input_enc = np.array(result[0]["embedding"])
                best_id = best_name = None
                best_dist = float("inf")
                for sid, name, enc in self.student_encodings:
                    dist = float(np.linalg.norm(input_enc - enc))
                    if dist < best_dist:
                        best_dist = dist
                        best_id = sid
                        best_name = name
                if best_dist < 10.0:
                    identified.append((best_id, best_name))
            except Exception:
                pass
        return identified

    def mjpeg(self):
        while self.running:
            with self.lock:
                frame = self.last_jpeg

            if frame is None:
                time_module.sleep(0.1)
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# ─── Worker management ──────────────────────────────────────────────────────

def sync_workers():
    with config_lock:
        cameras = [camera.copy() for camera in config["cameras"]]

    with camera_workers_lock:
        active_ids = {camera["id"] for camera in cameras}

        for camera_id in list(camera_workers):
            if camera_id not in active_ids:
                camera_workers[camera_id].stop()
                del camera_workers[camera_id]

        for camera_config in cameras:
            existing = camera_workers.get(camera_config["id"])
            if existing and (
                existing.source != camera_config["source"]
                or existing.enabled != camera_config.get("enabled", True)
                or existing.name != camera_config["name"]
            ):
                existing.stop()
                del camera_workers[camera_config["id"]]
                existing = None

            if existing is None:
                worker = CameraWorker(camera_config)
                camera_workers[camera_config["id"]] = worker
                worker.start()


workers_started = False
workers_started_lock = threading.Lock()


def ensure_workers_started():
    """Start camera workers on first dashboard visit, not at process boot.

    Cameras hold exclusive OS-level access on Windows, so starting them
    eagerly at boot blocks the browser's getUserMedia() calls used for
    face login/registration. Deferring until someone actually reaches the
    dashboard keeps the webcam free for those flows until it's needed.
    """
    global workers_started
    with workers_started_lock:
        if not workers_started:
            sync_workers()
            workers_started = True


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'logged_in' in session:
        return redirect(url_for('index'))

    # First-time setup — no admin exists yet
    if get_admin_count() == 0:
        return redirect(url_for('register'))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            error = "Please enter both username and password."
        else:
            user = get_user_by_username(username)
            if user and check_password_hash(user["password_hash"], password):
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for("index"))
            error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not username or not email or not password:
            error = "All fields are required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            hashed = generate_password_hash(password)
            success, msg = create_user(username, email, hashed)
            if success:
                return redirect(url_for("register_face", username=username))
            error = msg

    is_setup = get_admin_count() == 0
    return render_template("register.html", error=error, is_setup=is_setup)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    ensure_workers_started()
    return render_template("presence.html", username=session.get('username', ''))


@app.route("/video/<camera_id>")
@login_required
def video(camera_id):
    with camera_workers_lock:
        worker = camera_workers.get(camera_id)

    if worker is None:
        return "Camera not found", 404

    return Response(worker.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/config", methods=["GET", "POST"])
@login_required
def api_config():
    global config

    if request.method == "POST":
        payload = request.get_json(force=True)
        camera_source = payload.get("phone_source", "").strip()

        with config_lock:
            config["start_time"] = payload.get("start_time", config["start_time"])
            config["end_time"] = payload.get("end_time", config["end_time"])
            config["motion_sensitivity"] = int(payload.get("motion_sensitivity", config["motion_sensitivity"]))
            config["snapshot_cooldown_seconds"] = int(
                payload.get("snapshot_cooldown_seconds", config["snapshot_cooldown_seconds"])
            )

            for camera in config["cameras"]:
                if camera["id"] == "phone" and camera_source:
                    camera["source"] = camera_source
                if camera["id"] == "laptop":
                    camera["enabled"] = bool(payload.get("laptop_enabled", True))
                if camera["id"] == "phone":
                    camera["enabled"] = bool(payload.get("phone_enabled", True))

        sync_workers()

    with config_lock:
        return jsonify(config)


@app.route("/api/events")
@login_required
def api_events():
    with recent_events_lock:
        return jsonify(recent_events)


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT pl.id, pl.timestamp, pl.log_type, pl.detected_student_id,
               pl.snapshot_path, pl.message,
               CONCAT(s.first_name, ' ', s.last_name) AS student_name,
               s.photo_path
        FROM presence_logs pl
        LEFT JOIN students s ON pl.detected_student_id = s.student_id
        ORDER BY pl.timestamp DESC
        LIMIT 50
    """)
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("dashboard.html", logs=logs, username=session.get('username', ''))


@app.route("/register/face")
def register_face():
    username = request.args.get("username", "")
    if not username or not get_user_by_username(username):
        return redirect(url_for("login"))
    return render_template("register_face.html", username=username)


@app.route("/api/save-face", methods=["POST"])
def api_save_face():
    data = request.get_json(force=True)
    username = data.get("username", "")
    image_data = data.get("image", "")

    if not username or not image_data:
        return jsonify({"ok": False, "error": "Missing data"}), 400

    if not get_user_by_username(username):
        return jsonify({"ok": False, "error": "Unknown user"}), 404

    try:
        from deepface import DeepFace
        _, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        embedding = result[0]["embedding"]
        ok = save_face_encoding(username, embedding)
        if not ok:
            return jsonify({"ok": False, "error": "Could not save face for this user."}), 400
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/save-face-upload", methods=["POST"])
def api_save_face_upload():
    username = request.form.get("username", "").strip()
    file = request.files.get("photo")

    if not username or not file or not file.filename:
        return jsonify({"ok": False, "error": "Missing username or photo"}), 400

    if not get_user_by_username(username):
        return jsonify({"ok": False, "error": "Unknown user"}), 404

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return jsonify({"ok": False, "error": "Only JPG or PNG files are allowed"}), 400

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode image file"}), 400

    try:
        from deepface import DeepFace
        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        ok = save_face_encoding(username, result[0]["embedding"])
        if not ok:
            return jsonify({"ok": False, "error": "Could not save face for this user."}), 400
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/face-login", methods=["POST"])
def api_face_login():
    data = request.get_json(force=True)
    image_data = data.get("image", "")

    if not image_data:
        return jsonify({"ok": False, "error": "No image"}), 400

    try:
        from deepface import DeepFace
        _, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        input_enc = np.array(result[0]["embedding"])

        stored = get_all_face_encodings()
        if not stored:
            return jsonify({"ok": False, "error": "No faces registered yet."})

        best_id, best_dist = None, float("inf")
        for sid, enc in stored:
            dist = float(np.linalg.norm(input_enc - enc))
            if dist < best_dist:
                best_dist = dist
                best_id = sid

        THRESHOLD = 10.0
        if best_dist < THRESHOLD:
            session['logged_in'] = True
            session['username'] = best_id
            return jsonify({"ok": True, "username": best_id})
        else:
            return jsonify({"ok": False, "error": "Face not recognized. Please try again."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/upload")
@login_required
def upload():
    return render_template("upload.html", username=session.get('username', ''))


@app.route("/api/upload/students", methods=["POST"])
@login_required
def upload_students():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"ok": False, "error": "Please upload a CSV file."}), 400
    try:
        stream = file.stream.read().decode("utf-8").splitlines()
        reader = csv.DictReader(stream, fieldnames=["student_id", "last_name", "first_name"])
        rows = [r for r in reader if r["student_id"].strip() and r["student_id"].strip().lower() != "studentid"]
        inserted, skipped, errors = insert_students(rows)
        return jsonify({"ok": True, "inserted": inserted, "skipped": skipped, "errors": errors[:5]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/upload/schedules", methods=["POST"])
@login_required
def upload_schedules():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"ok": False, "error": "Please upload a CSV file."}), 400
    try:
        stream = file.stream.read().decode("utf-8").splitlines()
        reader = csv.DictReader(stream, fieldnames=["course_code", "days", "time_window", "room"])
        rows = [r for r in reader if r["course_code"].strip() and r["course_code"].strip().lower() != "coursecode"]
        inserted, skipped, errors = insert_schedules(rows)
        return jsonify({"ok": True, "inserted": inserted, "skipped": skipped, "errors": errors[:5]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/upload/enrollments", methods=["POST"])
@login_required
def upload_enrollments():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"ok": False, "error": "Please upload a CSV file."}), 400
    try:
        stream = file.stream.read().decode("utf-8").splitlines()
        reader = csv.DictReader(stream, fieldnames=["student_id", "course_code"])
        rows = [r for r in reader if r["student_id"].strip() and r["student_id"].strip().lower() != "studentid"]
        inserted, skipped, errors = insert_enrollments(rows)
        return jsonify({"ok": True, "inserted": inserted, "skipped": skipped, "errors": errors[:5]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/courses")
@login_required
def api_courses():
    return jsonify(get_all_courses())


@app.route("/api/courses/add", methods=["POST"])
@login_required
def api_add_course():
    data = request.get_json(force=True)
    course_code = data.get("course_code", "").strip()
    days        = data.get("days", "").strip().upper()
    time_window = data.get("time_window", "").strip()
    room        = data.get("room", "").strip()
    if not course_code or not days or not time_window or not room:
        return jsonify({"ok": False, "error": "All fields are required."}), 400
    ok, err = add_course(course_code, days, time_window, room)
    if ok:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": err}), 400


@app.route("/api/courses/<course_code>", methods=["DELETE"])
@login_required
def api_delete_course(course_code):
    ok = delete_course(course_code)
    return jsonify({"ok": ok})


@app.route("/api/students/<student_id>/enrollments")
@login_required
def api_student_enrollments(student_id):
    return jsonify(get_student_enrollments(student_id))


@app.route("/api/students/<student_id>/enroll", methods=["POST"])
@login_required
def api_enroll_student(student_id):
    data = request.get_json(force=True)
    course_code = data.get("course_code", "").strip()
    if not course_code:
        return jsonify({"ok": False, "error": "course_code required"}), 400
    ok, err = enroll_student(student_id, course_code)
    if ok:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": err}), 400


@app.route("/api/students/<student_id>/enroll/<course_code>", methods=["DELETE"])
@login_required
def api_unenroll_student(student_id, course_code):
    ok = unenroll_student(student_id, course_code)
    return jsonify({"ok": ok})


@app.route("/api/students/<student_id>/delete", methods=["DELETE"])
@login_required
def api_delete_student(student_id):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({"ok": False, "error": "Student not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


@app.route("/api/students/add", methods=["POST"])
@login_required
def api_add_student():
    data = request.get_json(force=True)
    student_id = data.get("student_id", "").strip()
    last_name  = data.get("last_name", "").strip()
    first_name = data.get("first_name", "").strip()

    if not student_id or not last_name or not first_name:
        return jsonify({"ok": False, "error": "All fields are required."}), 400

    inserted, skipped, errors = insert_students([{
        "student_id": student_id,
        "last_name": last_name,
        "first_name": first_name
    }])
    if errors:
        return jsonify({"ok": False, "error": errors[0]}), 400
    if skipped:
        return jsonify({"ok": False, "error": "Student ID already exists."}), 409
    return jsonify({"ok": True})


@app.route("/students/export")
@login_required
def export_students():
    import io
    all_students = get_all_students()
    output = io.StringIO()
    writer = csv.writer(output)
    for s in all_students:
        writer.writerow([s["student_id"], s["last_name"], s["first_name"]])
    csv_bytes = output.getvalue().encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=students.csv"}
    )


@app.route("/students")
@login_required
def students():
    return render_template("students.html",
                           students=get_all_students(),
                           courses=get_all_courses(),
                           username=session.get('username', ''))


@app.route("/students/<student_id>/capture")
@login_required
def student_capture(student_id):
    all_students = get_all_students()
    student = next((s for s in all_students if s["student_id"] == student_id), None)
    if not student:
        return "Student not found", 404
    return render_template("student_capture.html", student=student, username=session.get('username', ''))


@app.route("/api/students/<student_id>/save-face", methods=["POST"])
@login_required
def api_save_student_face(student_id):
    data = request.get_json(force=True)
    image_data = data.get("image", "")

    if not image_data:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    try:
        from deepface import DeepFace
        _, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        ok = save_student_face_encoding(student_id, result[0]["embedding"])
        if ok:
            os.makedirs(PHOTO_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(PHOTO_DIR, f"{student_id}.jpg"), frame)
            save_student_photo_path(student_id, f"student_photos/{student_id}.jpg")
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Student not found in database"}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/students/<student_id>/upload-photo", methods=["POST"])
@login_required
def api_upload_student_photo(student_id):
    """Upload a JPG/PNG photo, extract face encoding, save both to disk and DB."""
    file = request.files.get("photo")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No file provided"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return jsonify({"ok": False, "error": "Only JPG or PNG files are allowed"}), 400

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode image file"}), 400

    os.makedirs(PHOTO_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(PHOTO_DIR, f"{student_id}.jpg"), frame)
    save_student_photo_path(student_id, f"student_photos/{student_id}.jpg")

    try:
        from deepface import DeepFace
        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        save_student_face_encoding(student_id, result[0]["embedding"])
        return jsonify({"ok": True, "encoding_saved": True})
    except Exception as e:
        return jsonify({
            "ok": True,
            "encoding_saved": False,
            "warning": f"Photo saved but no face was detected — try a clearer, front-facing photo. ({e})"
        })


@app.route("/health")
def health():
    with camera_workers_lock:
        cameras = {
            camera_id: {
                "name": worker.name,
                "source": worker.source,
                "presence": worker.last_presence,
                "status": worker.last_status,
            }
            for camera_id, worker in camera_workers.items()
        }
    return jsonify({"ok": True, "cameras": cameras})


if __name__ == "__main__":
    ensure_log_files()
    init_db()
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
