"""
Proctoring server — Flask + SocketIO
All CV logic runs server-side per session.
"""

import os, time, base64, threading, urllib.request
from datetime import datetime
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, abort
from flask_socketio import SocketIO, emit, join_room, leave_room

# ── Model bootstrap ────────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task …")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH,
    )
    print("Done.")

# Object detector (COCO classes — used for "cell phone" + backup "person" signal)
OBJECT_MODEL_PATH = "efficientdet_lite0.tflite"
if not os.path.exists(OBJECT_MODEL_PATH):
    print("Downloading efficientdet_lite0.tflite …")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
        OBJECT_MODEL_PATH,
    )
    print("Done.")

def _make_detector():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=5,
        # Raised from 0.2 — low confidence let in phantom "second faces" from
        # shadows/edges/background clutter, causing random multi_face alerts.
        min_face_detection_confidence=0.6,
    )
    return vision.FaceLandmarker.create_from_options(opts)

def _make_object_detector():
    opts = vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=OBJECT_MODEL_PATH),
        max_results=8,
        score_threshold=0.35,
        category_allowlist=["cell phone", "person"],
    )
    return vision.ObjectDetector.create_from_options(opts)

# ── Constants ──────────────────────────────────────────────────────────────────
NOSE_TIP, LEFT_EYE, RIGHT_EYE = 1, 33, 263
DEVIATION_THRESHOLD = 0.4
SUSTAINED_SECS      = 1.5
AUTOFRAME_WINDOW    = 0.8
AUTOFRAME_MAX_DRIFT = 0.06
CALIB_MIN_TURN      = 0.25
CLEAR_SECS          = 0.5
MULTI_FACE_STREAK   = 2     # consecutive frames needed before trusting a 2nd face
OBJ_DETECT_EVERY_N  = 2     # run object detector every Nth frame (cheaper)

SCREENSHOTS_DIR      = "screenshots"   # one subfolder per sid, served via /screenshots/<sid>/<file>
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

CALIB_STEPS = [
    ("Look LEFT and hold",           4.0, "left",  "Please try again — turn head further LEFT."),
    ("Look RIGHT and hold",          4.0, "right", "Please try again — turn head further RIGHT."),
    ("Move BACK slightly and hold",  2.0, "back",  "Please try again — move back slightly."),
]

# ── Per-session state ──────────────────────────────────────────────────────────
class TesteeSession:
    """All mutable state for one testee."""

    def __init__(self, sid, name):
        self.sid            = sid
        self.name           = name
        self.detector       = _make_detector()
        self.obj_detector   = _make_object_detector()
        self.lock           = threading.Lock()

        # phase: "room_check" | "calib_prompt" | "calib_hold" | "monitoring"
        self.phase          = "room_check"
        self.calib_step     = 0           # 0-2
        self.calib_retry    = False

        # calibration per-step
        self.center_nose    = None
        self.center_fw      = None
        self.hold_start     = None
        self.final_nose     = None

        # monitoring
        self.baseline_nose  = None
        self.af_samples     = []
        self.autoframe_warn = False
        self.autoframe_time = 0.0
        self.movement_count = 0
        self.alert_start    = None
        self.flagged        = False
        self.clear_start    = None
        self.deviation      = 0.0

        # multi-face debouncing
        self.multi_face_streak = 0

        # object detection (phone / extra person)
        self.frame_idx       = 0
        self.last_obj_result = None
        self.phone_alert_cd  = 0.0   # cooldown timestamp
        self.person_alert_cd = 0.0

        # last raw frame (JPEG bytes) — kept so non-frame-triggered events
        # (e.g. a tester's manual flag) can still attach a screenshot
        self.last_frame_jpeg = None
        self.shot_count      = 0
        self.shot_dir        = os.path.join(SCREENSHOTS_DIR, sid)
        os.makedirs(self.shot_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("logs", exist_ok=True)
        self.log = open(f"logs/{name}_{ts}.txt", "w")
        self.log.write(f"Session: {name}  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("-" * 45 + "\n")

    def close(self):
        try:
            self.log.write("-" * 45 + "\n")
            self.log.write(f"Session ended {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log.write(f"Total movements: {self.movement_count}\n")
            self.log.close()
            self.detector.close()
            self.obj_detector.close()
        except Exception:
            pass


# Global session store  {sid: TesteeSession}
sessions: dict[str, TesteeSession] = {}
sessions_lock = threading.Lock()

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", max_http_buffer_size=5 * 1024 * 1024)

TESTER_PASSWORD = os.environ.get("TESTER_PASSWORD", "proctor123")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/testee")
def testee_page():
    name = request.args.get("name", "").strip()
    if not name:
        return redirect("/?error=name")
    return render_template("testee.html", name=name)

@app.route("/tester", methods=["GET", "POST"])
def tester_page():
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if pwd == TESTER_PASSWORD:
            session["tester"] = True
            return redirect(url_for("tester_page"))
        return render_template("tester_login.html", error=True)
    if not session.get("tester"):
        return render_template("tester_login.html", error=False)
    return render_template("tester.html")

@app.route("/screenshots/<sid>/<filename>")
def get_screenshot(sid, filename):
    """Serve a saved alert screenshot. Restricted to logged-in testers since
    these frames are sensitive (testee's camera feed)."""
    if not session.get("tester"):
        abort(403)
    return send_from_directory(os.path.join(SCREENSHOTS_DIR, sid), filename)

# ── Socket events — testee ─────────────────────────────────────────────────────
@socketio.on("testee_join")
def on_testee_join(data):
    sid  = request.sid
    name = data.get("name", "Unknown")
    with sessions_lock:
        if sid in sessions:
            sessions[sid].close()
        sessions[sid] = TesteeSession(sid, name)
    join_room(f"testee_{sid}")
    # Notify all testers
    socketio.emit("testee_connected", {"sid": sid, "name": name}, to="testers")
    emit("phase", {"phase": "room_check"})


@socketio.on("testee_space")
def on_testee_space(_):
    """Testee pressed SPACE."""
    sid = request.sid
    with sessions_lock:
        s = sessions.get(sid)
    if not s:
        return
    with s.lock:
        if s.phase == "room_check":
            s.phase = "calib_prompt"
            s.calib_step = 0
            s.calib_retry = False
            _emit_calib_prompt(s)
        elif s.phase == "calib_prompt":
            # Snapshot current nose as center reference — taken from last frame
            s.center_nose = s._last_nose if hasattr(s, "_last_nose") else None
            s.center_fw   = s._last_fw   if hasattr(s, "_last_fw")   else None
            s.hold_start  = None
            s.final_nose  = None
            s.phase = "calib_hold"
            socketio.emit("phase", {"phase": "calib_hold",
                           "instruction": CALIB_STEPS[s.calib_step][0],
                           "hold_time": CALIB_STEPS[s.calib_step][1],
                           "step": s.calib_step}, to=f"testee_{sid}")


@socketio.on("frame")
def on_frame(data):
    """Receive a JPEG frame as base64 from the testee."""
    sid = request.sid
    with sessions_lock:
        s = sessions.get(sid)
    if not s:
        return

    # Decode frame
    try:
        b64 = data.get("image", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception:
        return

    # Run face detection
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = s.detector.detect(mp_img)

    nose_x, face_width = None, None
    if result.face_landmarks:
        lm         = result.face_landmarks[0]
        nose_x     = lm[NOSE_TIP].x
        face_width = abs(lm[RIGHT_EYE].x - lm[LEFT_EYE].x)

    # Run object detection every Nth frame only (phone / extra person) — it's
    # heavier than face landmarking and doesn't need to run on every single frame
    # for an "instant" alert to still feel instant.
    obj_result = None
    with s.lock:
        s.frame_idx += 1
        run_obj = (s.frame_idx % OBJ_DETECT_EVERY_N == 0)
    if run_obj:
        obj_result = s.obj_detector.detect(mp_img)

    # Encode once — reused both for the live tester preview stream and as the
    # screenshot saved to disk if this frame ends up triggering an alert.
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_jpeg_bytes = buf.tobytes()

    with s.lock:
        # Cache latest nose for SPACE snapshot
        if nose_x is not None:
            s._last_nose = nose_x
            s._last_fw   = face_width

        if obj_result is not None:
            s.last_obj_result = obj_result

        # Cache latest frame so alert handlers (including a tester's manual
        # flag, which doesn't run inside this function) can screenshot it.
        s.last_frame_jpeg = frame_jpeg_bytes

        if s.phase == "calib_hold":
            _process_calib_frame(s, nose_x, face_width)
        elif s.phase == "monitoring":
            _process_monitor_frame(s, result, nose_x, face_width, sid)

    # Forward frame to tester room (re-encode at lower quality for the live
    # preview to keep bandwidth down; the saved screenshot above keeps the
    # higher-quality bytes for evidentiary purposes).
    _, preview_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    b64out = base64.b64encode(preview_buf).decode()
    socketio.emit("testee_frame",
                  {"sid": sid, "image": "data:image/jpeg;base64," + b64out,
                   "movement_count": s.movement_count,
                   "flagged": s.flagged,
                   "phase": s.phase},
                  to="testers")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    with sessions_lock:
        s = sessions.pop(sid, None)
    if s:
        s.close()
        socketio.emit("testee_disconnected", {"sid": sid}, to="testers")


# ── Socket events — tester ─────────────────────────────────────────────────────
@socketio.on("tester_join")
def on_tester_join(_):
    join_room("testers")
    with sessions_lock:
        active = [{"sid": sid, "name": s.name, "movement_count": s.movement_count,
                   "phase": s.phase, "flagged": s.flagged}
                  for sid, s in sessions.items()]
    emit("active_testees", {"testees": active})


@socketio.on("manual_flag")
def on_manual_flag(data):
    """A tester manually flags a testee from the dashboard. Always logs and
    raises an alert even if the testee was already flagged for something
    else, so manual review intent is never silently dropped."""
    if not session.get("tester"):
        return
    target_sid = (data or {}).get("sid")
    with sessions_lock:
        s = sessions.get(target_sid)
    if not s:
        return

    with s.lock:
        s.flagged         = True
        s.alert_start      = s.alert_start or time.time()
        s.clear_start       = None
        s.movement_count   += 1
        ts = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{ts}]  Movement #{s.movement_count:<3}  MANUAL FLAG (by proctor)\n"
        s.log.write(log_line)
        s.log.flush()
        shot = _save_screenshot(s, "manual")

    socketio.emit("alert", {
        "sid": target_sid, "name": s.name,
        "movement_count": s.movement_count,
        "alert_type": "manual",
        "ts": ts, "screenshot": shot,
    }, to="testers")


@socketio.on("manual_unflag")
def on_manual_unflag(data):
    """Clears a flag the tester raised (or any active flag) by hand."""
    if not session.get("tester"):
        return
    target_sid = (data or {}).get("sid")
    with sessions_lock:
        s = sessions.get(target_sid)
    if not s:
        return
    with s.lock:
        s.flagged     = False
        s.alert_start = None
        s.clear_start = None
    socketio.emit("alert_cleared", {"sid": target_sid}, to="testers")


# ── Calibration helpers ────────────────────────────────────────────────────────
def _emit_calib_prompt(s: TesteeSession):
    step = s.calib_step
    _, _, _, retry_msg = CALIB_STEPS[step]
    socketio.emit("phase", {
        "phase":     "calib_prompt",
        "prompt":    f"Press SPACE when ready to {CALIB_STEPS[step][0].split(' and ')[0]}.",
        "retry_msg": retry_msg if s.calib_retry else "",
        "step":      step,
    }, to=f"testee_{s.sid}")


def _process_calib_frame(s: TesteeSession, nose_x, face_width):
    now = time.time()
    _, hold_time, tag, _ = CALIB_STEPS[s.calib_step]

    if nose_x is not None:
        if s.hold_start is None:
            s.hold_start = now
        elapsed      = now - s.hold_start
        s.final_nose = nose_x
    else:
        s.hold_start = None
        elapsed      = 0.0

    frac = min(elapsed / hold_time, 1.0) if hold_time else 0
    socketio.emit("calib_progress", {"frac": frac, "elapsed": elapsed,
                                     "hold_time": hold_time, "no_face": nose_x is None},
                  to=f"testee_{s.sid}")

    if nose_x is not None and elapsed >= hold_time:
        # Validate
        valid = False
        if s.center_nose is not None and s.final_nose is not None and s.center_fw:
            disp  = abs(s.final_nose - s.center_nose) / s.center_fw
            valid = disp >= CALIB_MIN_TURN or tag == "back"
        else:
            valid = True  # fallback if snapshot missed

        if valid:
            s.calib_step += 1
            if s.calib_step >= len(CALIB_STEPS):
                # Calibration done → set baseline and switch to monitoring
                s.baseline_nose = nose_x
                s.phase         = "monitoring"
                socketio.emit("phase", {"phase": "monitoring"}, to=f"testee_{s.sid}")
                socketio.emit("testee_status", {"sid": s.sid, "phase": "monitoring"},
                              to="testers")
            else:
                s.calib_retry = False
                s.hold_start  = None
                s.phase       = "calib_prompt"
                _emit_calib_prompt(s)
        else:
            s.calib_retry = True
            s.hold_start  = None
            s.phase       = "calib_prompt"
            _emit_calib_prompt(s)


def _save_screenshot(s: TesteeSession, alert_type: str) -> str | None:
    """Persist the testee's most recent frame to disk, tagged with the alert
    type, and return a URL path the tester dashboard can load directly.
    Returns None if no frame has been received yet (e.g. alert fired before
    any frame arrived — shouldn't normally happen, but stay defensive)."""
    if not s.last_frame_jpeg:
        return None
    s.shot_count += 1
    ts_tag   = datetime.now().strftime("%H%M%S")
    fname    = f"{s.shot_count:04d}_{alert_type}_{ts_tag}.jpg"
    fpath    = os.path.join(s.shot_dir, fname)
    try:
        with open(fpath, "wb") as f:
            f.write(s.last_frame_jpeg)
    except Exception:
        return None
    return f"/screenshots/{s.sid}/{fname}"


def _check_object_alerts(s: TesteeSession, sid):
    """Instant alerts for phone / extra-person detections from the object model.
    Cooldown prevents the same held-up object from spamming an alert every frame.
    Returns True if a phone is visible in the current object-detection result —
    used by the caller to suppress a simultaneous head-turn alert, since raising
    a phone toward the camera naturally shifts head position too, and the phone
    is the more specific/important signal to show."""
    obj_result = s.last_obj_result
    if obj_result is None or not obj_result.detections:
        return False

    now = time.time()
    phone_seen  = False
    phone_score = 0.0
    person_seen = False
    for det in obj_result.detections:
        if not det.categories:
            continue
        cat   = det.categories[0].category_name
        score = det.categories[0].score
        if cat == "cell phone":
            phone_seen  = True
            phone_score = max(phone_score, score)
        elif cat == "person":
            person_seen = True

    if phone_seen and (now - s.phone_alert_cd) >= 4.0:
        s.phone_alert_cd  = now
        s.movement_count += 1
        ts       = datetime.now().strftime("%H:%M:%S")
        log_line = (f"[{ts}]  Movement #{s.movement_count:<3}  "
                     f"OBJECT: cell phone  (conf {phone_score:.2f})\n")
        s.log.write(log_line)
        s.log.flush()
        shot = _save_screenshot(s, "phone")
        socketio.emit("alert", {
            "sid": sid, "name": s.name,
            "movement_count": s.movement_count,
            "alert_type": "phone",
            "object_label": "cell phone", "object_score": round(phone_score, 2),
            "ts": ts, "screenshot": shot,
        }, to="testers")

    # "person" count of 2+ in COCO detections means someone besides the testee
    # is in frame — use it as a backup to face-count multi-face detection,
    # since a second person's face isn't always clearly visible/frontal.
    person_count = sum(1 for det in obj_result.detections
                       if det.categories and det.categories[0].category_name == "person")
    if person_count >= 2 and (now - s.person_alert_cd) >= 4.0:
        s.person_alert_cd  = now
        s.movement_count  += 1
        ts       = datetime.now().strftime("%H:%M:%S")
        log_line = (f"[{ts}]  Movement #{s.movement_count:<3}  "
                     f"OBJECT: extra person  (count {person_count})\n")
        s.log.write(log_line)
        s.log.flush()
        shot = _save_screenshot(s, "extra_person")
        socketio.emit("alert", {
            "sid": sid, "name": s.name,
            "movement_count": s.movement_count,
            "alert_type": "extra_person",
            "object_label": "person", "object_score": None,
            "ts": ts, "screenshot": shot,
        }, to="testers")


def _process_monitor_frame(s: TesteeSession, result, nose_x, face_width, sid):
    now           = time.time()
    raw_face_count = len(result.face_landmarks)

    # Debounce multi-face: require it to persist for MULTI_FACE_STREAK
    # consecutive frames before treating it as real. A single stray frame
    # (motion blur, shadow misread as a face) no longer triggers an alert.
    if raw_face_count > 1:
        s.multi_face_streak += 1
    else:
        s.multi_face_streak = 0
    multi_face_confirmed = s.multi_face_streak >= MULTI_FACE_STREAK

    raw_alert  = multi_face_confirmed
    direction  = ""
    deviation  = 0.0

    if nose_x is not None and face_width:
        if s.baseline_nose is None:
            s.baseline_nose = nose_x
        raw_dev   = nose_x - s.baseline_nose
        deviation = abs(raw_dev) / face_width
        s.deviation = deviation

        if deviation > DEVIATION_THRESHOLD:
            raw_alert = True
            direction = "LEFT" if raw_dev > 0 else "RIGHT"
            s.af_samples.append((now, nose_x))
            s.af_samples = [(t, x) for t, x in s.af_samples if now - t <= AUTOFRAME_WINDOW]
            if len(s.af_samples) >= 2:
                dur   = s.af_samples[-1][0] - s.af_samples[0][0]
                rng   = max(x for _, x in s.af_samples) - min(x for _, x in s.af_samples)
                if dur >= AUTOFRAME_WINDOW and rng < AUTOFRAME_MAX_DRIFT * face_width:
                    s.autoframe_warn = True
                    s.autoframe_time = now
        else:
            s.af_samples = []

    # Object-based instant alerts (phone / extra person) — independent of the
    # sustained-alert gate below, since these should fire immediately.
    _check_object_alerts(s, sid)

    # Sticky gate
    if raw_alert:
        s.clear_start = None
        if s.alert_start is None:
            s.alert_start = now
        sustained = now - s.alert_start
        if sustained >= SUSTAINED_SECS and not s.flagged:
            s.flagged         = True
            s.movement_count += 1
            ts = datetime.now().strftime("%H:%M:%S")

            # One unambiguous reason per alert — multi_face takes priority
            # since it's the most serious signal, then autoframe, then head-turn.
            if multi_face_confirmed:
                alert_type = "multi_face"
                log_desc   = "MULTIPLE FACES"
            elif s.autoframe_warn:
                alert_type = "autoframe"
                log_desc   = "AUTOFRAME SUSPECTED"
            else:
                alert_type = "head_turn"
                log_desc   = f"{direction}  ({sustained:.1f}s)"

            log_line = f"[{ts}]  Movement #{s.movement_count:<3}  {log_desc}\n"
            s.log.write(log_line)
            s.log.flush()
            shot = _save_screenshot(s, alert_type)
            # Alert tester
            socketio.emit("alert", {
                "sid": sid, "name": s.name,
                "movement_count": s.movement_count,
                "alert_type": alert_type,
                "direction": direction,
                "sustained": round(sustained, 1),
                "ts": ts, "screenshot": shot,
            }, to="testers")
    else:
        if s.flagged:
            if s.clear_start is None:
                s.clear_start = now
            if now - s.clear_start >= CLEAR_SECS:
                s.alert_start = None
                s.flagged     = False
                s.clear_start = None
                socketio.emit("alert_cleared", {"sid": sid}, to="testers")
        else:
            s.alert_start = None

    if s.autoframe_warn and (now - s.autoframe_time) >= 5.0:
        s.autoframe_warn = False

    # Send status back to testee UI
    socketio.emit("monitor_status", {
        "deviation":      round(deviation, 3),
        "flagged":        s.flagged,
        "movement_count": s.movement_count,
        "multi_face":     multi_face_confirmed,
        "autoframe_warn": s.autoframe_warn,
        "alert_start":    s.alert_start,
        "now":            now,
        "sustained_secs": SUSTAINED_SECS,
        "direction":      direction,
    }, to=f"testee_{sid}")


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Proctoring server starting on http://0.0.0.0:5002")
    print(f"Tester password: {TESTER_PASSWORD}")
    socketio.run(app, host="0.0.0.0", port=5002, debug=False)