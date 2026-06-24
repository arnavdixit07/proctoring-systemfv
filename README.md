# Proctoring System — Flask Edition

## What changed from the original

The CV logic is identical. What's new:

- **`server.py`** — Flask + SocketIO backend. Each testee gets an isolated `TesteeSession` with its own MediaPipe detector and state machine.
- **Testee UI** (`/testee?name=...`) — runs in a browser, captures camera via WebRTC, sends JPEG frames to the server at ~7 fps. The server runs detection and pushes status back.
- **Tester dashboard** (`/tester`) — password-protected. Shows a live video grid of all connected testees, real-time flag counts, and an alert sidebar.

---

## Setup

```bash
pip install -r requirements.txt
```

The `face_landmarker.task` model file is downloaded automatically on first run (same as the original script).

---

## Running

```bash
# Default tester password is proctor123
python server.py

# Or set a custom password:
TESTER_PASSWORD=mysecret python server.py
```

Server starts on `http://0.0.0.0:5000`.

---

## Access

| Role    | URL                                  |
|---------|--------------------------------------|
| Testee  | `http://<your-ip>:5000/`             |
| Tester  | `http://<your-ip>:5000/tester`       |

On macOS, find your LAN IP with: `ipconfig getifaddr en0`

Testees on the same network open `http://192.168.x.x:5000`, enter their name, and begin.

---

## How sessions work

1. Testee opens the page, enters name → joins socket room `testee_<sid>`
2. Browser captures camera at 640×480, sends JPEG frames every 150ms
3. Server decodes each frame, runs `FaceLandmarker`, applies the same phase state machine (room_check → calib_prompt → calib_hold × 3 → monitoring)
4. All events (phase changes, flag events, alert cleared) are pushed back to the testee UI via SocketIO
5. Every frame is also forwarded (compressed to 50% JPEG) to the `testers` room so the dashboard shows live feeds
6. Alert events are separately emitted to testers with name, direction, count, and timestamp
7. Per-session logs are written to `logs/<name>_<timestamp>.txt`

---

## Notes

- **HTTPS for production:** browsers require HTTPS to grant camera access on non-localhost origins. For a real deployment, put the server behind nginx with an SSL cert, or use a tunnel like `ngrok http 5000`.
- **Scale:** each testee runs a full MediaPipe detector in a Python thread. For large groups (50+), consider running detection in a process pool or on a GPU-enabled machine.
- **The original `proctor.py`** can still be run standalone — nothing in the original file was modified.
