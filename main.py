import asyncio
import io
import os
import random
import socket
import threading
import time
import urllib.request
from contextlib import asynccontextmanager

import cv2
import mediapipe as mp
import numpy as np
import qrcode
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

SESSION_DURATION = 10
THUMBS_HOLD_SECONDS = 2.0

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

state = {
    "active": False,
    "end_time": 0.0,
    "mode": "gesture",  # "gesture" | "demo"
}

ws_clients: list[WebSocket] = []
frame_lock = threading.Lock()
current_frame: np.ndarray | None = None
stop_event = threading.Event()
thumbs_hold_started_at: float | None = None

model: YOLO | None = None
main_loop: asyncio.AbstractEventLoop | None = None


def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand landmark model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("Hand model ready.")


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_thumbs_up(landmarks) -> bool:
    thumb_tip = landmarks[4]
    wrist     = landmarks[0]
    # thumb must point upward past the wrist
    if thumb_tip.y >= wrist.y:
        return False
    # thumb tip must be higher than every other fingertip
    return all(thumb_tip.y < landmarks[i].y for i in [8, 12, 16, 20])


_conf_value: float = 97.5
_conf_last_update: float = 0.0


def draw_yolo_frame(frame: np.ndarray, persons: list) -> np.ndarray:
    global _conf_value, _conf_last_update
    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if not persons:
        text = "Scanning for generous humans..."
        (tw, _), _ = cv2.getTextSize(text, font, 1.0, 2)
        cv2.putText(out, text, (w // 2 - tw // 2, h - 30),
                    font, 1.0, (160, 220, 210), 2, cv2.LINE_AA)
        return out

    # update confidence at most once per second so it doesn't flicker
    now = time.monotonic()
    if now - _conf_last_update >= 1.0:
        _conf_value = round(random.uniform(96.0, 99.9), 1)
        _conf_last_update = now

    for box in persons:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), (38, 204, 188), 3)

        label = f"Likely to donate to CAIR: {_conf_value}%"
        fs, th = 0.72, 2

        (lw, lh), _ = cv2.getTextSize(label, font, fs, th)
        bg_w = lw + 20
        bg_h = lh + 16
        by1 = max(0, y1 - bg_h - 6)

        cv2.rectangle(out, (x1, by1), (x1 + bg_w, y1 - 4), (255, 84, 143), -1)
        cv2.putText(out, label, (x1 + 10, by1 + lh + 6),
                    font, fs, (0, 0, 0), th, cv2.LINE_AA)

    return out


def camera_worker():
    global current_frame, thumbs_hold_started_at

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    t_start = time.monotonic()

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if state["mode"] == "gesture":
                # keep current_frame fresh so demo has something to show immediately
                with frame_lock:
                    current_frame = frame

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.monotonic() - t_start) * 1000)
                result = landmarker.detect_for_video(mp_img, ts_ms)

                detected = False
                if result.hand_landmarks:
                    for hand_lm in result.hand_landmarks:
                        if is_thumbs_up(hand_lm):
                            detected = True
                            break

                if detected:
                    now = time.monotonic()
                    if thumbs_hold_started_at is None:
                        thumbs_hold_started_at = now
                    progress = min((now - thumbs_hold_started_at) / THUMBS_HOLD_SECONDS, 1.0)
                    if main_loop:
                        asyncio.run_coroutine_threadsafe(
                            broadcast({"type": "thumbs_progress", "value": progress}),
                            main_loop,
                        )
                    if progress >= 1.0 and main_loop:
                        thumbs_hold_started_at = None
                        asyncio.run_coroutine_threadsafe(trigger_demo(), main_loop)
                else:
                    if thumbs_hold_started_at is not None:
                        thumbs_hold_started_at = None
                        if main_loop:
                            asyncio.run_coroutine_threadsafe(
                                broadcast({"type": "thumbs_progress", "value": 0}),
                                main_loop,
                            )

            else:  # demo mode
                thumbs_hold_started_at = None
                results = model(frame, classes=[0], verbose=False, imgsz=640)
                persons = [box for r in results for box in r.boxes]
                annotated = draw_yolo_frame(frame, persons)
                with frame_lock:
                    current_frame = annotated

    cap.release()
    with frame_lock:
        current_frame = None


async def trigger_demo() -> bool:
    if state["active"]:
        return False
    state["active"] = True
    state["mode"] = "demo"
    state["end_time"] = time.time() + SESSION_DURATION
    await broadcast({
        "type": "demo_started",
        "duration": SESSION_DURATION,
        "remaining": SESSION_DURATION,
    })
    return True


async def broadcast(msg: dict):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        if d in ws_clients:
            ws_clients.remove(d)


async def timeout_loop():
    while True:
        await asyncio.sleep(1)
        if not state["active"]:
            continue
        remaining = max(0, int(state["end_time"] - time.time()))
        await broadcast({"type": "tick", "remaining": remaining})
        if remaining == 0:
            state["active"] = False
            state["mode"] = "gesture"
            await broadcast({"type": "idle"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, main_loop
    main_loop = asyncio.get_running_loop()
    ensure_hand_model()
    model = YOLO("yolov8n.pt")
    t = threading.Thread(target=camera_worker, daemon=True)
    t.start()
    asyncio.create_task(timeout_loop())
    yield
    stop_event.set()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def tv_page():
    with open("static/tv.html") as f:
        return HTMLResponse(f.read())


@app.get("/mobile")
async def mobile_page():
    with open("static/mobile.html") as f:
        return HTMLResponse(f.read())


@app.get("/qr")
async def qr_endpoint(request: Request):
    public = os.environ.get("PUBLIC_URL", "").rstrip("/")
    if public:
        url = f"{public}/mobile"
    else:
        ip = get_local_ip()
        port = request.url.port or 8000
        url = f"http://{ip}:{port}/mobile"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# manual override — still works if needed
@app.post("/start")
async def start_demo():
    started = await trigger_demo()
    if started:
        return JSONResponse({"status": "started", "duration": SESSION_DURATION})
    remaining = max(0, int(state["end_time"] - time.time()))
    return JSONResponse({"status": "already_active", "remaining": remaining})


@app.get("/video_feed")
async def video_feed():
    async def generate():
        while state["active"]:
            with frame_lock:
                frame = current_frame
            if frame is None:
                await asyncio.sleep(0.03)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)

    if state["active"]:
        remaining = max(0, int(state["end_time"] - time.time()))
        await ws.send_json({
            "type": "demo_started",
            "duration": SESSION_DURATION,
            "remaining": remaining,
        })
    else:
        await ws.send_json({"type": "idle"})

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
