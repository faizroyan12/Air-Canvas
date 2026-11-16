import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------- Konfigurasi
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
WINDOW_NAME = "Air Canvas"

SMOOTHING = 0.5          # 0..1, makin kecil makin halus tapi makin "lambat"
TOOLBAR_H = 80           # tinggi toolbar warna (piksel)
UNDO_LIMIT = 15          # jumlah langkah undo
MIN_THICKNESS, MAX_THICKNESS = 2, 40
ERASER_THICKNESS = 45

# Warna dalam format BGR
COLORS = [
    ("Kuning", (0, 255, 255)),
    ("Merah", (0, 0, 255)),
    ("Hijau", (0, 220, 0)),
    ("Biru", (255, 120, 0)),
    ("Putih", (255, 255, 255)),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Pasangan (ujung jari, sendi PIP) untuk telunjuk, tengah, manis, kelingking
FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}


# ---------------------------------------------------------------- Fungsi bantu
def open_camera():
    """Coba kamera 1 dulu (kamera eksternal), lalu 0. DSHOW mempercepat di Windows."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    for index in (1, 0):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None


def create_detector():
    if not MODEL_PATH.exists():
        sys.exit(f"File model tidak ditemukan: {MODEL_PATH}")
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,  # tracking antar frame, lebih cepat
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def finger_states(pts):
    """True jika jari terangkat. Pakai jarak ke pergelangan tangan, jadi
    tetap benar walau tangan miring atau terbalik."""
    wrist = pts[0]
    return {
        name: math.dist(pts[tip], wrist) > math.dist(pts[pip], wrist)
        for name, (tip, pip) in FINGERS.items()
    }


def get_mode(up):
    idx, mid, ring, pinky = up["index"], up["middle"], up["ring"], up["pinky"]
    if idx and not mid and not ring and not pinky:
        return "draw"
    if idx and mid and not ring and not pinky:
        return "nav"
    if idx and mid and ring and not pinky:
        return "erase"
    return "idle"


def toolbar_boxes(w):
    size, gap = 50, 18
    total = len(COLORS) * size + (len(COLORS) - 1) * gap
    x0 = (w - total) // 2
    y0 = (TOOLBAR_H - size) // 2
    boxes = []
    for i, (_, color) in enumerate(COLORS):
        x = x0 + i * (size + gap)
        boxes.append((x, y0, x + size, y0 + size, color))
    return boxes


def draw_toolbar(img, boxes, selected):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], TOOLBAR_H), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (x1, y1, x2, y2, color) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        if i == selected:
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 2)


def composite(frame, canvas):
    """Tempelkan coretan di atas video."""
    out = frame.copy()
    mask = canvas.any(axis=2)
    out[mask] = canvas[mask]
    return out


def save_canvas(canvas):
    """Simpan coretan sebagai PNG transparan."""
    alpha = (canvas.any(axis=2) * 255).astype(np.uint8)
    bgra = np.dstack([canvas, alpha])
    name = time.strftime("air_canvas_%Y%m%d_%H%M%S.png")
    path = Path(__file__).parent / name
    cv2.imwrite(str(path), bgra)
    return path.name


# ---------------------------------------------------------------- Program utama
def main():
    detector = create_detector()
    cap = open_camera()
    if cap is None:
        detector.close()
        sys.exit("Kamera tidak ditemukan.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = True
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    canvas = None
    undo_stack = deque(maxlen=UNDO_LIMIT)
    color_index = 0
    thickness = 7
    smooth = None          # posisi telunjuk yang sudah dihaluskan
    prev_point = None      # titik terakhir dalam goresan yang sedang berjalan
    prev_mode = "idle"
    last_ts = 0
    notice, notice_until = "", 0.0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if canvas is None or canvas.shape[:2] != (h, w):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                undo_stack.clear()
            boxes = toolbar_boxes(w)

            # --- Deteksi tangan (timestamp harus selalu naik)
            ts = int(time.monotonic() * 1000)
            ts = max(ts, last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, ts)

            mode = "none"
            mode_text = "Mencari Tangan..."

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                pts = [(lm.x * w, lm.y * h) for lm in lms]

                # Kerangka tangan
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                             (int(pts[b][0]), int(pts[b][1])), (0, 0, 255), 2)
                for px, py in pts:
                    cv2.circle(frame, (int(px), int(py)), 5, (0, 102, 255), cv2.FILLED)

                # Posisi telunjuk yang dihaluskan (EMA) untuk mengurangi getaran
                raw = np.array(pts[8])
                smooth = raw if smooth is None else SMOOTHING * raw + (1 - SMOOTHING) * smooth
                x, y = int(smooth[0]), int(smooth[1])

                mode = get_mode(finger_states(pts))

                # Goresan baru dimulai: simpan snapshot untuk undo
                if mode in ("draw", "erase") and prev_mode != mode:
                    undo_stack.append(canvas.copy())
                    prev_point = None

                if mode == "draw":
                    mode_text = "Mode: Menggambar"
                    color = COLORS[color_index][1]
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (x, y), color, thickness)
                    else:
                        cv2.circle(canvas, (x, y), max(thickness // 2, 1), color, -1)
                    prev_point = (x, y)
                    cv2.circle(frame, (x, y), thickness // 2 + 3, color, 2)

                elif mode == "erase":
                    mode_text = "Mode: Penghapus"
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (x, y), (0, 0, 0), ERASER_THICKNESS)
                    else:
                        cv2.circle(canvas, (x, y), ERASER_THICKNESS // 2, (0, 0, 0), -1)
                    prev_point = (x, y)
                    cv2.circle(frame, (x, y), ERASER_THICKNESS // 2, (255, 255, 255), 2)

                elif mode == "nav":
                    mode_text = "Mode: Navigasi (Diam)"
                    prev_point = None
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), cv2.FILLED)
                    mx, my = int(pts[12][0]), int(pts[12][1])
                    cv2.circle(frame, (mx, my), 8, (0, 255, 0), cv2.FILLED)
                    if y < TOOLBAR_H:
                        for i, (x1, y1, x2, y2, _) in enumerate(boxes):
                            if x1 <= x <= x2:
                                color_index = i
                                break
                else:
                    mode_text = "Siap"
                    prev_point = None
            else:
                # Tangan hilang: putus goresan supaya tidak ada garis loncat
                smooth = None
                prev_point = None

            prev_mode = mode

            # --- Render
            output = composite(frame, canvas)
            draw_toolbar(output, boxes, color_index)

            cv2.putText(output, "CODING - Air Canvas", (20, TOOLBAR_H + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output, mode_text, (20, TOOLBAR_H + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(output, f"Kuas: {thickness}px", (20, TOOLBAR_H + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if time.monotonic() < notice_until:
                cv2.putText(output, notice, (20, TOOLBAR_H + 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(
                output,
                "1 jari: gambar | 2 jari: pilih warna | 3 jari: hapus | "
                "C: bersihkan | Z: undo | S: simpan | +/-: kuas | F: layar penuh | Q: keluar",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, output)

            # --- Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("c"), ord("C")):
                undo_stack.append(canvas.copy())
                canvas[:] = 0
                prev_point = None
            elif key in (ord("z"), ord("Z")):
                if undo_stack:
                    canvas = undo_stack.pop()
                    prev_point = None
            elif key in (ord("s"), ord("S")):
                notice = f"Tersimpan: {save_canvas(canvas)}"
                notice_until = time.monotonic() + 3
            elif key in (ord("+"), ord("=")):
                thickness = min(thickness + 2, MAX_THICKNESS)
            elif key in (ord("-"), ord("_")):
                thickness = max(thickness - 2, MIN_THICKNESS)
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

            # Tutup jendela lewat tombol X
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
