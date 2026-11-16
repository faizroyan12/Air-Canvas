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

NUM_HANDS = 2            # tiap tangan bekerja sendiri-sendiri
SMOOTHING = 0.5          # 0..1, makin kecil makin halus tapi makin "lambat"
TOOLBAR_H = 80
UNDO_LIMIT = 15
MIN_THICKNESS, MAX_THICKNESS = 2, 40
ERASER_THICKNESS = 45

COLORS = [  # BGR
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

FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}

MODE_NAMES = {"draw": "Menggambar", "erase": "Penghapus", "nav": "Pilih warna", "idle": "Siap"}


# ---------------------------------------------------------------- Fungsi bantu
def open_camera():
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
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def finger_states(pts):
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
    return [(x0 + i * (size + gap), y0, x0 + i * (size + gap) + size, y0 + size, color)
            for i, (_, color) in enumerate(COLORS)]


def draw_toolbar(img, boxes, selected, rainbow):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], TOOLBAR_H), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (x1, y1, x2, y2, color) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        if i == selected and not rainbow:
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 2)


def draw_stroke(canvas, prev, cur, color, thickness, mirror):
    """Gambar garis dari prev ke cur. Kalau prev None, gambar titik.
    Kalau mirror aktif, gambar juga bayangan cermin horizontalnya."""
    w = canvas.shape[1]

    def flip(p):
        return None if p is None else (w - 1 - p[0], p[1])

    pairs = [(prev, cur)]
    if mirror:
        pairs.append((flip(prev), flip(cur)))
    for p, c in pairs:
        if p is None:
            cv2.circle(canvas, c, max(thickness // 2, 1), color, -1)
        else:
            cv2.line(canvas, p, c, color, thickness)


def composite(base, canvas, glow):
    """Tempelkan coretan di atas base (video atau papan), dengan efek glow opsional."""
    out = base.copy()
    if glow:
        h, w = canvas.shape[:2]
        small = cv2.resize(canvas, (max(w // 4, 1), max(h // 4, 1)))
        small = cv2.GaussianBlur(small, (0, 0), 6)
        halo = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        out = cv2.add(out, cv2.convertScaleAbs(halo, alpha=1.8))
    mask = canvas.any(axis=2)
    out[mask] = canvas[mask]
    return out


def save_canvas(canvas):
    alpha = (canvas.any(axis=2) * 255).astype(np.uint8)
    bgra = np.dstack([canvas, alpha])
    name = time.strftime("air_canvas_%Y%m%d_%H%M%S.png")
    path = Path(__file__).parent / name
    cv2.imwrite(str(path), bgra)
    return path.name


# ---------------------------------------------------------------- Logika utama
class AirCanvas:
    def __init__(self, detector):
        self.detector = detector
        self.canvas = None
        self.undo_stack = deque(maxlen=UNDO_LIMIT)
        self.hands = {}          # status tiap tangan (halus, titik sebelumnya, mode)
        self.last_ts = 0
        self.color_index = 0
        self.thickness = 7
        self.hue = 0
        self.glow = False
        self.rainbow = False
        self.mirror = False
        self.board = False
        self.notice = ""
        self.notice_until = 0.0

    def current_color(self):
        if self.rainbow:
            self.hue = (self.hue + 3) % 180
            bgr = cv2.cvtColor(np.uint8([[[self.hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            return tuple(int(v) for v in bgr)
        return COLORS[self.color_index][1]

    def push_undo(self):
        self.undo_stack.append(self.canvas.copy())

    def process(self, frame):
        """Proses satu frame (sudah di-flip) dan kembalikan gambar untuk ditampilkan."""
        h, w = frame.shape[:2]
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.undo_stack.clear()
        boxes = toolbar_boxes(w)

        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, ts)

        seen = set()
        render = []          # info untuk digambar di atas hasil akhir
        mode_texts = []
        pushed = False

        for i, lms in enumerate(result.hand_landmarks or []):
            # Kunci tangan: "Left"/"Right" dari MediaPipe, cadangan: nomor urut
            label = f"H{i}"
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name
            key = label if label not in seen else f"{label}{i}"
            seen.add(key)
            st = self.hands.setdefault(key, {"smooth": None, "prev": None, "mode": "none"})

            pts = [(lm.x * w, lm.y * h) for lm in lms]
            raw = np.array(pts[8])
            st["smooth"] = raw if st["smooth"] is None else (
                SMOOTHING * raw + (1 - SMOOTHING) * st["smooth"])
            x, y = int(st["smooth"][0]), int(st["smooth"][1])

            mode = get_mode(finger_states(pts))

            if mode in ("draw", "erase") and st["mode"] != mode:
                if not pushed:
                    self.push_undo()
                    pushed = True
                st["prev"] = None

            if mode == "draw":
                color = self.current_color()
                draw_stroke(self.canvas, st["prev"], (x, y), color, self.thickness, self.mirror)
                st["prev"] = (x, y)
                render.append((pts, mode, x, y, color))
            elif mode == "erase":
                draw_stroke(self.canvas, st["prev"], (x, y), (0, 0, 0),
                            ERASER_THICKNESS, self.mirror)
                st["prev"] = (x, y)
                render.append((pts, mode, x, y, (255, 255, 255)))
            elif mode == "nav":
                st["prev"] = None
                if y < TOOLBAR_H:
                    for idx, (x1, _, x2, _, _) in enumerate(boxes):
                        if x1 <= x <= x2:
                            self.color_index = idx
                            self.rainbow = False
                            break
                render.append((pts, mode, x, y, (0, 255, 0)))
            else:
                st["prev"] = None
                render.append((pts, mode, x, y, (0, 102, 255)))

            st["mode"] = mode
            mode_texts.append(MODE_NAMES[mode])

        # Tangan yang hilang dari kamera: putus goresannya
        for key in list(self.hands):
            if key not in seen:
                del self.hands[key]

        # --- Render
        base = np.full_like(frame, 25) if self.board else frame
        output = composite(base, self.canvas, self.glow)
        draw_toolbar(output, boxes, self.color_index, self.rainbow)
        if self.mirror:
            cv2.line(output, (w // 2, TOOLBAR_H), (w // 2, h), (90, 90, 90), 1)

        for pts, mode, x, y, color in render:
            for a, b in HAND_CONNECTIONS:
                cv2.line(output, (int(pts[a][0]), int(pts[a][1])),
                         (int(pts[b][0]), int(pts[b][1])), (0, 0, 255), 2)
            for px, py in pts:
                cv2.circle(output, (int(px), int(py)), 5, (0, 102, 255), cv2.FILLED)
            if mode == "draw":
                cv2.circle(output, (x, y), self.thickness // 2 + 3, color, 2)
            elif mode == "erase":
                cv2.circle(output, (x, y), ERASER_THICKNESS // 2, color, 2)
            elif mode == "nav":
                cv2.circle(output, (x, y), 8, color, cv2.FILLED)
                cv2.circle(output, (int(pts[12][0]), int(pts[12][1])), 8, color, cv2.FILLED)

        effects = [n for n, on in (("Glow", self.glow), ("Pelangi", self.rainbow),
                                   ("Simetri", self.mirror), ("Papan", self.board)) if on]
        mode_line = "Mode: " + " | ".join(mode_texts) if mode_texts else "Mencari Tangan..."
        y0 = TOOLBAR_H
        cv2.putText(output, "CODING - Air Canvas", (20, y0 + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(output, mode_line, (20, y0 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(output, f"Kuas: {self.thickness}px   Efek: {', '.join(effects) or '-'}",
                    (20, y0 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if time.monotonic() < self.notice_until:
            cv2.putText(output, self.notice, (20, y0 + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, "1 jari: gambar | 2 jari: pilih warna | 3 jari: hapus | "
                    "boleh pakai 2 tangan sekaligus",
                    (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(output, "C: bersihkan | Z: undo | S: simpan | +/-: kuas | G: glow | "
                    "R: pelangi | M: simetri | B: papan | F: layar penuh | Q: keluar",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return output

    def handle_key(self, key):
        """Return 'quit', 'fullscreen', atau None."""
        k = chr(key).lower() if 0 <= key < 256 else ""
        if k == "q":
            return "quit"
        if k == "f":
            return "fullscreen"
        if k == "c":
            self.push_undo()
            self.canvas[:] = 0
            self.hands.clear()
        elif k == "z":
            if self.undo_stack:
                self.canvas = self.undo_stack.pop()
                self.hands.clear()
        elif k == "s":
            self.notice = f"Tersimpan: {save_canvas(self.canvas)}"
            self.notice_until = time.monotonic() + 3
        elif k in ("+", "="):
            self.thickness = min(self.thickness + 2, MAX_THICKNESS)
        elif k in ("-", "_"):
            self.thickness = max(self.thickness - 2, MIN_THICKNESS)
        elif k == "g":
            self.glow = not self.glow
        elif k == "r":
            self.rainbow = not self.rainbow
        elif k == "m":
            self.mirror = not self.mirror
        elif k == "b":
            self.board = not self.board
        return None


def main():
    detector = create_detector()
    cap = open_camera()
    if cap is None:
        detector.close()
        sys.exit("Kamera tidak ditemukan.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = True
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    app = AirCanvas(detector)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.imshow(WINDOW_NAME, app.process(frame))

            action = app.handle_key(cv2.waitKey(1) & 0xFF)
            if action == "quit":
                break
            if action == "fullscreen":
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
