import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from unet import SimpleUNet3ch
from base_ctrl import BaseController
import time
import sys, select, tty, termios
from scipy.stats import linregress
import os

# === 설정 ===
FRAME_W, FRAME_H = 640, 360
CENTER_X = FRAME_W // 2

KP, KD = 0.0027, 0.0015
BASE_SPEED = 0.13
MIN_SPEED, MAX_SPEED = 0.05, 0.5
MAX_CONTROL = 0.3
SAMPLE_Y_DEFAULT = 250
ERROR_DEADZONE = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet3ch().to(device)
model.load_state_dict(torch.load("lane_model_3ch.pth", map_location=device))
model.eval()
transform = T.Compose([T.Resize((FRAME_H, FRAME_W)), T.ToTensor()])

base = BaseController("/dev/ttyUSB0", 115200)
base.prev_error = 0.0

state = "first"
x_target = CENTER_X
os.makedirs("debug_images", exist_ok=True)

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

def check_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None

def fit_line_from_mask(mask, color, img=None):
    if cv2.countNonZero(mask) < 50:
        return None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    pts = np.squeeze(largest)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None, None
    xs, ys = pts[:, 0], pts[:, 1]
    slope, intercept, *_ = linregress(ys, xs)
    return slope, intercept

# === GStreamer 카메라 (leaky queue 적용) ===
gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=640, height=360, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "queue max-size-buffers=1 leaky=downstream ! appsink"
)
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("❌ GStreamer 카메라 열기 실패")

# ... (기존 import와 설정 생략 동일)

SWITCH_SLOW_DURATION = 4  # 제어량 감속 시간(초)
SWITCH_CONTROL_FACTOR = 0.5  # 제어량 감쇠 계수
last_switch_time = None

try:
    frame_count = 0
    while True:
        key = check_key()
        if key == 't':
            state = "second" if state == "first" else "first"
            print(f"▶ 차선 변경: now → {state}")
            base.prev_error = 0.0
            last_switch_time = time.time()  # 전환 시점 기록
        elif key == 'q':
            break

        for _ in range(3):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)[0].cpu().numpy()
        masks = (pred > 0.1).astype(np.uint8)

        slopes, inters = [], []
        for i in range(3):
            s, b = fit_line_from_mask(masks[i]*255, None)
            slopes.append(s); inters.append(b)

        sample_y = SAMPLE_Y_DEFAULT
        if state == "first":
            if None not in (slopes[0], inters[0], slopes[1], inters[1]):
                lane_x1 = slopes[0]*sample_y + inters[0]
                lane_x2 = slopes[1]*sample_y + inters[1]
                x_raw = (lane_x1 + lane_x2) / 2
            else:
                continue
        elif state == "second":
            if None not in (slopes[1], inters[1], slopes[2], inters[2]):
                lane_x1 = slopes[1]*sample_y + inters[1]
                lane_x2 = slopes[2]*sample_y + inters[2]
                x_raw = (lane_x1 + lane_x2) / 2
            else:
                continue
        else:
            continue

        x_target = x_raw
        error = x_target - CENTER_X
        derivative = error - base.prev_error
        control = 0.0 if abs(error) < ERROR_DEADZONE else np.clip(KP * error + KD * derivative, -MAX_CONTROL, MAX_CONTROL)

        # 🔻 전환 직후 제어량 감쇠 적용
        if last_switch_time and time.time() - last_switch_time < SWITCH_SLOW_DURATION:
            control *= SWITCH_CONTROL_FACTOR

        base.prev_error = error

        left = np.clip(BASE_SPEED + control, MIN_SPEED, MAX_SPEED)
        right = np.clip(BASE_SPEED - control, MIN_SPEED, MAX_SPEED)
        right = right + 0.04
        
        base.base_json_ctrl({"T":1, "L":left, "R":right})

        vis = frame.copy()
        for i, (slope, intercept) in enumerate(zip(slopes, inters)):
            if slope is not None and intercept is not None:
                y1, y2 = 0, FRAME_H
                x1 = int(slope * y1 + intercept)
                x2 = int(slope * y2 + intercept)
                color = [(0,0,255), (0,255,0), (255,0,0)][i]
                cv2.line(vis, (x1,y1), (x2,y2), color, 2)

        print(f" y={sample_y}, x_raw={x_raw:.1f}, err={error:.1f}, ctrl={control:.3f}, L={left:.2f}, R={right:.2f}")
        time.sleep(0.05)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    base.base_json_ctrl({"T":1, "L":0.0, "R":0.0})
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

