import cv2
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import alexnet
from base_ctrl import BaseController
import sys
import select
import tty
import termios

# === Constants ===
FRAME_WIDTH = 640
CENTERLINE_TRUE = 450
CENTERLINE_FALSE = 220
BASE_SPEED = 0.18
KP = 0.0015
KD = 0.001
MIN_SPEED = 0.0
MAX_SPEED = 0.4
MAX_CONTROL = 0.35

# === Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load Road Following Model ===
rf_model = alexnet()
rf_model.classifier[6] = torch.nn.Linear(4096, 2)
rf_model.load_state_dict(torch.load("road_following_model.pth", map_location=device))
rf_model.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Camera Config ===
gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=640, height=360, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
)
camera = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
if not camera.isOpened():
    raise RuntimeError("❌ 카메라 열기 실패")

# === Rover Init ===
base = BaseController('/dev/ttyUSB0', 115200)

# === State Machine ===
state = "WAIT_SIGNAL"
prev_error = 0
centerline_state = True
center_x = CENTERLINE_TRUE

# === Keyboard Input Setup ===
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

def check_key_pressed():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None

# === Line Following ===
def run_line_following(frame, center_x):
    global prev_error
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_x = rf_model(input_tensor)[0][0].item()
        pred_x = (pred_x / 2 + 0.5) * FRAME_WIDTH

    error = pred_x - center_x  # ✅ 무조건 이 방식 고정
    derivative = error - prev_error
    prev_error = error
    control = KP * error + KD * derivative
    control = max(min(control, MAX_CONTROL), -MAX_CONTROL)
    left = BASE_SPEED + control
    right = BASE_SPEED - control
    return max(min(left, MAX_SPEED), MIN_SPEED), max(min(right, MAX_SPEED), MIN_SPEED), pred_x


# === Main Loop ===
try:
    while True:
        key = check_key_pressed()
        if key == 't':
            centerline_state = not centerline_state
            center_x = CENTERLINE_TRUE if centerline_state else CENTERLINE_FALSE
            prev_error = 0
            print(f"🔁 centerline_state 전환됨 → center_x = {center_x}")

        for _ in range(3):
            camera.grab()

        ret, frame = camera.read()
        if not ret:
            print("❌ 프레임 수신 실패")
            continue

        if state == "WAIT_SIGNAL":
            state = "FOLLOW_LINE"

        elif state == "FOLLOW_LINE":
            reverse = not centerline_state  # ✅ 방향 반전 여부 결정
            l_speed, r_speed, pred_x = run_line_following(frame, center_x)
            base.base_json_ctrl({"T": 1, "L": l_speed, "R": r_speed})
            print(f"[LINE] State={'TRUE' if centerline_state else 'FALSE'} / CenterX={center_x} / PredX={int(pred_x)} / L={l_speed:.2f}, R={r_speed:.2f}")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("🛑 종료 요청됨 (Ctrl+C)")

finally:
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
    camera.release()
    cv2.destroyAllWindows()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
