import cv2
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import alexnet
from ultralytics import YOLO
from base_ctrl import BaseController
import os
from datetime import datetime

# === Constants ===
FRAME_WIDTH = 640
CENTER_X = FRAME_WIDTH / 2
BASE_SPEED = 0.18
SLOW_SPEED = 0.13
KP = 0.0015
KD = 0.0015
MIN_SPEED = 0.0
MAX_SPEED = 0.5
MAX_CONTROL = 0.3
WAIT_TIME = 2


# === Device and Labels ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['vehicle', 'red', 'green', 'straight', 'right', 'left', 'slow', 'stop']

TRIGGER_THRESHOLDS = {
    'red': 40,
    'stop': 50,
    'slow': 50,
    'vehicle': 50,
    'left': 50,
    'right': 50
}

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

# === Load YOLO Model ===
yolo_model = YOLO("best.pt")

# === Camera Config ===
gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=640, height=360, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
camera = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
if not camera.isOpened():
    raise RuntimeError("❌ 카메라 열기 실패")

# === Rover Init ===
base = BaseController('/dev/ttyUSB0', 115200)

# === State Machine ===
state = "WAIT_SIGNAL"
prev_error = 0
wait_start = None
stop_count = 0  # ✅ stop 감지를 카운트하는 변수
slow_count = 0
vehicle_avoided = False  # ✅ 차량 회피를 한 번만 수행하기 위한 플래그


# === Screenshot Directory ===
os.makedirs("snapshots", exist_ok=True)
last_snapshot_time = 0

# === YOLO Detection ===
def detect_objects(frame):
    results = yolo_model.predict(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = CLASSES[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        height = y2 - y1
        if label == "vehicle" and conf < 0.79:
            continue  # vehicle은 0.7 이상일 때만 인식
        if conf > 0.5:
            detections.append((label, height, conf, (int(x1), int(y1), int(x2), int(y2))))
    return detections

# ==============회피 기동=======================
def avoid_vehicle(vehicle_box):
    # x1, y1, x2, y2 = vehicle_box
    # box_center_x = (x1 + x2) / 2

    print("🛑 vehicle 감지됨 → 정지")
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
    time.sleep(1.0)
    base.base_json_ctrl({"T":1,"L":-0.3,"R": 0.3})
    time.sleep(1)

    base.base_json_ctrl({"T":1,"L":0.3,"R": 0.3})
    time.sleep(1.3)

    base.base_json_ctrl({"T":1,"L":0.3,"R": -0.3})
    time.sleep(0.8)

    base.base_json_ctrl({"T":1,"L":0.3,"R": 0.3})
    time.sleep(2)

    base.base_json_ctrl({"T":1,"L":0.3,"R": -0.3})
    time.sleep(0.8)
    base.base_json_ctrl({"T":1,"L":0.3,"R": 0.3})
    time.sleep(1.3)

    base.base_json_ctrl({"T":1,"L":-0.3,"R": 0.3})
    time.sleep(0.5)
    # if box_center_x < CENTER_X:
    #     print("↪ vehicle 왼쪽 → 오른쪽으로 회피")
    #     # 오른쪽으로 꺾고 → 왼쪽으로 되돌기
    #     base.base_json_ctrl({"T":1,"L": 0.3,"R": -0.3})
    #     time.sleep(1)

    #     base.base_json_ctrl({"T":1,"L":0.18,"R": 0.18})
    #     time.sleep(0.8)

    #     base.base_json_ctrl({"T":1,"L": -0.3,"R": 0.3})
    #     time.sleep(0.5)

    #     base.base_json_ctrl({"T":1,"L": 0.18,"R": 0.18})
    #     time.sleep(3)

    #     base.base_json_ctrl({"T":1,"L": -0.3,"R": 0.3})
    #     time.sleep(0.5)

    #     base.base_json_ctrl({"T":1,"L":0.18,"R": 0.18})
    #     time.sleep(1.2)


    # else:
    #     print("↩ vehicle 오른쪽 → 왼쪽으로 회피")
    #     # 왼쪽으로 꺾고 → 오른쪽으로 되돌기
    #     base.base_json_ctrl({"T":1,"L": -0.3,"R": 0.3})
    #     time.sleep(1)

    #     base.base_json_ctrl({"T":1,"L":0.18,"R": 0.18})
    #     time.sleep(1)

    #     base.base_json_ctrl({"T":1,"L": 0.3,"R": -0.3})
    #     time.sleep(1)

    #     base.base_json_ctrl({"T":1,"L": 0.18,"R": 0.18})
    #     time.sleep(1)

    #     base.base_json_ctrl({"T":1,"L": 0.3,"R": -0.3})

    #     time.sleep(1)

    #     base.base_json_ctrl({"T":1,"L":0.18,"R": 0.18})
    #     time.sleep(1)

    #     base.base_json_ctrl({"T":1,"L": -0.3,"R": 0.3})
    #     time.sleep(1)


# === Line Following ===
def run_line_following(frame):
    global prev_error
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_x = rf_model(input_tensor)[0][0].item()
        pred_x = (pred_x / 2 + 0.5) * FRAME_WIDTH
    error = pred_x - CENTER_X
    derivative = error - prev_error
    prev_error = error
    control = 0 if abs(error) < 5 else KP * error + KD * derivative
    control = max(min(control, MAX_CONTROL), -MAX_CONTROL)
    left = BASE_SPEED + control
    right = BASE_SPEED - control
    return max(min(left, MAX_SPEED), MIN_SPEED), max(min(right, MAX_SPEED), MIN_SPEED)

# === Main Loop ===
try:
    while True:
        for _ in range(3):
            camera.grab()

        ret, frame = camera.read()
        if not ret:
            print("❌ 프레임 수신 실패")
            continue

        current_time = time.time()
        detections = detect_objects(frame)

        vis_frame = frame.copy()
        for (label, height, conf, (x1, y1, x2, y2)) in detections:
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"{label} {conf:.2f} h={int(height)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        labels_triggered = {label for (label, height, conf, _) in detections
                            if label in TRIGGER_THRESHOLDS and height > TRIGGER_THRESHOLDS[label]}

        if detections:
            print(f"[STATE: {state}] ▶ 감지: {[d[:2] for d in detections]}")

        if detections and current_time - last_snapshot_time >= 0.5:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = f"snapshots/detection_{timestamp}.jpg"
            cv2.imwrite(path, vis_frame)
            print(f"📸 스냅샷 저장됨: {path}")
            last_snapshot_time = current_time

        # === 상태 처리 ===
        if state == "WAIT_SIGNAL":
            if "red" in labels_triggered:
                print("🟥 red 감지 → 정지 유지")
                base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
            else:
                print("✅ red 없음 → FOLLOW_LINE")
                state = "FOLLOW_LINE"

        elif state == "FOLLOW_LINE":
            if "red" in labels_triggered:
                print("🟥 red 재감지 → 정지")
                base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
                state = "WAIT_SIGNAL"
                continue

            if "stop" in labels_triggered:
                stop_count += 1
                if stop_count <= 2:
                    print("🛑 stop 감지 → 일시 정지")
                    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
                    wait_start = time.time()
                    state = "WAIT_STOP"
                    continue

            elif "slow" in labels_triggered and slow_count < 2:
                slow_count += 1
                print("🐢 slow 감지 → 속도 감속")
                BASE_SPEED = SLOW_SPEED
                l_speed, r_speed = run_line_following(frame)
                base.base_json_ctrl({"T": 1, "L": l_speed, "R": r_speed})
                time.sleep(2)
                BASE_SPEED = 0.18
                continue


            elif "vehicle" in labels_triggered and not vehicle_avoided:
                print("🚗 vehicle 감지 → 회피 동작 시작")
                # 감지된 vehicle 중 첫 번째 박스를 회피 대상으로 선택
                for (label, height, conf, box) in detections:
                    if label == "vehicle":
                        avoid_vehicle(box)
                        vehicle_avoided = True
                        break
                continue
            

            elif "left" in labels_triggered:
                print("⬅️ left 감지 → 좌회전")
                base.base_json_ctrl({"T": 1, "L": 0.00, "R": 0.35})
                time.sleep(1.0)
                continue

            elif "right" in labels_triggered:
                print("➡️ right 감지 → 우회전")
                base.base_json_ctrl({"T": 1, "L": 0.35, "R": 0.00})
                time.sleep(1.0)
                continue

            l_speed, r_speed = run_line_following(frame)
            base.base_json_ctrl({"T": 1, "L": l_speed, "R": r_speed})
            print(f"[LINE] L={l_speed:.2f}, R={r_speed:.2f}")

        elif state == "WAIT_STOP":
            if time.time() - wait_start >= WAIT_TIME:
                print("⏱ stop 완료 → 재출발")
                state = "FOLLOW_LINE"

        time.sleep(0.05)

except KeyboardInterrupt:
    print("🛑 종료 요청됨 (Ctrl+C)")

finally:
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
    camera.release()
    cv2.destroyAllWindows()
