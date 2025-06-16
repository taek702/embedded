import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.stats import linregress
from unet import SimpleUNet3ch

# === 모델 로드 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet3ch().to(device)
model.load_state_dict(torch.load("lane_model_3ch.pth", map_location=device))
model.eval()

# === 전처리 (Colab 기준 유지) ===
transform = T.Compose([
    T.Resize((360, 640)),
    T.ToTensor()
])

# === GStreamer 카메라 스트림 ===
gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=640, height=360, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("❌ GStreamer 카메라 열기 실패")

# === 유틸 함수: 직선 피팅 ===
def fit_line_from_mask(mask, color, label, img=None, min_points=50):
    if cv2.countNonZero(mask) < min_points:
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
    for y in range(0, 360, 2):
        x = int(slope * y + intercept)
        if img is not None and 0 <= x < 640:
            cv2.circle(img, (x, y), 1, color, -1)
    return slope, intercept

# === 유틸 함수: 중앙선 그리기 ===
def draw_centerline(s1, i1, s2, i2, img, color):
    if None in [s1, i1, s2, i2]:
        return
    avg_slope = (s1 + s2) / 2
    avg_inter = (i1 + i2) / 2
    for y in range(0, 360, 2):
        x = int(avg_slope * y + avg_inter)
        if 0 <= x < img.shape[1]:
            cv2.circle(img, (x, y), 1, color, -1)

# === 실시간 루프 ===
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ✅ 전처리
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = model(input_tensor)[0].cpu().numpy()

        binary_mask = (pred_mask > 0.1).astype(np.uint8)

        # ✅ 색상 고정: Left = Red, Center = Green, Right = Blue
        color_mask = np.zeros((360, 640, 3), dtype=np.uint8)
        color_mask[:, :, 2] = binary_mask[2] * 255  # Right → Blue
        color_mask[:, :, 1] = binary_mask[1] * 255  # Center → Green
        color_mask[:, :, 0] = binary_mask[0] * 255  # Left → Red

        # ✅ 원본 프레임 resize
        resized_frame = cv2.resize(frame, (640, 360))
        overlay = cv2.addWeighted(resized_frame, 0.7, color_mask, 0.5, 0)

        # ✅ 직선 피팅
        slope_l, inter_l = fit_line_from_mask(binary_mask[0] * 255, (0, 0, 255), "Left", overlay)    # 빨강
        slope_c, inter_c = fit_line_from_mask(binary_mask[1] * 255, (0, 255, 0), "Center", overlay)  # 초록
        slope_r, inter_r = fit_line_from_mask(binary_mask[2] * 255, (255, 0, 0), "Right", overlay)   # 파랑

        # ✅ 중앙선 (좌-중 / 중-우)
        draw_centerline(slope_l, inter_l, slope_c, inter_c, overlay, (0, 255, 255))    # 좌-중 → 하늘색
        draw_centerline(slope_c, inter_c, slope_r, inter_r, overlay, (255, 255, 255))  # 중-우 → 흰색

        # ✅ 출력
        cv2.imshow("🛣️ Real-Time Lane + Centerline", overlay)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
