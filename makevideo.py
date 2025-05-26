# record_video.py
import cv2
import os
import time

# === GStreamer 파이프라인 ===
sensor_id = 0
downscale = 2
width, height = (1280, 720)
_width, _height = (width // downscale, height // downscale)
frame_rate = 30
flip_method = 0
contrast = 1.3
brightness = 0.2

gstreamer_pipeline = (
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
    "nvvidconv flip-method=%d, interpolation-method=1 ! "
    "videobalance contrast=%.1f brightness=%.1f ! "
    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
    % (
        sensor_id, width, height, frame_rate,
        flip_method, contrast, brightness,
        _width, _height,
    )
)

# === 저장 파일명 생성 ===
def get_next_filename(prefix='video/car', ext='.avi'):
    i = 0
    while os.path.exists(f"{prefix}{i}{ext}"):
        i += 1
    return f"{prefix}{i}{ext}"

# 폴더 없으면 만들기
os.makedirs("video", exist_ok=True)

# 카메라 열기
camera = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
if not camera.isOpened():
    print("❌ 카메라 연결 실패!")
    exit()

# 저장기 설정
filename = get_next_filename()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, frame_rate, (_width, _height))
print(f"🎥 녹화 시작: {filename}")

# 영상 저장 루프 (팝업 없음)
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ 프레임 수신 실패")
            break
        out.write(frame)
        time.sleep(1.0 / frame_rate)  # 프레임 속도 맞춰서 잠깐 쉬기

except KeyboardInterrupt:
    print("🛑 녹화 중단 (Ctrl+C)")

finally:
    print("📦 녹화 종료 및 저장 완료")
    camera.release()
    out.release()

    