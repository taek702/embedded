import cv2
import os
import json
import numpy as np

# === 설정 ===
IMG_FILENAME_FMT = 'avoid/images1/frame_{:06d}.png'
OUTPUT_JSON = 'avoid/jin.json'
START_INDEX = 1380 # 이미지 파일명의 시작 인덱스
MAX_LANES = 3 # 최대 차선 개수
POINT_RADIUS = 4 # 점의 반지름
LINE_THICKNESS = 1 # 가이드라인 두께

# === 전역 변수 ===
current_frame_idx = 0
lane_selected = 0 # 0, 1, 2 (빨강, 초록, 파랑)
lane_colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR 순서 (OpenCV 기본: 빨강, 초록, 파랑)

# 이미지 정보 (초기 로드 시 설정)
img_height = 0
img_width = 0
y_samples = []
all_lane_points = [] # 모든 프레임, 모든 차선의 점들을 저장

# ⭐️ 추가: 마우스의 마지막 위치를 저장할 전역 변수
last_mouse_x = -1
last_mouse_y = -1

# === 초기화 함수 ===
def initialize_data():
    global img_height, img_width, y_samples, all_lane_points, num_frames

    # 기본 해상도 640x360으로 설정
    img_height, img_width = 360, 640

    # 'line/images/' 디렉토리 존재 여부 확인 및 생성
    if not os.path.exists(os.path.dirname(IMG_FILENAME_FMT)):
        os.makedirs(os.path.dirname(IMG_FILENAME_FMT))
        print(f"경고: 디렉토리 '{os.path.dirname(IMG_FILENAME_FMT)}'가 생성되었습니다. 이미지를 추가해주세요.")
        num_frames = 0
    else:
        num_frames = len(os.listdir(os.path.dirname(IMG_FILENAME_FMT)))

    if num_frames == 0:
        print("경고: 'line/images/' 경로에 이미지가 없습니다. 기본값으로 초기화합니다.")
        # upperbound를 기준으로 3개로 나누어 y_samples 설정
        upperbound = img_height * 0.9  # 95% 위치
        lowerbound = img_height * 0.2   # 40% 위치
        step = (upperbound - lowerbound) / 4

        y_samples = [int(lowerbound + step * i) for i in range(4)]
        all_lane_points = [[[-2 for _ in y_samples] for _ in range(MAX_LANES)] for _ in range(1)] # 최소 1프레임
        print("초기화: 빈 캔버스로 시작합니다. 이미지를 추가해주세요.")
    else:
        # 첫 번째 이미지 로드하여 크기 확인
        first_image_path = IMG_FILENAME_FMT.format(START_INDEX + 0)
        if not os.path.exists(first_image_path):
            print(f"오류: 초기 이미지 '{first_image_path}'를 찾을 수 없습니다. start_index부터 이미지가 있는지 확인해주세요.")
            img_height, img_width = 720, 1280 # 기본 해상도
        else:
            first_image = cv2.imread(first_image_path)
            if first_image is None:
                print(f"오류: 이미지 '{first_image_path}'를 로드할 수 없습니다. 파일이 손상되었거나 형식이 올바르지 않습니다.")
                img_height, img_width = 720, 1280
            else:
                img_height, img_width = first_image.shape[:2]

        # upperbound를 기준으로 3개로 나누어 y_samples 설정
        upperbound = img_height * 0.9  # 95% 위치
        lowerbound = img_height * 0.2   # 40% 위치
        step = (upperbound - lowerbound) / 4

        y_samples = [int(lowerbound + step * i) for i in range(4)]
        all_lane_points = [[[-2 for _ in y_samples] for _ in range(MAX_LANES)] for _ in range(num_frames)]
        print(f"총 {num_frames}개의 프레임을 로드할 준비가 되었습니다. 이미지 크기: {img_width}x{img_height}")


# === 이미지 표시 함수 ===
def draw_image_and_points():
    current_image_path = IMG_FILENAME_FMT.format(START_INDEX + current_frame_idx)
    img = cv2.imread(current_image_path)

    if img is None:
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        cv2.putText(img, f"이미지 없음: {os.path.basename(current_image_path)}",
                    (50, img_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if img.shape[0] != img_height or img.shape[1] != img_width:
            print(f"경고: 이미지 크기 불일치 ({img.shape[1]}x{img.shape[0]}). 창 크기에 맞게 조정합니다.")
            img = cv2.resize(img, (img_width, img_height))

    # Y 가이드라인 그리기 (3개만 표시)
    for y in y_samples:
        cv2.line(img, (0, y), (img_width, y), (100, 100, 100), LINE_THICKNESS) # 회색

    # 현재 프레임의 차선 점들 그리기
    for l_idx in range(MAX_LANES):
        lane_color = lane_colors_bgr[l_idx]
        for i, x in enumerate(all_lane_points[current_frame_idx][l_idx]):
            if x != -2: # -2는 점이 없는 상태
                cv2.circle(img, (x, y_samples[i]), POINT_RADIUS, lane_color, -1) # -1은 채우기

    # 현재 선택된 차선과 프레임 정보 표시
    info_text = f"프레임: {current_frame_idx + 1}/{num_frames} | 차선: {lane_selected} ({['빨강', '초록', '파랑'][lane_selected]})"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # 흰색 텍스트

    # ⭐️ 마우스 현재 위치 표시 (디버깅 및 사용자 안내용)
    if last_mouse_x != -1 and last_mouse_y != -1:
        cv2.circle(img, (last_mouse_x, last_mouse_y), POINT_RADIUS + 2, (255, 255, 0), 1) # 노란색 테두리

    cv2.imshow("Lane Labeler", img)


# === 마우스 이벤트 핸들러 ===
def mouse_callback(event, x, y, flags, param):
    global lane_selected, last_mouse_x, last_mouse_y

    # ⭐️ 마우스 이동 시 현재 위치 업데이트
    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_x = x
        last_mouse_y = y
        # 마우스 이동 시에도 화면을 업데이트하여 마우스 위치 표시를 실시간으로 반영
        draw_image_and_points()

    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭
        # 가장 가까운 y_sample 인덱스 찾기
        closest_y_idx = min(range(len(y_samples)), key=lambda i: abs(y_samples[i] - y))
        
        # 현재 선택된 차선에 점 추가
        all_lane_points[current_frame_idx][lane_selected][closest_y_idx] = x
        print(f"마우스 클릭: 프레임 {current_frame_idx+1}, 차선 {lane_selected}, y_sample {y_samples[closest_y_idx]} 에 x={x} 점 찍음")
        draw_image_and_points() # 변경 사항 반영


# === 데이터 저장 함수 ===
def save_labels():
    data_to_save = []
    for i in range(num_frames):
        sample = {
            "lanes": all_lane_points[i],
            "h_samples": y_samples,
            "raw_file": IMG_FILENAME_FMT.format(START_INDEX + i)
        }
        data_to_save.append(sample)

    output_dir = os.path.dirname(OUTPUT_JSON)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_JSON, 'w') as f:
        for item in data_to_save:
            json.dump(item, f)
            f.write("\n")
    print(f"[✅] 라벨 데이터가 '{OUTPUT_JSON}' 에 저장되었습니다.")

# === 메인 실행 부분 ===
if __name__ == "__main__":
    initialize_data()

    if num_frames == 0:
        print("\n** [중요] 이미지를 추가한 후 다시 실행해주세요. **")
        print("    예: 'line/images/frame_000701.png', 'line/images/frame_000702.png' 등")
        # 이미지가 없어도 창은 열어두고 사용자 입력을 기다립니다.
        cv2.namedWindow("Lane Labeler")
        cv2.setMouseCallback("Lane Labeler", mouse_callback)
        draw_image_and_points() # 빈 이미지를 한 번 그려줍니다.
        cv2.waitKey(0) # 아무 키나 누르면 종료
        cv2.destroyAllWindows()
        exit() # 이미지가 없으면 바로 종료

    cv2.namedWindow("Lane Labeler") # 윈도우 생성
    cv2.setMouseCallback("Lane Labeler", mouse_callback) # 마우스 콜백 연결

    draw_image_and_points() # 첫 이미지 그리기

    while True:
        key = cv2.waitKey(1) & 0xFF # 1ms 대기, 키 입력 감지

        if key == ord('q'): # 'q' 키: 이전 프레임
            current_frame_idx = max(0, current_frame_idx - 1)
            draw_image_and_points()
            print(f"프레임 이동: {current_frame_idx + 1}")
        elif key == ord('e'): # 'e' 키: 다음 프레임
            current_frame_idx = min(num_frames - 1, current_frame_idx + 1)
            draw_image_and_points()
            print(f"프레임 이동: {current_frame_idx + 1}")
        elif key == ord('a'): # 'a' 키: 차선 왼쪽 이동
            lane_selected = (lane_selected - 1 + MAX_LANES) % MAX_LANES
            print(f"차선 선택: {lane_selected} ({['빨강', '초록', '파랑'][lane_selected]})")
            draw_image_and_points()
        elif key == ord('d'): # 'd' 키: 차선 오른쪽 이동
            lane_selected = (lane_selected + 1) % MAX_LANES
            print(f"차선 선택: {lane_selected} ({['빨강', '초록', '파랑'][lane_selected]})")
            draw_image_and_points()
        elif key == 32: # 스페이스바: 마우스 현재 위치에 점 찍기
            if last_mouse_x != -1 and last_mouse_y != -1:
                # 가장 가까운 y_sample 인덱스 찾기
                closest_y_idx = min(range(len(y_samples)), key=lambda i: abs(y_samples[i] - last_mouse_y))
                
                # 현재 선택된 차선에 점 추가
                all_lane_points[current_frame_idx][lane_selected][closest_y_idx] = last_mouse_x
                print(f"스페이스바: 프레임 {current_frame_idx+1}, 차선 {lane_selected}, y_sample {y_samples[closest_y_idx]} 에 x={last_mouse_x} 점 찍음")
                draw_image_and_points() # 변경 사항 반영
            else:
                print("경고: 마우스가 윈도우 내에 있지 않아 스페이스바로 점을 찍을 수 없습니다. 마우스를 움직여 주세요.")
        elif key == ord('r'): # 'r' 키: 현재 선택된 차선의 모든 점 초기화 (reset)
            all_lane_points[current_frame_idx][lane_selected] = [-2 for _ in y_samples]
            print(f"현재 프레임, 차선 {lane_selected}의 모든 점을 초기화했습니다.")
            draw_image_and_points()
        elif key == ord('s'): # 's' 키: 라벨 저장 (save)
            save_labels()
        elif key == 27: # ESC 키: 프로그램 종료
            print("ESC 키를 눌러 프로그램을 종료합니다.")
            break

    cv2.destroyAllWindows() # 모든 OpenCV 윈도우 닫기
