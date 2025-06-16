import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

json_path = '000_1612_fixed.json'
image_root = './images'
mask_save_root = './masks_3ch'

os.makedirs(mask_save_root, exist_ok=True)

with open(json_path, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

for item in tqdm(data):
    h_samples = item['h_samples']
    lanes = item['lanes']
    raw_file = item['raw_file']
    image_path = os.path.join(image_root, os.path.basename(raw_file))

    if not os.path.exists(image_path):
        continue

    mask = np.zeros((3, 360, 640), dtype=np.uint8)

    for idx, lane in enumerate(lanes):
        if idx > 2:
            break

        pts = [(x, y) for x, y in zip(lane, h_samples) if x != -2]

        if len(pts) < 2:
            continue

        # 3개 이상일 경우 부드러운 곡선
        if len(pts) >= 3:
            pts = np.array(pts)
            xs, ys = pts[:, 0], pts[:, 1]

            # y 기준으로 정렬 (아래로 내려가는 차선)
            sorted_idx = np.argsort(ys)
            xs, ys = xs[sorted_idx], ys[sorted_idx]

            try:
                spline_x = make_interp_spline(ys, xs, k=2)  # 2차 보간
                y_dense = np.linspace(min(ys), max(ys), 100)
                x_dense = spline_x(y_dense)

                for i in range(len(y_dense) - 1):
                    pt1 = (int(x_dense[i]), int(y_dense[i]))
                    pt2 = (int(x_dense[i + 1]), int(y_dense[i + 1]))
                    if 0 <= pt1[0] < 640 and 0 <= pt1[1] < 360 and 0 <= pt2[0] < 640 and 0 <= pt2[1] < 360:
                        if idx == 0:
                            cv2.line(mask[2], pt1, pt2, 255, thickness=4)
                        elif idx == 1:
                            cv2.line(mask[1], pt1, pt2, 255, thickness=4)
                        elif idx == 2:
                            cv2.line(mask[0], pt1, pt2, 255, thickness=4)
            except Exception as e:
                print(f"[spline 오류] {e}")
                continue
        else:
            # 두 점만 있으면 직선
            pt1 = tuple(pts[0])
            pt2 = tuple(pts[1])
            if idx == 0:
                cv2.line(mask[2], pt1, pt2, 255, thickness=4)
            elif idx == 1:
                cv2.line(mask[1], pt1, pt2, 255, thickness=4)
            elif idx == 2:
                cv2.line(mask[0], pt1, pt2, 255, thickness=4)

    mask_rgb = np.transpose(mask, (1, 2, 0))
    img_name = os.path.basename(raw_file).replace('.jpg', '_mask.png').replace('.png', '_mask.png')
    cv2.imwrite(os.path.join(mask_save_root, img_name), mask_rgb)

