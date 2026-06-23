import cv2
import numpy as np
from collections import deque

from dem_visualize import dem_to_image

# マウス選択用グローバル変数
rect = {"start": None, "end": None, "selecting": False}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rect["start"] = (x, y)
        rect["selecting"] = True
    elif event == cv2.EVENT_MOUSEMOVE and rect["selecting"]:
        rect["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        rect["end"] = (x, y)
        rect["selecting"] = False

def find_upstream(dem: np.ndarray, sy: int, sx: int) -> np.ndarray:
    h, w = dem.shape
    valid = np.zeros((h, w), dtype=bool) # False でリセット
    valid[sy, sx] = True
    queue = deque([(sy, sx)])

    while queue:
        cy, cx = queue.popleft()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not valid[ny, nx]:
                    if float(dem[ny, nx])+0.2 >= float(dem[cy, cx]):
                        valid[ny, nx] = True
                        queue.append((ny, nx))

    return valid

def select_upstream(npy_path: str, output_path: str, valid_path: str):
    dem = np.load(npy_path)
    img = dem_to_image(dem)

    # マウス選択
    cv2.namedWindow("DEM")
    cv2.setMouseCallback("DEM", mouse_callback)

    print("マウスドラックで矩形域選択, Enterで確定, ESCでキャンセル")
    while True:
        display = img.copy()
        if rect["start"] and rect["end"]:
            cv2.rectangle(display, rect["start"], rect["end"], (0, 255, 0), 10)
        cv2.imshow("DEM", display)

        key = cv2.waitKey(1)
        if key == 13 and rect["start"] and rect["end"]:  # Enterで確定
            break
        elif key == 27:  # ESC でキャンセル
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # 選択矩形内の最低点
    x0 = min(rect["start"][0], rect["end"][0])
    x1 = max(rect["start"][0], rect["end"][0])
    y0 = min(rect["start"][1], rect["end"][1])
    y1 = max(rect["start"][1], rect["end"][1])

    region = dem[y0:y1, x0:x1]
    local_idx = np.argmin(np.where(region >= 0, region, np.inf))
    ly, lx = np.unravel_index(local_idx, region.shape)
    gy, gx = y0 + ly, x0 + lx
    print(f"最低点: ({gx}, {gy}), 標高: {dem[gy, gx]:.2f}m")

    # 上流域BFS
    valid = find_upstream(dem, gy, gx)

    # 結果表示
    result = img.copy()
    result[~valid] = 0
    cv2.imshow("upstream", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # nodata部分を-9999に、有効部分のみ標高を保持
    out = np.full(dem.shape, -9999.0)
    out[valid & (dem >= 0)] = dem[valid & (dem >= 0)]
    np.save(output_path, out)
    np.save(valid_path, valid)