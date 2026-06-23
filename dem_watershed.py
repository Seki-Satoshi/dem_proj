import heapq
import numpy as np
import cv2

UNLABELED = 0
CATCHMENT = 1
OUTSIDE   = 2

def watershed_dem(dem: np.ndarray, seed_y: int, seed_x: int) -> np.ndarray:
    h, w = dem.shape
    labels   = np.zeros((h, w), dtype=int)
    in_queue = np.zeros((h, w), dtype=bool)

    labels[seed_y, seed_x] = CATCHMENT
    labels[0, :]  = OUTSIDE
    labels[-1, :] = OUTSIDE
    labels[:, 0]  = OUTSIDE
    labels[:, -1] = OUTSIDE

    heap = []
    for y in range(h):
        for x in range(w):
            if labels[y, x] != UNLABELED:
                heapq.heappush(heap, (dem[y, x], y, x))
                in_queue[y, x] = True

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while heap:
        elev, cy, cx = heapq.heappop(heap)
        cur = labels[cy, cx]

        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if labels[ny, nx] == UNLABELED:
                labels[ny, nx] = cur
                if not in_queue[ny, nx]:
                    # 記事に合わせてmin優先度を使用
                    heapq.heappush(heap, (min(elev, dem[ny, nx]), ny, nx))
                    in_queue[ny, nx] = True
            elif labels[ny, nx] not in (cur, UNLABELED):
                labels[ny, nx] = -1  # 分水嶺

    return labels


def make_test_dem(h=300, w=300) -> np.ndarray:
    """中央から左右に2つの円錐形の谷（少し重なる）"""
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    cx_l, cy_l = w // 3,     h // 2   # 左の谷の中心
    cx_r, cy_r = 2 * w // 3, h // 2   # 右の谷の中心
    R = int(w * 0.38)                  # 半径（重なるように少し大きめ）

    dist_l = np.sqrt((x - cx_l)**2 + (y - cy_l)**2)
    dist_r = np.sqrt((x - cx_r)**2 + (y - cy_r)**2)

    # 円錐形：中心が最低点、外側はRでクリップ（平坦な台地）
    cone_l = np.clip(dist_l, 0, R).astype(float)
    cone_r = np.clip(dist_r, 0, R).astype(float)

    # 各点は近い方の谷に属する
    return np.minimum(cone_l, cone_r)


def test_watershed():
    dem = make_test_dem()

    # 左の谷の最低点をシードに
    h, w = dem.shape
    seed_y, seed_x = h // 2, w // 3

    labels = watershed_dem(dem, seed_y, seed_x)

    # 可視化
    vis = np.zeros((*dem.shape, 3), dtype=np.uint8)
    vis[labels == CATCHMENT] = (100, 200, 100)  # 緑: 集水域
    vis[labels == OUTSIDE]   = ( 50,  50, 150)  # 青: 外部
    vis[labels == -1]        = (255, 255, 255)  # 白: 分水嶺
    vis[seed_y, seed_x]      = (  0,   0, 255)  # 赤: シード

    # DEMをグレースケールで重ねて地形を見やすく
    dem_norm = ((dem - dem.min()) / (dem.max() - dem.min()) * 80).astype(np.uint8)
    for c in range(3):
        vis[:, :, c] = np.clip(vis[:, :, c].astype(int) + dem_norm, 0, 255)

    cv2.imshow("watershed test", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_watershed()