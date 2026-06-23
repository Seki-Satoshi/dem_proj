import json
from typing import BinaryIO
from PIL import Image, ImageDraw
import numpy as np
import cv2

from dem_utils import upsample_2x_bilinear


STATUS_COLORS = {
    "CONVERTED": (100, 200, 100),
    "MISSING"  : (200, 100, 100),
    "PENDING"  : (180, 180, 180),
    "INVALID"  : (230, 160,  50),
    "FAILED"   : (150,  50,  50),
    "FILLED"   : (100, 150, 220),
}

def visualize_db(db_path:str, output_path: str, cell_size: int = 40):
    with open(db_path, "rt") as fdb:
        db = json.load(fdb)
    
    grid_w = db["grid_width"]
    grid_h = db["grid_height"]

    legend_h = len(STATUS_COLORS) * 20 + 20
    img_w = grid_w * cell_size
    img_h = grid_h * cell_size
    img = Image.new("RGB", (img_w, img_h + legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for code, tile in db["tiles"].items():
        x, y = tile["x"], tile["y"]
        color = STATUS_COLORS.get(tile["status"], (200, 200, 200))

        # y軸反転(y=0が南=画像下)
        px = x * cell_size
        py = (grid_h - 1 - y) * cell_size

        draw.rectangle(
            [px, py, px+cell_size-1, py+cell_size-1],
            fill=color, outline=(50,50,50)
        )
        #コード下4桁
        draw.text((px+2,py+2), code[-4:], fill=(0,0,0))
    
    for i, (status, color) in enumerate(STATUS_COLORS.items()):
        y_pos = img_h + 10 + i*20
        draw.rectangle([10, y_pos, 25, y_pos+15], fill=color, outline=(50,50,50))
        draw.text((30,y_pos), status, fill=(0,0,0))
    
    img.save(output_path)
    print(f"保存: {output_path}")


def dem_to_image(dem: np.ndarray) -> np.ndarray:
    mask = dem >= 0
    d_min = dem[mask].min()
    d_max = dem[mask].max()

    normalized = np.zeros(dem.shape, dtype=float)
    normalized[mask] = (dem[mask] - d_min) / (d_max - d_min) * 255.0

    return normalized.astype(np.uint8)


def visualize_dem(dem_npy: BinaryIO, output_path: str, upsample: bool = True, printout: bool = False, allow_pickle: bool = False):
    if not dem_npy.readable():
        raise IOError("dem_npy is not readable")

    dem = np.load(dem_npy, allow_pickle=allow_pickle)

    if upsample:
        dem = upsample_2x_bilinear(dem)

    img = dem_to_image(dem)

    if printout:
        cv2.imshow("DEM", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        cv2.imwrite(output_path, img)