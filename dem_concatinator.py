import json
import numpy as np
from typing import BinaryIO

# all npy to one npy
def concat_tiles(db_path: str, output_path: str):
    with open(db_path, "rt") as fdb:
        db = json.load(fdb)

    width, height = db["grid_width"], db["grid_height"]
    tile_view = [[""]*width for _ in range(height)]
    for code, tile in db["tiles"].items():
        if tile["status"] == "PENDING":
            continue
        tile_view[tile["y"]][tile["x"]] = tile["npy_path"]

    rows = []
    for y in range(height-1, -1, -1):
        row_tiles = [np.load(tile_view[y][x]) for x in range(width)]
        rows.append(np.hstack(row_tiles))

    result = np.vstack(rows)
    np.save(output_path, result)