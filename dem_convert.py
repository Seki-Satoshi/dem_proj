import json
from typing import TextIO

from dem_utils import InvalidDataException, mesh_to_xy, xml2npy, xy_to_mesh


def init_db(tile_list: list[tuple], offset: tuple[int, int], db_path: str):
    x0, y0 = mesh_to_xy(offset[0])
    xm, ym = mesh_to_xy(offset[1])
    known = {code: (x, y, path) for code, x, y, path in tile_list}

    db = {
        "offset_lower": offset[0],
        "offset_upper": offset[1],
        "grid_width": xm-x0+1,
        "grid_height": ym-y0+1,
        "tiles": {}
    }

    for y in range(db["grid_height"]):
        for x in range(db["grid_width"]):
            code = xy_to_mesh(x+x0, y+y0)
            if code in known:
                _, _, xml_path = known[code]
                entry = {"x":x, "y":y, "status":"PENDING", "data_count":None, "npy_path":"", "xml_path":xml_path}
            else:
                entry = {"x":x, "y":y, "status":"MISSING", "data_count":None, "npy_path":"", "xml_path":""}
            db["tiles"][str(code)] = entry

    db["tiles"] = dict(sorted(db["tiles"].items(), key=lambda x: int(x[0])))
    with open(db_path, "wt") as fdb:
        json.dump(db, fdb, indent=2)

def convert_tiles(db_path: str, output_dir: str, filling:bool=False):
    with open(db_path, "rt") as fdb:
        db = json.load(fdb)
    
    for code, tile in db["tiles"].items():
        if tile["status"] != "PENDING":
            continue
        
        # status == PENDING here
        try:
            # open files
            ifp = open(tile["xml_path"])
            npy_path = f"{output_dir}/{code}.npy"
            ofp = open(npy_path, "wb")
            # convert
            count = xml2npy(ifp, ofp, filling)
            # set db
            tile["data_count"] = count
            tile["status"] = "CONVERTED"
            tile["npy_path"] = npy_path
        except InvalidDataException as e:
            print(f"{code}:データ数異常 >> {e}")
            tile["status"] = "INVALID"
        except Exception as e:
            tile["status"] = "FAILED"
            with open(db_path, "wt") as fdb:
                json.dump(db, fdb, indent=2)
            raise RuntimeError(f"{code} 変換失敗: {e}") from e
        finally:
            # close files
            ifp.close()
            ofp.close()

        # write db per tile
        with open(db_path, "wt") as fdb:
            json.dump(db, fdb, indent=2)