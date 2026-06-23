import json
import numpy as np
from dem_convert import convert_tiles, init_db
from dem_utils import dem_path_generator, mesh_to_xy, offset, xml2npy
from dem_visualize import upsample_2x_bilinear, visualize_db, visualize_dem


# 653726 番地だけなのでそれだけ埋める
def dem653726_1m():
    DEM_PATH = "/Users/fukaya/Documents/pythons/dem_proj/input/FG-GML-543726-DEM1A-20250822"
    OUTPUT_DIR = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_1m"
    DB_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/db_1m.json"
    VIS_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_1m.png"
    print("generate dem_list")
    dem_list = dem_path_generator(DEM_PATH)
    offsets = offset(DEM_PATH)
    print("init db")
    init_db(dem_list, offsets, DB_PATH)
    print("converting...")
    convert_tiles(DB_PATH, OUTPUT_DIR, True)

    # 1mDEM cant use 5mDEM
    # so convert later
    # then db must "PENDING"
    with open(DB_PATH, "rt") as f:
        db = json.load(f)
    for code, tile in db["tiles"].items():
        if tile["status"] == "CONVERTED":
            tile["status"] = "PENDING"
    with open(DB_PATH, "wt") as f:
        json.dump(db, f, indent=2)

    visualize_db(DB_PATH, VIS_PATH)
    # for code, x, y, xml_path in dem_list:
    #     ifp = open(xml_path, "rt")
    #     ofp = open(f"{OUTPUT_DIR}/{code}.npy")
    #     xml2npy(ifp, ofp, True)
    #     ifp.close()
    #     ofp.close()

def dem653726_10m():
    DEM_PATH = "/Users/fukaya/Documents/pythons/dem_proj/input/dem10b/FG-GML-5437-26-dem10b-20161001.xml"
    OUTPUT_DIR = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_10m"
    print("converting...")
    ifp = open(DEM_PATH, "rt")
    ofp = open(f"{OUTPUT_DIR}/543726.npy", "wb")
    xml2npy(ifp, ofp, True)
    ifp.close()
    ofp.close()

def dem_resample_2x():
    forward_mesh = 543726
    DEM_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_10m/543726.npy"
    OUTPUT_DIR = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_10as5m"
    DB_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/db_10m.json"
    VIS_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_10m.png"

    print("dem 10m resample")
    with open(DEM_PATH, "rb") as f:
        dem_10m = np.load(f)
    dem_10m_as_5m = upsample_2x_bilinear(dem_10m)
    print(f"{dem_10m_as_5m.shape=}")

    print("init db")
    init_db([], (int(f"{forward_mesh}00"),int(f"{forward_mesh}99")), DB_PATH)
    with open(DB_PATH, "rt") as f:
        db = json.load(f)
    
    print("cut tile")
    for code, tile in db["tiles"].items():
        c3 = int(code[-1])
        r3 = int(code[-2])

        x0 = c3 * 225
        y0 = (9-r3) * 150
        arr = dem_10m_as_5m[y0:y0+150, x0:x0+225]

        npy_path = f"{OUTPUT_DIR}/{code}.npy"
        np.save(npy_path, arr)

        tile["status"] = "CONVERTED"
        tile["data_count"] = 225 * 150
        tile["npy_path"] = npy_path

        with open(DB_PATH, "wt") as f:
            json.dump(db, f, indent=2)
    
    visualize_db(DB_PATH, VIS_PATH)

def dem_downsample_5x():
    OUTPUT_DIR = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_1as5m"
    DB_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/db_1m.json"
    VIS_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/543726_1m_cv.png"

    print("load db")
    with open(DB_PATH, "rt") as f:
        db = json.load(f)
    
    print("resample dem")
    for code, tile in db["tiles"].items():
        if tile["status"] != "PENDING":
            continue

        with open(tile["npy_path"], "rb") as f:
            dem_1m:np.ndarray = np.load(f)
        dem_1m_as_5m = dem_1m.reshape(150, 5, 225, 5).mean(axis=(1,3))
        npy_path = f"{OUTPUT_DIR}/{code}.npy"
        with open(npy_path, "wb") as f:
            np.save(f, dem_1m_as_5m)
        
        tile["status"] = "CONVERTED"
        tile["data_count"] = 225 * 150
        tile["npy_path"] = npy_path

        with open(DB_PATH, "wt") as f:
            json.dump(db, f, indent=2)
    
    visualize_db(DB_PATH, VIS_PATH)
    print("downsample done")


# TODO: fill INVALID and MISSING in tiles
def dem_filler():
    DB_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/db.json"
    DB1M_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/db_1m.json"
    DB10M_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/db_10m.json"
    VIS_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/filled.png"

    print("load db")
    with open(DB_PATH, "rt") as f:
        db = json.load(f)
    with open(DB1M_PATH, "rt") as f:
        db_1m = json.load(f)
    with open(DB10M_PATH, "rt") as f:
        db_10m = json.load(f)
    
    print("filling...")
    for code, tile in db["tiles"].items():
        if tile["status"] == "INVALID":
            # 1mで埋められるか試して、埋められないなら10mを利用
            dem1m_tile = db_1m["tiles"].get(code)
            if dem1m_tile is not None and dem1m_tile["status"] == "CONVERTED":
                # 1mで埋められるので、npy_pathを入れ替え、statusを変更
                tile["npy_path"] = dem1m_tile["npy_path"]
                tile["status"] = "FILLED"
            else:
                # 10mじゃないと埋められない
                dem10m_tile = db_10m["tiles"].get(code)
                if dem10m_tile is not None and dem10m_tile["status"] == "CONVERTED":
                    # エラーなく入れ替え可能
                    tile["npy_path"] = dem10m_tile["npy_path"]
                    tile["data_count"] = 225 * 150
                    tile["status"] = "FILLED"
                else:
                    raise ValueError("NO FILL DATA FOUND")
        elif tile["status"] == "MISSING":
            # 今回は10m一択
            dem10m_tile = db_10m["tiles"].get(code)
            if dem10m_tile is not None and dem10m_tile["status"] == "CONVERTED":
                # エラーなく入れ替え可能
                tile["npy_path"] = dem10m_tile["npy_path"]
                tile["data_count"] = 225 * 150
                tile["status"] = "FILLED"
            else:
                raise ValueError("NO FILL DATA FOUND")
        else:
            continue

        with open(DB_PATH, "wt") as f:
            json.dump(db, f, indent=2)
    
    visualize_db(DB_PATH, VIS_PATH)
    print("fill done")


if __name__ == "__main__":
    print("start")
    dem653726_1m()
    print("1m done")
    dem653726_10m()
    print("10m done")
    # with open("/Users/fukaya/Documents/pythons/dem_proj/output/543726_10m/543726.npy","rb") as f:
    #     visualize_dem(f, "/Users/fukaya/Documents/pythons/dem_proj/output/543726_10.png",False,printout=True)
    dem_resample_2x()
    dem_downsample_5x()
    dem_filler()