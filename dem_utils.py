import os
import copy
import xml.etree.ElementTree as ET
import numpy as np
from typing import BinaryIO, TextIO


def mesh_to_xy(mesh_code: int):
    s = str(mesh_code)
    p  = int(s[0:2])
    u  = int(s[2:4])
    r2 = int(s[4])
    c2 = int(s[5])
    r3 = int(s[6])
    c3 = int(s[7])
    
    x = u * 80 + c2 * 10 + c3
    y = p * 80 + r2 * 10 + r3
    return x, y

def xy_to_mesh(x: int, y: int):
    p  = y // 80
    r2 = (y%80) // 10
    r3 = y % 10

    
    u  = x // 80
    c2 = (x%80) // 10
    c3 = x % 10

    return p * 1000000 + u * 10000 + r2 * 1000 + c2 * 100 + r3 * 10 + c3

def check_missing(dem_list: list[int], offset_lower: int, offset_upper: int):
    x0, y0 = mesh_to_xy(offset_lower)
    xm, ym = mesh_to_xy(offset_upper)
    xl, yl = xm-x0+1, ym-y0+1

    matrix = [[False] * xl for _ in range(yl)]

    for code in dem_list:
        x, y = mesh_to_xy(code)
        matrix[y - y0][x - x0] = True
    
    missing = []
    for y in range(yl):
        for x in range(xl):
            if not matrix[y][x]:
                missing.append((x,y))
    
    print(f"範囲： {xl} x {yl} = {xl*yl}セル")
    print(f"取得数： {sum(row.count(True) for row in matrix)}")
    if missing:
        print(f"欠損 {len(missing)} 件:")
        for x, y in missing:
            print(f"　ローカル({x}, {y}) → メッシュ {xy_to_mesh(x+x0, y+y0)}")
    else:
        print("欠損なし ✔︎")

    return matrix, (not bool(missing))


# dem_selector
DEM_DIR = "/Users/fukaya/Documents/20260422141454327-001/DEMDATAS"

def get_meshcode(dem_path: str):
    with open(dem_path, "rt") as fxml:
        root = ET.parse(fxml).getroot()
    ns = {
        "gml": 'http://www.opengis.net/gml/3.2',
        "dem": 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
    }
    return int(root.find(".//dem:mesh", ns).text)

def offset(dem_dir:str) -> tuple[int, int]:
    dem_list = sorted(os.listdir(dem_dir))
    offset_lower = get_meshcode(os.path.join(dem_dir,dem_list[0])) # 0,0
    offset_upper = get_meshcode(os.path.join(dem_dir,dem_list[-1]))
    return offset_lower, offset_upper

def dem_path_generator(dem_dir:str) -> list[tuple[int, int, int, str]]:
    # それぞれのファイル名から位置を割り出し、{id,x,y,path}のリストを返す
    dem_list = sorted(os.listdir(dem_dir))

    x0, y0 = mesh_to_xy(offset(dem_dir)[0])

    if dem_list[0].find("DEM5")!=-1:
        dem_path = [
            (
                int(d[7:d.find("DEM5")-1].replace("-","")),
                mesh_to_xy(int(d[7:d.find("DEM5")-1].replace("-","")))[0]-x0,
                mesh_to_xy(int(d[7:d.find("DEM5")-1].replace("-","")))[1]-y0,
                os.path.join(dem_dir, d)
            ) for d in dem_list
        ]
    else:
        dem_path = [
            (
                get_meshcode(os.path.join(dem_dir, d)),
                mesh_to_xy(get_meshcode(os.path.join(dem_dir, d)))[0] - x0,
                mesh_to_xy(get_meshcode(os.path.join(dem_dir, d)))[1] - y0,
                os.path.join(dem_dir, d)
            ) for d in dem_list
        ]

    return dem_path


# dem_xml2npy
class InvalidDataException(IndexError):
    def __init__(self, *args):
        super().__init__(*args)

def tupleList2tuple(tupleList: str, getcode: bool = False):
    lines = tupleList.split("\n")
    if getcode:
        lines_strip = [str(l.strip()[:l.strip().find(',')]) for l in lines if l!='']
    else:
        lines_strip = [float(l.strip()[l.strip().find(',')+1:]) for l in lines if l!='']
    return tuple(lines_strip)

def xml2npy(ifp: TextIO, ofp: BinaryIO, filling:bool=False):
    if not ifp.readable():
        raise IOError("ifp is not readable")
    if not ofp.writable():
        raise IOError("ofp is not writable")

    # XML reading
    tree = ET.parse(ifp)
    root = tree.getroot()
    ns = {
        "gml": 'http://www.opengis.net/gml/3.2',
        "dem": 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
    }
    tuples = root.find('.//gml:tupleList', ns).text
    lines_strip = tupleList2tuple(tuples)

    # get width and height
    envelope = root.find('.//gml:GridEnvelope',ns)
    low = envelope.find('gml:low', ns).text
    high = envelope.find('gml:high', ns).text
    low_x, low_y = map(int, low.split())
    high_x, high_y = map(int, high.split())
    width = high_x - low_x + 1
    height = high_y - low_y + 1

    # check data size
    if len(lines_strip) != width*height:
        if not filling:
            raise InvalidDataException(f"{len(lines_strip)} != {width*height}")
        else:
            lines_strip = lines_strip + tuple([-9999.0]) * abs( width*height - len(lines_strip) )

    # convert tuple to face
    initializer = [0.0]*width
    composed_data = [copy.deepcopy(initializer) for _ in range(height)]
    for y in range(height):
        for x in range(width):
            composed_data[y][x] = float(lines_strip[x + y*width])

    # list[list[float]] to ndarray(float)
    dem_data = np.array(composed_data, dtype=np.float64)

    # save to ofp
    np.save(ofp, dem_data)

    return width*height

# resampler
def upsample_2x_nearest(arr: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
def upsample_2x_bilinear(arr: np.ndarray) -> np.ndarray:
    H, W = arr.shape
    out = np.empty((2*H, 2*W), dtype=float)

    # 偶数行・偶数列
    out[0::2, 0::2] = arr

    # 偶数行・奇数列
    out[0::2, 1:-1:2] = (arr[:, :-1] + arr[:, 1:]) /2
    out[0::2, -1] = arr[:, -1]

    # 奇数行・偶数列
    out[1:-1:2, 0::2] = (arr[:-1, :] + arr[1:, :]) /2
    out[-1, 0::2] = arr[-1, :]

    # 奇数行・奇数列
    out[1:-1:2, 1:-1:2] = (arr[:-1, :-1] + arr[:-1, 1:] + arr[1:, :-1] + arr[1:, 1:]) /4
    out[1:-1:2, -1] = (arr[:-1, -1] + arr[1:, -1]) /2
    out[-1, 1:-1:2] = (arr[-1, :-1] + arr[-1, 1:]) /2
    out[-1, -1] = arr[-1, -1]

    return out