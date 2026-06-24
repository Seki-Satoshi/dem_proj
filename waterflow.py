import xml.etree.ElementTree as ET
import numpy as np
import cv2

from dem_utils import tupleList2tuple

# dem data
DEM_PATH = "/Users/fukaya/Documents/20260422141454327-001/DEMDATAS/FG-GML-5438-01-14-DEM5A-20250819.xml"
# XML reading
tree = ET.parse(DEM_PATH)
root = tree.getroot()
ns = {
    "gml": 'http://www.opengis.net/gml/3.2',
    "dem": 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
}
tuples = root.find('.//gml:tupleList', ns).text
lines_strip = tupleList2tuple(tuples)
code_list = tupleList2tuple(tuples, True)

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
    lines_strip = lines_strip + tuple([-9999.0]) * abs( width*height - len(lines_strip) )
    code_list = code_list + tuple(["データなし"]) * abs( width*height - len(code_list) )

# convert tuple to face
composed_data = np.array(lines_strip).reshape(height, width)
code_data = np.array(code_list).reshape(height, width)

# list[list[float]] to ndarray(float64)
dem = np.array(composed_data, dtype=np.float64)
mapping, _ = np.unique(code_data, return_inverse=True)
codes = _.reshape(code_data.shape)
print(mapping)
print(codes)
idx = np.where(mapping == "内水面")[0][0] # (array([0]),)が返ってくる
dem = np.where(codes == idx, dem - 3, dem)

h, w = dem.shape
print(f"{h=}, {w=}")
# exit(0)

# water tile
wtile = np.full_like(dem, 0.05, dtype=np.float64)
print(wtile)
# exit(0)

# direction tile
dtile = np.zeros(dem.shape+(8,), dtype=np.float64)
directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
"D8 method"
for y in range(h):
    for x in range(w):
        elev = dem[y, x]
        d8 = [float(0)]*8
        for i,d in enumerate(directions):
            ax, ay = x+d[0], y+d[1]
            if (ax<0 or w<=ax) or (ay<0 or h<=ay):
                d8[i] = float('inf')
            else:
                d8[i] = dem[ay, ax] - elev
        dtile[y, x, np.argmin(d8)] = 1.0
print(dtile)
# exit(0)

# flow speed
"manning+compress"
def fs2():
    global wtile

    newflow = np.copy(wtile)
    surface = dem + wtile
    for y in range(h):
        for x in range(w):
            i = np.argmax(dtile[y, x])
            ax, ay = x+directions[i][0], y+directions[i][1]
            if ax<0 or w<=ax or ay<0 or h<=ay:
                continue
            hi = wtile[y, x]
            dS = surface[y, x] - surface[ay, ax]
            dx = 5.0 if i%2==0 else 7.0710678118654755
            dQ = (2/3*hi)*np.cbrt(np.square(5*hi/(2*hi+5)))*np.sqrt(dS/dx)
            ftile = dQ if dQ<=hi else hi
            newflow[y, x] -= ftile
            newflow[ay, ax] += ftile
    wtile = newflow
def fs3():
    global wtile

    newflow = np.copy(wtile)
    outflow = np.zeros((h, w), dtype="float64")
    inflow  = np.zeros_like(dtile, dtype="float64")
    surface = np.copy(dem + wtile)
    sur_pad = np.pad(surface, 1, constant_values=0) 

    for i, (dx, dy) in enumerate(directions):
        hi = np.copy(wtile)
        dz = 5.0 if i%2==0 else 7.0710678118654755
        dS = surface - sur_pad[1+dy:h+1+dy, 1+dx:w+1+dx]
        dS = np.where(dS>=0, dS, 0)
        dQ = (2/3*hi)*np.cbrt(np.square(5*hi/(2*hi+5)))*np.sqrt(dS/dz)
        dQ = np.minimum(dQ, hi)
        dQ = np.where(dQ>=0, dQ, 0)
        dQ = dQ * dtile[:, :, i]

        outflow += dQ
        inflow[:, :, (i+4)%8] += dQ
        # inflow_pad = np.pad(dQ, 1)
        # inflow += inflow_pad[1-dy:h+1-dy, 1-dx:w+1-dx]
    newflow = wtile - outflow + np.sum(inflow, axis=2)
    wtile = newflow
# exit(0)

# generate picture
def water_imager(water: np.ndarray, save_path: str):

    mask = water >= 0
    w_min = water[mask].min()
    w_max = water[mask].max()
    ratio = water / 0.25 if w_max>0 else np.zeros_like(water, dtype=np.float64)

    r = np.zeros_like(ratio, dtype=np.uint8)
    g = np.zeros_like(ratio, dtype=np.uint8)
    b = np.full_like(ratio, 139, dtype=np.uint8)

    r[ratio <= 0.8] = 65;  g[ratio <= 0.8] = 105; b[ratio <= 0.8] = 225
    r[ratio <= 0.6] = 100; g[ratio <= 0.6] = 149; b[ratio <= 0.6] = 237
    r[ratio <= 0.4] = 173; g[ratio <= 0.4] = 216; b[ratio <= 0.4] = 230
    r[ratio <= 0.2] = 240; g[ratio <= 0.2] = 248; b[ratio <= 0.2] = 255

    BlueGR = np.stack([b, g, r], axis=2)

    cv2.imwrite(save_path, BlueGR)

OUTPUT_DIR = "/Users/fukaya/Documents/pythons/dem_proj/output/images"
# calc start
for step in range(5000+1):
    fs3()
    if step%50 == 0:
        print(f"{step*0.1:.1f}sec")
        water_imager(wtile, OUTPUT_DIR+f"/step_{step:05d}.png")
        print(f"saved in {step=}")