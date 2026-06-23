import numpy as np
import cv2

# dem data
DEM_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/tiles/54380114.npy"
dem = np.array(np.load(DEM_PATH), dtype="float64")
h, w = dem.shape
print(f"{h=}, {w=}")
# exit(0)

# water tile
wtile = np.full_like(dem, 0.05, dtype="float64")
print(wtile)
# exit(0)

# direction tile
dtile = np.zeros(dem.shape+(8,), dtype="float64")
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
"manning+numpynature"
def fs():
    global wtile

    ftile = np.zeros_like(dtile, dtype="float64")
    for y in range(h):
        for x in range(w):
            for i,flows in enumerate(dtile[y, x]):
                if flows == 0:
                    continue
                ax, ay = x+directions[i][0], y+directions[i][1]
                hi = np.float64(wtile[y, x])
                dx = np.float64(np.linalg.norm([(ax-x)*5, (ay-y)*5]))
                z1 = np.float64(dem[y, x])
                z2 = np.float64(dem[ay, ax])
                dh = (10/75)*np.cbrt(np.pow(hi*dx,5)/(4*hi*hi + 4*hi*dx + dx*dx))*np.sqrt((z1-z2)/dx)
                ftile[y, x, i] = dh*flows
    DY = np.array([ 1, 1, 0,-1,-1,-1, 0, 1])
    DX = np.array([ 0, 1, 1, 1, 0,-1,-1,-1])
    flow_pad = np.pad(ftile, ((1,1),(1,1),(0,0)))
    y_idx = np.arange(h)[:, None, None]
    x_idx = np.arange(w)[None, :, None]
    d_idx = np.arange(8)[None, None, :]

    inflow_3d = flow_pad[y_idx+1-DY, x_idx+1-DX, d_idx]

    inflow = inflow_3d.sum(axis=2)
    outflow = ftile.sum(axis=2)

    wtile = wtile - outflow + inflow
"manning+compress"
def fs2():
    global wtile

    inflow = np.zeros_like(dem, dtype=np.float64)
    outflow = np.zeros_like(dem, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            for i,flows in enumerate(dtile[y, x]):
                if flows == 0:
                    continue
                ax, ay = x+directions[i][0], y+directions[i][1]
                hi = np.float64(wtile[y, x])
                dx = np.float64(np.linalg.norm([(ax-x)*5, (ay-y)*5]))
                z1 = np.float64(dem[y, x])
                z2 = np.float64(dem[ay, ax])
                dh = (10/75)*np.cbrt(np.pow(hi*dx,5)/(4*hi*hi + 4*hi*dx + dx*dx))*np.sqrt((z1-z2)/dx)
                ftile = dh*flows
                inflow[ay, ax] += ftile
                outflow[y, x] += ftile
    wtile = wtile - outflow + inflow
"manning+multied"
def fs3():
    global wtile

    ftile = np.zeros_like(dtile, dtype="float64")
    dem_pad = np.pad(dem, 1, constant_values=-9999)
    for i, (dx, dy) in enumerate(directions):
        hi = wtile
        dist = np.float64(np.linalg.norm([dx*5, dy*5]))
        z1 = dem
        z2 = dem_pad[1-dy:h+1-dy, 1-dx:w+1-dx]  # 隣接セルの標高
        dh = (10/75) * np.cbrt(np.pow(hi * dist, 5) / (4*hi*hi + 4*hi*dist + dist*dist)) * np.sqrt(np.maximum(z1 - z2, 0) / dist)
        ftile[:, :, i] = dh * dtile[:, :, i]
    DY = np.array([ 1, 1, 0,-1,-1,-1, 0, 1])
    DX = np.array([ 0, 1, 1, 1, 0,-1,-1,-1])
    flow_pad = np.pad(ftile, ((1,1),(1,1),(0,0)))
    y_idx = np.arange(h)[:, None, None]
    x_idx = np.arange(w)[None, :, None]
    d_idx = np.arange(8)[None, None, :]

    inflow_3d = flow_pad[y_idx+1-DY, x_idx+1-DX, d_idx]

    inflow = inflow_3d.sum(axis=2)
    outflow = ftile.sum(axis=2)

    wtile = wtile - outflow + inflow
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
    print(f"{step*0.1:.1f}sec")
    fs2()
    if step%50 == 0:
        water_imager(wtile, OUTPUT_DIR+f"/step_{step:05d}.png")
        print(f"saved in {step=}")