import numpy as np

# dem data
DEM_PATH = "/Users/fukaya/Documents/pythons/dem_proj/output/tiles/54380114.npy"
dem = np.array(np.load(DEM_PATH), dtype="float64")
h, w = dem.shape
print(f"{h=}, {w=}")
# exit(0)

# water tile
wtile = np.full(dem.shape, 0.05, dtype="float64")
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
exit(0)

# flow speed
"manning+numpynature"
def fs():
    ftile = np.zeros_like(dtile, dtype="float64")
    for y in range(h):
        for x in range(w):
            d8 = dtile[y, x]
            for i,flows in enumerate(d8):
                if flows == 0:
                    continue
                ax, ay = x+directions[i][0], y+directions[i][1]
                hi = np.float64(wtile[y, x])
                dx = np.float64(np.linalg.norm([ax-x, ay-y]))
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
    inflow = np.zeros_like(dem, dtype="float64")
    outflow = np.zeros_like(dem, dtype="float64")
    for y in range(h):
        for x in range(w):
            d8 = dtile[y, x]
            ftile = [float(0)]*8
            for i,flows in enumerate(d8):
                if flows == 0:
                    continue
                ax, ay = x+directions[i][0], y+directions[i][1]
                hi = np.float64(wtile[y, x])
                dx = np.float64(np.linalg.norm([ax-x, ay-y]))
                z1 = np.float64(dem[y, x])
                z2 = np.float64(dem[ay, ax])
                dh = (10/75)*np.cbrt(np.pow(hi*dx,5)/(4*hi*hi + 4*hi*dx + dx*dx))*np.sqrt((z1-z2)/dx)
                ftile[i] = dh*flows
            
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


def brange(end: int):
    if not isinstance(end, int):
        raise TypeError("end needs to be integer")
    if end<1:
        raise ValueError("end must be greater than 0")
    return range(1, end+1)
OUTPUT_DIR = "/Users/fukaya/Documents/pythons/dem_proj/output/images"
# calc start
for step in brange(5000):
    print(f"{step*0.1}sec")
    fs()