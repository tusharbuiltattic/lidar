import io, os, csv, math, tempfile
from dataclasses import dataclass
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cv2
import svgwrite
import pydeck as pdk

# ---------------- App config ----------------
st.set_page_config(page_title="LiDAR → Floor Plan", layout="wide")
TITLE = "LiDAR → 2D Floor-Plan Extractor"
MAX_POINTS = 1_000_000
SAMPLE_FOR_VIEW = 120_000
DEFAULT_RES_M = 0.03
MIN_BAND_PTS = 400   # auto-relax target

# Optional readers
try:
    from plyfile import PlyData
    HAS_PLY = True
except Exception:
    HAS_PLY = False
try:
    from pypcd import pypcd
    HAS_PCD = True
except Exception:
    HAS_PCD = False

# ---------------- Types ----------------
@dataclass
class GridMeta:
    origin_xy: np.ndarray
    res: float
    shape: tuple  # (H, W)

# ---------------- IO helpers ----------------
def _to_tempfile(uploaded_file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getvalue())
        return f.name

def read_pcd_ascii(f):
    header = []
    while True:
        line = f.readline()
        if not line: raise ValueError("Invalid PCD header.")
        header.append(line.strip())
        if line.strip().lower().startswith("data"): break
    fields = None
    for h in header:
        if h.lower().startswith("fields"):
            fields = h.split()[1:]
    if not fields or not set(["x","y","z"]).issubset(set(fields)):
        raise ValueError("PCD fields must include x y z.")
    ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
    pts = []
    for line in f:
        if not line.strip(): continue
        p = line.strip().split()
        pts.append([float(p[ix]), float(p[iy]), float(p[iz])])
    return np.asarray(pts, dtype=np.float32)

def load_cloud(uploaded_file):
    name = uploaded_file.name.lower()
    buf = uploaded_file.getvalue()
    suffix = os.path.splitext(name)[1]

    if name.endswith(".ply"):
        if not HAS_PLY: st.error("Install `plyfile`."); st.stop()
        ply = PlyData.read(io.BytesIO(buf))
        v = ply["vertex"].data
        return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    if name.endswith(".pcd"):
        if HAS_PCD:
            path = _to_tempfile(uploaded_file, suffix)
            try:
                pc = pypcd.PointCloud.from_path(path)
                x = pc.pc_data["x"].astype(np.float32)
                y = pc.pc_data["y"].astype(np.float32)
                z = pc.pc_data["z"].astype(np.float32)
                return np.vstack([x, y, z]).T
            except Exception as e:
                try:
                    return read_pcd_ascii(io.StringIO(buf.decode("utf-8")))
                except Exception:
                    st.error(f"PCD read failed: {e}"); st.stop()
        else:
            try:
                return read_pcd_ascii(io.StringIO(buf.decode("utf-8")))
            except Exception as e:
                st.error(f"PCD reader not available. Error: {e}"); st.stop()

    if name.endswith(".npz"):
        arr = np.load(io.BytesIO(buf))
        key = "points" if "points" in arr else list(arr.keys())[0]
        return arr[key].astype(np.float32)

    if name.endswith((".csv", ".txt")):
        s = buf.decode("utf-8", errors="ignore")
        rows = []
        for r in csv.reader(io.StringIO(s)):
            if len(r) < 3: continue
            try: rows.append([float(r[0]), float(r[1]), float(r[2])])
            except ValueError: continue
        if not rows: st.error("CSV/TXT needs x,y,z columns."); st.stop()
        return np.asarray(rows, dtype=np.float32)

    st.error("Use .ply, .pcd, .npz, .csv, or .txt."); st.stop()

# ---------------- Demo generators ----------------
def rect_outline(cx, cy, w, h, zmin=0.1, zmax=2.4, n=2000, noise=0.004, rng=None):
    rng = rng or np.random.RandomState(0)
    # perimeter sampling
    t = np.linspace(0, 1, n, endpoint=False)
    q = n // 4
    x = np.empty_like(t); y = np.empty_like(t)
    x[:q] = cx - w/2 + t[:q]*w;          y[:q] = cy - h/2
    x[q:2*q] = cx + w/2;                 y[q:2*q] = cy - h/2 + (t[:q])*h
    x[2*q:3*q] = cx + w/2 - (t[:q])*w;   y[2*q:3*q] = cy + h/2
    x[3*q:] = cx - w/2;                  y[3*q:] = cy + h/2 - (t[:n-3*q])*h
    z = rng.uniform(zmin, zmax, size=n)
    xy = np.stack([x, y], axis=1) + rng.normal(0, noise, size=(n,2))
    return np.column_stack([xy, z])

def add_door_gap(points, cx, cy, w, h, side="S", gap=0.9):
    dc = {"S":(cx,cy-h/2), "N":(cx,cy+h/2), "W":(cx-w/2,cy), "E":(cx+w/2,cy)}[side]
    rad = gap/2
    if side in ["S","N"]:
        keep = ~((np.abs(points[:,0]-dc[0])<rad) & (np.abs(points[:,1]-dc[1])<0.12))
    else:
        keep = ~((np.abs(points[:,1]-dc[1])<rad) & (np.abs(points[:,0]-dc[0])<0.12))
    return points[keep]

def demo_scene(name="2BHK", seed=42):
    rng = np.random.RandomState(seed)
    clouds = []
    if name == "Studio":
        cx, cy, w, h = 0.0, 0.0, 6.0, 5.0
        R = rect_outline(cx, cy, w, h, rng=rng)
        R = add_door_gap(R, cx, cy, w, h, side="S", gap=0.9)
        clouds.append(R)
    elif name == "2BHK":
        specs = [(-2.5, 0.0, 4.5, 3.8, "S"),
                 ( 2.8, 0.2, 4.7, 3.6, "E"),
                 ( 0.2, 3.7, 10.2, 4.0, "N"),
                 ( 0.0,-3.2, 10.2, 2.0, "S")]
        for (cx,cy,w,h,side) in specs:
            R = rect_outline(cx, cy, w, h, rng=rng)
            R = add_door_gap(R, cx, cy, w, h, side=side, gap=0.9)
            clouds.append(R)
    else:  # Corridor network
        specs = [(0.0, 0.0, 18.0, 2.5, "E"),
                 (0.0, 4.0, 18.0, 2.5, "W"),
                 (-8.5, 2.0, 2.5, 7.0, "S"),
                 ( 8.5, 2.0, 2.5, 7.0, "N")]
        for (cx,cy,w,h,side) in specs:
            R = rect_outline(cx, cy, w, h, rng=rng, noise=0.006)
            R = add_door_gap(R, cx, cy, w, h, side=side, gap=1.0)
            clouds.append(R)
    return np.concatenate(clouds, axis=0)

# ---------------- Geometry ----------------
def voxel_downsample(pts, leaf=0.03):
    if pts.size == 0: return pts
    grid = np.floor(pts[:, :3] / leaf).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return pts[idx]

def statistical_outlier_fast(pts, cell=0.06, min_pts=3):
    if pts.size == 0: return pts
    ijk = np.floor(pts[:, :3] / cell).astype(np.int64)
    keys = ijk[:,0]*73856093 ^ ijk[:,1]*19349663 ^ ijk[:,2]*83492791
    uniq, counts = np.unique(keys, return_counts=True)
    dense = set(uniq[counts >= min_pts])
    keep = np.fromiter((k in dense for k in keys), dtype=bool, count=len(keys))
    return pts[keep]

def ransac_ground(pts, thresh=0.02, iters=600):
    if pts.shape[0] < 100: return np.empty((0,3)), pts
    rng = np.random.default_rng(0)
    best_inliers = None; best_count = -1
    for _ in range(iters):
        ids = rng.choice(pts.shape[0], size=3, replace=False)
        p0,p1,p2 = pts[ids]
        n = np.cross(p1 - p0, p2 - p0); n_norm = np.linalg.norm(n)
        if n_norm < 1e-6: continue
        n = n / n_norm; d = -np.dot(n, p0)
        dist = np.abs(pts.dot(n) + d)
        inliers = dist < thresh; c = int(inliers.sum())
        if c > best_count: best_count = c; best_inliers = inliers
    if best_inliers is None: return np.empty((0,3)), pts
    return pts[best_inliers], pts[~best_inliers]

def height_band(pts, zmin=0.1, zmax=2.4):
    m = (pts[:,2] >= zmin) & (pts[:,2] <= zmax)
    return pts[m]

def to_grid(xy, res=DEFAULT_RES_M, pad_ratio=0.02):
    min_xy = xy.min(axis=0); max_xy = xy.max(axis=0)
    pad = np.maximum((max_xy - min_xy) * pad_ratio, [0.2, 0.2])
    min_xy -= pad; max_xy += pad
    size = np.ceil((max_xy - min_xy)/res).astype(int) + 1
    size = np.maximum(size, 16)
    H, W = int(size[1]), int(size[0])
    max_side = 8000
    H = min(H, max_side); W = min(W, max_side)
    grid = np.zeros((H, W), np.uint8)
    ij = ((xy - min_xy)/res).astype(int)
    ij[:,0] = np.clip(ij[:,0], 0, W-1); ij[:,1] = np.clip(ij[:,1], 0, H-1)
    grid[ij[:,1], ij[:,0]] = 255
    return grid, GridMeta(origin_xy=min_xy, res=float(res), shape=(H, W))

def wall_mask_from_points(grid, dil=2):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, dil), max(1, dil)))
    wall = cv2.dilate(grid, k, iterations=1)
    return cv2.threshold(wall, 128, 255, cv2.THRESH_BINARY)[1]

def extract_lines(wall_mask, min_len_px=60, theta_res=np.pi/180, rho_res=1, thresh=80):
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=rho_res, theta=theta_res, threshold=thresh,
                            minLineLength=min_len_px, maxLineGap=8)
    out = []
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2 = map(int, l)
            out.append((x1,y1,x2,y2))
    return out, edges

def merge_collinear(lines, angle_tol_deg=5):
    if not lines: return []
    def angle(l): x1,y1,x2,y2=l; return math.degrees(math.atan2(y2-y1, x2-x1)) % 180
    used = [False]*len(lines); merged=[]
    for i,l in enumerate(lines):
        if used[i]: continue
        ag = angle(l); pts=[(l[0],l[1]),(l[2],l[3])]; used[i]=True
        for j,m in enumerate(lines):
            if used[j]: continue
            am = angle(m)
            if min(abs(am-ag), abs(abs(am-ag)-180)) <= angle_tol_deg:
                pts += [(m[0],m[1]),(m[2],m[3])]; used[j]=True
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        if abs(math.sin(math.radians(ag))) < 0.5:
            ymed=int(np.median(ys)); merged.append((min(xs), ymed, max(xs), ymed))
        else:
            xmed=int(np.median(xs)); merged.append((xmed, min(ys), xmed, max(ys)))
    return merged

def detect_doors(wall_mask, res, default_m=(0.8, 1.2)):
    door_min_px = max(10, int(default_m[0] / res))
    door_max_px = min(240, int(default_m[1] / res))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    band = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k, iterations=1)
    inv = 255 - band
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    door_core = (dist > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(door_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doors=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c); span=max(w,h)
        if door_min_px <= span <= door_max_px:
            doors.append((x + w//2, y + h//2, span))
    return doors

def overlay(wall_mask, lines, doors):
    color = np.dstack([wall_mask]*3)
    for (x1,y1,x2,y2) in lines:
        cv2.line(color, (x1,y1), (x2,y2), (0,255,0), 2)
    for (cx,cy,span) in doors:
        cv2.circle(color, (cx,cy), max(2, span//6), (0,0,255), 2)
    return color

def px_to_m(p, meta: GridMeta):
    x_px, y_px = p
    return meta.origin_xy[0] + x_px*meta.res, meta.origin_xy[1] + y_px*meta.res

def export_svg(lines, doors, meta: GridMeta):
    H, W = meta.shape
    vb_w, vb_h = W*meta.res, H*meta.res
    dwg = svgwrite.Drawing(size=("1000px","1000px"), viewBox=f"0 0 {vb_w:.3f} {vb_h:.3f}")
    dwg.add(dwg.rect(insert=(0,0), size=(vb_w, vb_h), fill="white"))
    layer_walls = dwg.add(dwg.g(id="WALLS", stroke="black", fill="none", stroke_width=0.02))
    for (x1,y1,x2,y2) in lines:
        x1m,y1m = px_to_m((x1,y1), meta); x2m,y2m = px_to_m((x2,y2), meta)
        y1m = meta.origin_xy[1] + (meta.shape[0]-y1)*meta.res
        y2m = meta.origin_xy[1] + (meta.shape[0]-y2)*meta.res
        layer_walls.add(dwg.line(start=(x1m,y1m), end=(x2m,y2m)))
    layer_doors = dwg.add(dwg.g(id="DOORS", stroke="red", fill="none", stroke_width=0.02))
    for (cx,cy,span) in doors:
        xm, ym = px_to_m((cx,cy), meta)
        ym = meta.origin_xy[1] + (meta.shape[0]-cy)*meta.res
        r = max(0.05, span*meta.res*0.3)
        layer_doors.add(dwg.circle(center=(xm, ym), r=r))
    return dwg.tostring().encode("utf-8")

# ---------------- UI ----------------
st.title(TITLE)
with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Upload", "Demo"], index=1)
    if mode == "Upload":
        up = st.file_uploader("Point cloud", type=["ply","pcd","npz","csv","txt"])
    else:
        demo_name = st.selectbox("Demo scene", ["Studio", "2BHK", "Corridor network"], index=1)
        demo_seed = st.number_input("Seed", 0, 9999, 42)

    st.header("Preprocess")
    voxel = st.slider("Voxel (m)", 0.005, 0.10, 0.03, 0.005)
    density_cell = st.slider("Outlier cell (m)", 0.02, 0.20, 0.06, 0.01)
    density_min = st.slider("Min pts/cell", 1, 10, 3, 1)
    zmin = st.number_input("z_min (m)", 0.0, 1.0, 0.10, 0.05)
    zmax = st.number_input("z_max (m)", 1.5, 4.0, 2.40, 0.05)
    auto_relax = st.checkbox("Auto-relax if too sparse", value=True)

    st.header("Grid / Vectorization")
    res = st.slider("Grid res (m/px)", 0.01, 0.10, DEFAULT_RES_M, 0.01)
    dil = st.slider("Wall dilation (px)", 1, 7, 2, 1)
    min_len = st.slider("Min line length (px)", 10, 200, 60, 5)
    hough_thresh = st.slider("Hough threshold", 10, 200, 80, 5)
    angle_merge = st.slider("Merge angle tol (deg)", 1, 15, 5, 1)

    st.header("3D View")
    engine = st.selectbox("Engine", ["Plotly", "PyDeck"], index=0)
    point_size = st.slider("Point size", 1, 8, 2, 1)
    color_mode = st.selectbox("Color by", ["height(z)", "density"], index=0)

    run = st.button("Run pipeline", type="primary")

# ---------------- Pipeline ----------------
if mode == "Upload" and not run:
    st.info("Upload a file and click Run.")
elif mode == "Demo" and not run:
    st.info("Pick a demo and click Run.")

if run:
    with st.spinner("Preparing data"):
        pts = load_cloud(up) if mode == "Upload" else demo_scene(demo_name, seed=int(demo_seed))
        if pts.ndim != 2 or pts.shape[1] < 3: st.error("Expect Nx3 points [x,y,z]."); st.stop()
        orig_n = len(pts)
        if orig_n > MAX_POINTS:
            step = int(np.ceil(orig_n / MAX_POINTS)); pts = pts[::step]

        # preprocess
        pts = voxel_downsample(pts, leaf=float(voxel))
        pts = statistical_outlier_fast(pts, cell=float(density_cell), min_pts=int(density_min))
        _, rest = ransac_ground(pts, thresh=0.02, iters=600)
        band = height_band(rest, zmin=float(zmin), zmax=float(zmax))

        # auto relax if too sparse
        if auto_relax and band.shape[0] < MIN_BAND_PTS:
            zmin2 = max(0.0, zmin - 0.05)
            zmax2 = zmax + 0.3
            pts2 = voxel_downsample(pts, leaf=float(voxel)*1.2)
            pts2 = statistical_outlier_fast(pts2, cell=float(density_cell)*1.2, min_pts=max(1, int(density_min)-1))
            _, rest2 = ransac_ground(pts2, thresh=0.03, iters=400)
            band2 = height_band(rest2, zmin=zmin2, zmax=zmax2)
            if band2.shape[0] > band.shape[0]:
                band = band2

        if band.shape[0] < 50:
            st.error("Too few points after filtering. Loosen filters."); st.stop()

        xy = band[:, :2]
        grid, meta = to_grid(xy, res=float(res))
        wall = wall_mask_from_points(grid, dil=int(dil))
        lines_raw, _ = extract_lines(wall, min_len_px=int(min_len), theta_res=np.pi/180, rho_res=1, thresh=int(hough_thresh))
        lines = merge_collinear(lines_raw, angle_tol_deg=int(angle_merge))
        doors = detect_doors(wall, res=meta.res)
        overlay_img = overlay(wall, lines, doors)

    # --------------- Visuals ---------------
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("3D view")
        # sample for speed
        samp = band[np.random.choice(len(band), size=min(SAMPLE_FOR_VIEW, len(band)), replace=False)]
        if color_mode == "height(z)":
            cvals = samp[:,2]
        else:
            # density by 2D binning
            xy_ = samp[:,:2]
            res_d = max(DEFAULT_RES_M, float(res))
            ij = np.floor((xy_ - xy_.min(0))/res_d).astype(int)
            keys = ij[:,0]*73856093 ^ ij[:,1]*19349663
            _, inv, cnts = np.unique(keys, return_inverse=True, return_counts=True)
            cvals = cnts[inv]

        if engine == "Plotly":
            fig = go.Figure(data=[go.Scatter3d(
                x=samp[:,0], y=samp[:,1], z=samp[:,2],
                mode='markers',
                marker=dict(size=point_size, color=cvals, colorscale='Viridis', opacity=0.9, colorbar=dict(len=0.6))
            )])
            fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z',
                                         aspectmode='data'),
                              margin=dict(l=0,r=0,t=0,b=0), height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # PyDeck PointCloudLayer with z-based color
            z = cvals.astype(float)
            zmin_, zmax_ = float(z.min()), float(z.max()) if z.size else (0.0, 1.0)
            if zmax_ == zmin_: zmax_ = zmin_ + 1.0
            zn = (z - zmin_) / (zmax_ - zmin_ + 1e-9)
            colors = np.stack([(zn*255).astype(int), (1-zn)*255, np.full_like(zn, 180, dtype=int)], axis=1)
            data = [{"position":[float(x),float(y),float(zv)], "color":[int(r),int(g),int(b)]}
                    for (x,y,zv),(r,g,b) in zip(samp, colors)]
            layer = pdk.Layer("PointCloudLayer", data=data, get_position="position",
                              get_color="color", point_size=point_size, pickable=False)
            deck = pdk.Deck(layers=[layer],
                            initial_view_state=pdk.ViewState(x=0, y=0, z=0, zoom=0.5, pitch=45),
                            map_provider=None)
            st.pydeck_chart(deck, use_container_width=True)

        st.caption(f"Loaded: {orig_n:,} pts → kept: {pts.shape[0]:,}. Band used: {band.shape[0]:,}.")

    with col2:
        st.subheader("2D layers")
        st.image(wall, caption="Wall mask", use_container_width=True, clamp=True)
        st.image(overlay_img, caption="Vector overlay (green=walls, red≈doors)", use_container_width=True, clamp=True)

    # --------------- Metrics ---------------
    st.subheader("Metrics")
    st.write({
        "grid_size_px": meta.shape,
        "grid_resolution_m_per_px": meta.res,
        "lines_detected": len(lines),
        "door_candidates": len(doors),
    })

    # --------------- Exports ---------------
    svg_bytes = export_svg(lines, doors, meta)
    st.download_button("Download SVG plan", data=svg_bytes, file_name="plan.svg", mime="image/svg+xml")
    ok, png_bytes = cv2.imencode(".png", wall)
    if ok:
        st.download_button("Download wall mask PNG", data=png_bytes.tobytes(), file_name="wall_mask.png", mime="image/png")

    # lines CSV
    rows = ["x1_m,y1_m,x2_m,y2_m"]
    for (x1,y1,x2,y2) in lines:
        x1m,y1m = px_to_m((x1,y1), meta); x2m,y2m = px_to_m((x2,y2), meta)
        y1m = meta.origin_xy[1] + (meta.shape[0]-y1)*meta.res
        y2m = meta.origin_xy[1] + (meta.shape[0]-y2)*meta.res
        rows.append(f"{x1m:.3f},{y1m:.3f},{x2m:.3f},{y2m:.3f}")
    st.download_button("Download wall lines CSV", data=("\n".join(rows)).encode("utf-8"),
                       file_name="walls.csv", mime="text/csv")

    st.success("Done")
