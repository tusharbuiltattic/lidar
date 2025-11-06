import io, os, csv, math, json, tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import cv2
import svgwrite
import pydeck as pdk

# ==================== App config ====================
st.set_page_config(page_title="LiDAR → Floor Plan", layout="wide")
# Avoid creating many inotify watchers on Streamlit Cloud
try:
    st.set_option("server.fileWatcherType", "poll")  # avoid inotify on some platforms
except Exception:
    pass

TITLE = "LiDAR → 2D Floor‑Plan Extractor (v6 • Demo‑only • Auto‑run)"
MAX_POINTS = 1_000_000
SAMPLE_FOR_VIEW = 160_000
DEFAULT_RES_M = 0.03
MIN_BAND_PTS = 400   # auto‑relax target

# ==================== Types ====================
@dataclass
class GridMeta:
    origin_xy: np.ndarray
    res: float
    shape: tuple  # (H, W)
    rotation_deg: float = 0.0  # rotation applied to XY for orthogonal alignment

# ==================== Demo generators ====================
def rect_outline(cx, cy, w, h, zmin=0.1, zmax=2.4, n=2200, noise=0.004, rng=None):
    rng = rng or np.random.RandomState(0)
    # perimeter sampling
    t = np.linspace(0, 1, n, endpoint=False)
    q = n // 4
    x = np.empty_like(t); y = np.empty_like(t)
    x[:q] = cx - w/2 + t[:q]*w;          y[:q] = cy - h/2
    x[q:2*q] = cx + w/2;                 y[q:2*q] = cy - h/2 + (t[:q])*h
    x[2*q:3*q] = cx + w/2 - (t[:q])*w;   y[2*q:3*q] = cy + h/2
    x[3*q:] = cx - w/2;                  y[3*q:] = cy + h/2 - (t[:len(t)-3*q])*h
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

@st.cache_data(show_spinner=False)
def demo_scene(name="2BHK", seed=42, noise_scale=1.0):
    rng = np.random.RandomState(seed)
    clouds = []
    if name == "Studio":
        cx, cy, w, h = 0.0, 0.0, 6.0, 5.0
        R = rect_outline(cx, cy, w, h, rng=rng, noise=0.004*noise_scale)
        R = add_door_gap(R, cx, cy, w, h, side="S", gap=0.9)
        clouds.append(R)
    elif name == "2BHK":
        specs = [(-2.5, 0.0, 4.5, 3.8, "S"),
                 ( 2.8, 0.2, 4.7, 3.6, "E"),
                 ( 0.2, 3.7, 10.2, 4.0, "N"),
                 ( 0.0,-3.2, 10.2, 2.0, "S")]
        for (cx,cy,w,h,side) in specs:
            R = rect_outline(cx, cy, w, h, rng=rng, noise=0.004*noise_scale)
            R = add_door_gap(R, cx, cy, w, h, side=side, gap=0.9)
            clouds.append(R)
    else:  # Corridor network
        specs = [(0.0, 0.0, 18.0, 2.5, "E"),
                 (0.0, 4.0, 18.0, 2.5, "W"),
                 (-8.5, 2.0, 2.5, 7.0, "S"),
                 ( 8.5, 2.0, 2.5, 7.0, "N")]
        for (cx,cy,w,h,side) in specs:
            R = rect_outline(cx, cy, w, h, rng=rng, noise=0.006*noise_scale)
            R = add_door_gap(R, cx, cy, w, h, side=side, gap=1.0)
            clouds.append(R)
    return np.concatenate(clouds, axis=0)

# ==================== Geometry ====================
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

# ----- PCA alignment (orthogonal snap) -----
def pca_align(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rotate XY so the first PC aligns with X axis; returns (rotated_xy, angle_deg)."""
    xy0 = xy - xy.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(xy0, full_matrices=False)
    R = Vt.T  # principal axes as columns
    rot = np.arctan2(R[1,0], R[0,0])  # angle of first PC
    c, s = np.cos(-rot), np.sin(-rot)
    R2 = np.array([[c, -s],[s, c]])
    xy_rot = (xy - xy.mean(0)) @ R2.T
    return xy_rot, np.degrees(rot)

# ----- Grid & vectorization -----
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

def wall_mask_from_points(grid, dil=2, close=1):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, dil), max(1, dil)))
    wall = cv2.dilate(grid, k, iterations=1)
    if close > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k2, iterations=close)
    return cv2.threshold(wall, 128, 255, cv2.THRESH_BINARY)[1]

def extract_lines(wall_mask, min_len_px=60, theta_res=np.pi/180, rho_res=1, thresh=80, max_gap=8):
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=rho_res, theta=theta_res, threshold=thresh,
                            minLineLength=min_len_px, maxLineGap=max_gap)
    out = []
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2 = map(int, l)
            out.append((x1,y1,x2,y2))
    return out, edges

def merge_collinear(lines: List[Tuple[int,int,int,int]], angle_tol_deg=5, snap_ortho=False):
    if not lines: return []
    def angle(l): x1,y1,x2,y2=l; return (math.degrees(math.atan2(y2-y1, x2-x1)) + 180) % 180
    used = [False]*len(lines); merged=[]
    for i,l in enumerate(lines):
        if used[i]: continue
        ag = angle(l)
        if snap_ortho:
            ag = 0 if min(abs(ag-0), abs(ag-180)) < 45 else 90
        pts=[(l[0],l[1]),(l[2],l[3])]; used[i]=True
        for j,m in enumerate(lines):
            if used[j]: continue
            am = angle(m)
            if snap_ortho:
                am = 0 if min(abs(am-0), abs(am-180)) < 45 else 90
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

def overlay(wall_mask, lines, doors, alpha=0.65):
    base = cv2.normalize(wall_mask, None, 0, 255, cv2.NORM_MINMAX)
    color = np.dstack([base, base, base])
    overlay_img = color.copy()
    for (x1,y1,x2,y2) in lines:
        cv2.line(overlay_img, (x1,y1), (x2,y2), (0,255,0), 2)
    for (cx,cy,span) in doors:
        cv2.circle(overlay_img, (cx,cy), max(2, span//6), (0,0,255), 2)
    return cv2.addWeighted(overlay_img, alpha, color, 1-alpha, 0)

def px_to_m(p, meta: GridMeta):
    x_px, y_px = p
    return meta.origin_xy[0] + x_px*meta.res, meta.origin_xy[1] + y_px*meta.res

def export_svg(lines, doors, meta: GridMeta):
    H, W = meta.shape
    vb_w, vb_h = W*meta.res, H*meta.res
    dwg = svgwrite.Drawing(size=("1200px","1200px"), viewBox=f"0 0 {vb_w:.3f} {vb_h:.3f}")
    dwg.add(dwg.rect(insert=(0,0), size=(vb_w, vb_h), fill="white"))
    # Grid/scale layer
    grid_layer = dwg.add(dwg.g(id="GRID", stroke="#ddd", fill="none", stroke_width=0.005))
    step = max(0.5, round(1.0 / max(0.5, int(1.0/meta.res))) )  # simple approx grid step in meters
    x=0
    while x < vb_w:
        grid_layer.add(dwg.line(start=(x,0), end=(x,vb_h)))
        x += step
    y=0
    while y < vb_h:
        grid_layer.add(dwg.line(start=(0,y), end=(vb_w,y)))
        y += step

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

    # scale bar
    sb_len_m = 1.0
    sb_x, sb_y = 0.5, 0.5
    dwg.add(dwg.line(start=(sb_x, sb_y), end=(sb_x+sb_len_m, sb_y), stroke="#111", stroke_width=0.05))
    dwg.add(dwg.text(f"{sb_len_m} m", insert=(sb_x+sb_len_m+0.1, sb_y+0.1), font_size=0.2, fill="#111"))

    return dwg.tostring().encode("utf-8")

# ======= GeoJSON export for walls & doors =======
def export_geojson(lines, doors, meta: GridMeta, rooms: List[Dict]=None) -> Dict:
    feats = []
    for (x1,y1,x2,y2) in lines:
        x1m,y1m = px_to_m((x1,y1), meta); x2m,y2m = px_to_m((x2,y2), meta)
        y1m = meta.origin_xy[1] + (meta.shape[0]-y1)*meta.res
        y2m = meta.origin_xy[1] + (meta.shape[0]-y2)*meta.res
        feats.append({
            "type":"Feature",
            "properties":{"type":"wall"},
            "geometry":{"type":"LineString","coordinates":[[x1m,y1m],[x2m,y2m]]}
        })
    for (cx,cy,span) in doors:
        xm, ym = px_to_m((cx,cy), meta)
        ym = meta.origin_xy[1] + (meta.shape[0]-cy)*meta.res
        feats.append({
            "type":"Feature",
            "properties":{"type":"door","span_m": float(span*meta.res)},
            "geometry":{"type":"Point","coordinates":[xm,ym]}
        })
    if rooms:
        for r in rooms:
            feats.append({
                "type":"Feature",
                "properties":{"type":"room","area_m2": r.get("area_m2", None)},
                "geometry":{"type":"Polygon","coordinates":[r["coords"]]}
            })
    return {"type":"FeatureCollection","features":feats}

# ======= Simple room polygonization (contours of free space) =======
def detect_rooms(wall_mask: np.ndarray, meta: GridMeta, min_area_m2: float = 2.0, approx_eps_px: int = 6) -> List[Dict]:
    inv = 255 - wall_mask
    # fill small holes then find contours
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k, iterations=1)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rooms = []
    for c in contours:
        area_px = cv2.contourArea(c)
        area_m2 = area_px * (meta.res ** 2)
        if area_m2 < min_area_m2:
            continue
        approx = cv2.approxPolyDP(c, epsilon=approx_eps_px, closed=True)
        # Convert to meters and flip Y like in SVG export
        poly = []
        for pt in approx.reshape(-1,2):
            x,y = int(pt[0]), int(pt[1])
            xm, ym = px_to_m((x,y), meta)
            ym = meta.origin_xy[1] + (meta.shape[0]-y)*meta.res
            poly.append([float(xm), float(ym)])
        rooms.append({"area_m2": float(area_m2), "coords": poly})
    return rooms

# ==================== Sidebar (Demo‑only) ====================
st.title(TITLE)
with st.sidebar:
    st.header("Demo Data 🎛️")
    demo_name = st.selectbox("Scene", ["Studio", "2BHK", "Corridor network"], index=1)
    demo_seed = st.number_input("Seed", 0, 9999, 42)
    noise_scale = st.slider("Noise scale", 0.5, 2.0, 1.0, 0.1)

    st.header("Preprocess")
    preset = st.selectbox("Presets", ["Default","WallsDense","Loose"], index=0,
                          help="Quickly set typical filter strengths")
    voxel = st.slider("Voxel (m)", 0.005, 0.10, 0.03 if preset=="Default" else (0.02 if preset=="WallsDense" else 0.05), 0.005)
    density_cell = st.slider("Outlier cell (m)", 0.02, 0.20, 0.06 if preset=="Default" else (0.05 if preset=="WallsDense" else 0.10), 0.01)
    density_min = st.slider("Min pts/cell", 1, 10, 3 if preset=="Default" else (4 if preset=="WallsDense" else 2), 1)
    zmin = st.number_input("z_min (m)", 0.0, 1.0, 0.10, 0.05)
    zmax = st.number_input("z_max (m)", 1.5, 4.0, 2.40, 0.05)
    auto_relax = st.checkbox("Auto‑relax if too sparse", value=True)

    st.header("ROI & Alignment")
    use_percent_roi = st.checkbox("Crop by percentiles", value=False)
    p_lo, p_hi = st.slider("XY keep percentiles", 0, 100, (5,95)) if use_percent_roi else (5,95)
    do_pca_align = st.checkbox("Align to dominant axes (PCA)", value=True)

    st.header("Grid / Vectorization")
    res = st.slider("Grid res (m/px)", 0.01, 0.10, DEFAULT_RES_M, 0.01)
    dil = st.slider("Wall dilation (px)", 1, 7, 2, 1)
    morph_close = st.slider("Morph close iters", 0, 4, 1, 1)
    min_len = st.slider("Min line length (px)", 10, 200, 60, 5)
    max_gap = st.slider("Max line gap (px)", 0, 50, 8, 1)
    hough_thresh = st.slider("Hough threshold", 10, 200, 80, 5)
    angle_merge = st.slider("Merge angle tol (deg)", 1, 15, 5, 1)
    snap_ortho = st.checkbox("Snap merge to 0/90°", value=True)

    st.header("3D View ✨")
    point_size = st.slider("Point size", 1, 10, 3, 1)
    color_mode = st.selectbox("Color by", ["height(z)", "density"], index=0)
    cam_preset = st.selectbox("Camera", ["Isometric","Top","Front","Side"], index=0)
    az = st.slider("Azimuth", -180, 180, 40, 5)
    el = st.slider("Elevation", 0, 90, 35, 5)

# ==================== Auto‑run pipeline (no button) ====================
with st.spinner("Synthesizing demo & extracting plan…"):
    pts = demo_scene(demo_name, seed=int(demo_seed), noise_scale=float(noise_scale))
    orig_n = len(pts)
    if orig_n > MAX_POINTS:
        step = int(np.ceil(orig_n / MAX_POINTS)); pts = pts[::step]

    if use_percent_roi:
        lo = np.percentile(pts[:,:2], p_lo, axis=0)
        hi = np.percentile(pts[:,:2], p_hi, axis=0)
        m = (pts[:,0]>=lo[0]) & (pts[:,0]<=hi[0]) & (pts[:,1]>=lo[1]) & (pts[:,1]<=hi[1])
        pts = pts[m]

    pts = voxel_downsample(pts, leaf=float(voxel))
    pts = statistical_outlier_fast(pts, cell=float(density_cell), min_pts=int(density_min))
    _, rest = ransac_ground(pts, thresh=0.02, iters=600)
    band = height_band(rest, zmin=float(zmin), zmax=float(zmax))

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
        st.error("Too few points after filtering. Loosen filters.")
        st.stop()

    xy = band[:, :2]
    rotation_deg = 0.0
    if do_pca_align:
        xy, rotation_deg = pca_align(xy)

    grid, meta = to_grid(xy, res=float(res))
    meta.rotation_deg = float(rotation_deg)
    wall = wall_mask_from_points(grid, dil=int(dil), close=int(morph_close))
    lines_raw, edges = extract_lines(wall, min_len_px=int(min_len), theta_res=np.pi/180, rho_res=1, thresh=int(hough_thresh), max_gap=int(max_gap))
    lines = merge_collinear(lines_raw, angle_tol_deg=int(angle_merge), snap_ortho=bool(snap_ortho))
    doors = detect_doors(wall, res=meta.res)
    overlay_img = overlay(wall, lines, doors, alpha=0.75)

    inv = 255 - wall
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    wall_thickness_px = float(np.percentile(dist[wall>0], 90)) if np.any(wall>0) else 0.0
    wall_thickness_m = wall_thickness_px * meta.res * 2

    # room polygons
    rooms = detect_rooms(wall, meta, min_area_m2=2.0, approx_eps_px=6)

# ---------------------- Visuals (tabs) ----------------------
t1, t2, t3 = st.tabs(["Immersive 3D", "2D Layers", "Analytics & Exports"])

with t1:
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.subheader("3D point cloud (interactive)")
        samp = band[np.random.choice(len(band), size=min(SAMPLE_FOR_VIEW, len(band)), replace=False)]
        if color_mode == "height(z)":
            cvals = samp[:,2]
        else:
            xy_ = samp[:,:2]
            res_d = max(DEFAULT_RES_M, float(res))
            ij = np.floor((xy_ - xy_.min(0))/res_d).astype(int)
            keys = ij[:,0]*73856093 ^ ij[:,1]*19349663
            _, invk, cnts = np.unique(keys, return_inverse=True, return_counts=True)
            cvals = cnts[invk]

        # camera presets
        if cam_preset == "Top":
            az_eff, el_eff = 0, 90
        elif cam_preset == "Front":
            az_eff, el_eff = 0, 10
        elif cam_preset == "Side":
            az_eff, el_eff = 90, 10
        else:  # Isometric
            az_eff, el_eff = az, el

        fig = go.Figure(data=[go.Scatter3d(
            x=samp[:,0], y=samp[:,1], z=samp[:,2],
            mode='markers',
            marker=dict(size=point_size, color=cvals, colorscale='Viridis', opacity=0.92, colorbar=dict(len=0.5))
        )])
        cam = dict(eye=dict(x=float(np.cos(np.radians(az_eff))*2.2), y=float(np.sin(np.radians(az_eff))*2.2), z=float(np.sin(np.radians(el_eff))*2.0)))
        cam["projection"] = dict(type='perspective')
        fig.update_layout(
            scene=dict(
                xaxis_title='x', yaxis_title='y', zaxis_title='z',
                xaxis=dict(showbackground=True, backgroundcolor='rgba(0,0,0,0.02)', gridcolor='rgba(0,0,0,0.2)'),
                yaxis=dict(showbackground=True, backgroundcolor='rgba(0,0,0,0.02)', gridcolor='rgba(0,0,0,0.2)'),
                zaxis=dict(showbackground=True, backgroundcolor='rgba(0,0,0,0.02)', gridcolor='rgba(0,0,0,0.2)'),
                aspectmode='data', camera=cam
            ),
            margin=dict(l=0,r=0,t=0,b=0), height=560
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top‑down heatmap")
        heat = cv2.GaussianBlur(wall, (0,0), 1)
        if rooms:
            # overlay room contours
            heat_col = cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR)
            for r in rooms:
                pts = np.array([[int((x - meta.origin_xy[0]) / meta.res), int(meta.shape[0] - (y - meta.origin_xy[1]) / meta.res)] for (x,y) in r["coords"]], dtype=np.int32)
                cv2.polylines(heat_col, [pts], isClosed=True, color=(0,255,255), thickness=2)
            fig_hm = px.imshow(heat_col, origin='upper')
        else:
            fig_hm = px.imshow(heat, origin='upper')
        fig_hm.update_layout(coloraxis_showscale=False, height=560, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_hm, use_container_width=True)

with t2:
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        st.subheader("Wall mask")
        st.image(wall, use_container_width=True, clamp=True)
    with colB:
        st.subheader("Edges")
        st.image(edges, use_container_width=True, clamp=True)
    with colC:
        st.subheader("Vector overlay")
        alpha = st.slider("Overlay alpha", 0.1, 1.0, 0.75, 0.05)
        st.image(overlay(wall, lines, doors, alpha=float(alpha)), use_container_width=True, clamp=True)

with t3:
    st.subheader("Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Grid H×W (px)", f"{meta.shape[0]} × {meta.shape[1]}")
    m2.metric("Resolution (m/px)", f"{meta.res:.3f}")
    m3.metric("Lines", len(lines))
    m4.metric("Doors", len(doors))
    m5.metric("~Wall thickness (m)", f"{wall_thickness_m:.2f}")

    # Door stats and room list
    door_spans_m = [round(d[2]*meta.res, 2) for d in doors]
    st.write({
        "camera_rotation_deg": round(meta.rotation_deg, 1),
        "door_count": len(doors),
        "door_spans_m": door_spans_m,
        "room_count": len(rooms),
        "room_areas_m2": [round(r["area_m2"], 2) for r in rooms],
    })

    st.markdown("---")
    st.subheader("Downloads")
    # SVG
    svg_bytes = export_svg(lines, doors, meta)
    st.download_button("Download SVG plan", data=svg_bytes, file_name="plan.svg", mime="image/svg+xml", use_container_width=True)
    # PNGs
    ok, png_bytes = cv2.imencode(".png", wall)
    if ok:
        st.download_button("Download wall mask PNG", data=png_bytes.tobytes(), file_name="wall_mask.png", mime="image/png", use_container_width=True)
    ok2, overlay_png = cv2.imencode(".png", overlay_img)
    if ok2:
        st.download_button("Download overlay PNG", data=overlay_png.tobytes(), file_name="overlay.png", mime="image/png", use_container_width=True)

    # CSV (walls)
    rows = ["x1_m,y1_m,x2_m,y2_m"]
    for (x1,y1,x2,y2) in lines:
        x1m,y1m = px_to_m((x1,y1), meta); x2m,y2m = px_to_m((x2,y2), meta)
        y1m = meta.origin_xy[1] + (meta.shape[0]-y1)*meta.res
        y2m = meta.origin_xy[1] + (meta.shape[0]-y2)*meta.res
        rows.append(f"{x1m:.3f},{y1m:.3f},{x2m:.3f},{y2m:.3f}")
    csv_text = "\n".join(rows)
    csv_bytes = csv_text.encode("utf-8")
    st.download_button("Download wall lines CSV", data=csv_bytes, file_name="walls.csv", mime="text/csv", use_container_width=True)

    # GeoJSON (walls+doors+rooms)
    gj = export_geojson(lines, doors, meta, rooms=rooms)
    gj_bytes = json.dumps(gj, indent=2).encode("utf-8")
    st.download_button("Download GeoJSON", data=gj_bytes, file_name="plan.geojson", mime="application/geo+json", use_container_width=True)

    # Metrics JSON
    metrics = {
        "grid_size_px": meta.shape,
        "grid_resolution_m_per_px": meta.res,
        "lines_detected": len(lines),
        "door_candidates": len(doors),
        "door_spans_m": door_spans_m,
        "rooms": rooms,
    }
    st.download_button("Download metrics JSON", data=json.dumps(metrics, indent=2).encode("utf-8"), file_name="metrics.json", mime="application/json", use_container_width=True)

# Footer ribbon + subtle gradient
st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg, #0b1020 0%, #0b1020 40%, #0f172a 100%);} 
      .stApp, .stMarkdown, .stPlotlyChart {color: #e5e7eb !important}
    </style>
    <div style=\"position:fixed;right:16px;bottom:16px;background:#111827;color:#e5e7eb;padding:8px 12px;border-radius:12px;opacity:0.9;font-size:12px;\">
      Demo‑only build • Streamlit • CV2 • Plotly
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("About this demo"):
    st.write(
        "This v6 build removes uploads and buttons, auto‑runs the demo, disables file watchers to avoid inotify limits, adds camera presets, grid/scale in SVG, room polygonization & area stats, overlay PNG export, and GeoJSON with rooms."
    )
