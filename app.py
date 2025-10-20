import io, os, tempfile, math
from dataclasses import dataclass
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cv2
import svgwrite
from scipy import ndimage

# Optional: Open3D for PCD/PLY
try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

st.set_page_config(page_title="LiDAR → Floor Plan", layout="wide")

# ---------- Utilities ----------
@dataclass
class GridMeta:
    origin_xy: np.ndarray   # min x,y (meters)
    res: float              # meters per pixel
    shape: tuple            # (H, W)

def _to_tempfile(uploaded_file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return f.name

def load_cloud(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pcd") or name.endswith(".ply"):
        if not HAS_O3D:
            st.error("Open3D not available. Install `open3d` for PCD/PLY.")
            st.stop()
        path = _to_tempfile(uploaded_file, os.path.splitext(name)[1])
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts
    elif name.endswith(".npz"):
        arr = np.load(io.BytesIO(uploaded_file.getvalue()))
        key = "points" if "points" in arr else list(arr.keys())[0]
        pts = arr[key].astype(np.float32)
        return pts
    elif name.endswith(".csv") or name.endswith(".txt"):
        data = uploaded_file.read().decode("utf-8").strip().splitlines()
        pts = np.loadtxt(data, delimiter=",", dtype=np.float32)
        return pts
    else:
        st.error("Unsupported file. Use .pcd, .ply, .npz, .csv, or .txt (x,y,zz).")
        st.stop()

def make_demo(n_rooms=3, noise=0.01, seed=42):
    rng = np.random.RandomState(seed)
    clouds = []
    x0, y0 = 0.0, 0.0
    for i in range(n_rooms):
        w = rng.uniform(3.0, 5.0)
        h = rng.uniform(3.0, 5.5)
        # draw rectangle wall points at z in [0,2.5]
        per = int(800)
        t = np.linspace(0, 1, per, endpoint=False)
        rect = np.vstack([
            np.where(t < 0.25, x0 + t*4*w, np.where(t < 0.5, x0 + w, np.where(t < 0.75, x0 + (1 - (t-0.5)*4)*w, x0))),
            np.where(t < 0.25, y0, np.where(t < 0.5, y0 + (t-0.25)*4*h, np.where(t < 0.75, y0 + h, y0 + (1 - (t-0.75)*4)*h)))
        ]).T
        z = rng.uniform(0.2, 2.4, size=(per, 1))
        rect += rng.normal(0, noise, rect.shape)
        clouds.append(np.hstack([rect, z]))
        # door gap on the bottom wall
        door_x = x0 + w*0.5 + rng.uniform(-0.3, 0.3)
        door_pts = np.linspace(-0.45, 0.45, 200)  # ~0.9 m gap
        door = np.stack([door_x + 0*door_pts, y0 + door_pts*0, rng.uniform(0.05, 2.0, door_pts.shape)], axis=1)
        # remove wall points around door by not adding them: already simulated as a gap
        x0 += w + 1.5
        y0 += rng.uniform(-0.5, 0.5)
    pts = np.concatenate(clouds, axis=0)
    return pts

def voxel_downsample(pts, leaf=0.03):
    if pts.size == 0: return pts
    grid = np.floor(pts[:, :3] / leaf).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return pts[idx]

def statistical_outlier(pts, nb=20, std_ratio=2.0):
    if pts.shape[0] < nb+1: return pts
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(nb, len(pts)-1)).fit(pts[:, :3])
    dists, _ = nn.kneighbors(pts[:, :3])
    mean_d = dists[:, 1:].mean(axis=1)
    m, s = mean_d.mean(), mean_d.std()
    keep = mean_d <= (m + std_ratio * s)
    return pts[keep]

def ransac_ground(pts, thresh=0.02, iters=1000):
    # Fit plane ax+by+cz+d=0
    if pts.shape[0] < 100: return np.empty((0,3)), pts
    rng = np.random.default_rng(0)
    best_inliers = None
    best_count = -1
    for _ in range(iters):
        ids = rng.choice(pts.shape[0], size=3, replace=False)
        p0, p1, p2 = pts[ids]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-6: continue
        n = n / np.linalg.norm(n)
        d = -np.dot(n, p0)
        dist = np.abs(pts.dot(n) + d)
        inliers = dist < thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_inliers = inliers
    if best_inliers is None:
        return np.empty((0,3)), pts
    ground = pts[best_inliers]
    rest = pts[~best_inliers]
    return ground, rest

def height_band(pts, zmin=0.1, zmax=2.4):
    m = (pts[:,2] >= zmin) & (pts[:,2] <= zmax)
    return pts[m]

def to_grid(xy, res=0.03, pad_ratio=0.02):
    min_xy = xy.min(axis=0); max_xy = xy.max(axis=0)
    span = max_xy - min_xy
    pad = np.maximum(span * pad_ratio, [0.2, 0.2])
    min_xy -= pad; max_xy += pad
    size = np.ceil((max_xy - min_xy)/res).astype(int) + 1
    size = np.maximum(size, 16)
    H, W = int(size[1]), int(size[0])
    grid = np.zeros((H, W), np.uint8)
    ij = ((xy - min_xy)/res).astype(int)
    ij[:,0] = np.clip(ij[:,0], 0, W-1)
    ij[:,1] = np.clip(ij[:,1], 0, H-1)
    grid[ij[:,1], ij[:,0]] = 255
    meta = GridMeta(origin_xy=min_xy, res=float(res), shape=(H, W))
    return grid, meta

def wall_mask_from_points(grid, dil=1):
    # thicken sparse samples to solid walls
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, dil), max(1, dil)))
    wall = cv2.dilate(grid, k, iterations=1)
    wall = cv2.threshold(wall, 128, 255, cv2.THRESH_BINARY)[1]
    return wall

def extract_lines(wall_mask, min_len_px=50, theta_res=np.pi/180, rho_res=1, thresh=80):
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=rho_res, theta=theta_res, threshold=thresh,
                            minLineLength=min_len_px, maxLineGap=8)
    out = []
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2 = map(int, l)
            out.append((x1,y1,x2,y2))
    return out, edges

def merge_collinear(lines, angle_tol_deg=5, dist_tol_px=8):
    if not lines: return []
    def angle(l):
        x1,y1,x2,y2 = l
        return math.degrees(math.atan2(y2-y1, x2-x1)) % 180
    def norm(l):
        x1,y1,x2,y2 = l
        if (x2-x1)**2 + (y2-y1)**2 == 0: return l
        return l
    used = [False]*len(lines)
    merged = []
    for i,l in enumerate(lines):
        if used[i]: continue
        ag = angle(l)
        x1,y1,x2,y2 = l
        pts = [(x1,y1),(x2,y2)]
        used[i]=True
        for j,m in enumerate(lines):
            if used[j]: continue
            if abs(angle(m) - ag) <= angle_tol_deg or abs(abs(angle(m)-ag)-180)<=angle_tol_deg:
                # if near same line support, append endpoints if close
                pts += [(m[0],m[1]), (m[2],m[3])]
                used[j]=True
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        # project to the dominant axis for extension
        if abs(math.sin(math.radians(ag))) < 0.5:  # near horizontal
            ymed = int(np.median(ys))
            xmn, xmx = min(xs), max(xs)
            merged.append((xmn, ymed, xmx, ymed))
        else:  # near vertical
            xmed = int(np.median(xs))
            ymn, ymx = min(ys), max(ys)
            merged.append((xmed, ymn, xmed, ymx))
    return merged

def detect_doors(wall_mask, door_min_px=20, door_max_px=60):
    # door ≈ narrow gap in wall band near floor slice. Heuristic on 2D mask.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    band = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k, iterations=1)
    inv = 255 - band
    # Find gaps along walls by looking at thin free corridors abutting wall pixels
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    # door cores as ridges in free space adjacent to walls
    door_core = (dist > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(door_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doors = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        span = max(w,h)
        if door_min_px <= span <= door_max_px:
            cx = x + w//2; cy = y + h//2
            doors.append((cx, cy, span))
    return doors

def overlay(wall_mask, lines, doors):
    color = np.dstack([wall_mask]*3)
    # walls in white already; draw lines in green, doors in red
    for (x1,y1,x2,y2) in lines:
        cv2.line(color, (x1,y1), (x2,y2), (0,255,0), 2)
    for (cx,cy,span) in doors:
        cv2.circle(color, (cx,cy), max(2, span//6), (0,0,255), 2)
    return color

def px_to_m(p, meta: GridMeta):
    x_px, y_px = p
    x_m = meta.origin_xy[0] + x_px * meta.res
    y_m = meta.origin_xy[1] + y_px * meta.res
    return x_m, y_m

def export_svg(lines, doors, meta: GridMeta):
    H, W = meta.shape
    # SVG in meters: set viewBox in meters
    vb_w = W * meta.res
    vb_h = H * meta.res
    dwg = svgwrite.Drawing(size=("1000px","1000px"),
                           viewBox=f"0 0 {vb_w:.3f} {vb_h:.3f}")
    # white background
    dwg.add(dwg.rect(insert=(0,0), size=(vb_w, vb_h),
                     fill="white"))
    # walls
    layer_walls = dwg.add(dwg.g(id="WALLS", stroke="black", fill="none", stroke_width=0.02))
    for (x1,y1,x2,y2) in lines:
        x1m,y1m = px_to_m((x1,y1), meta)
        x2m,y2m = px_to_m((x2,y2), meta)
        # y axis inversion (image coords to Cartesian). Flip by (H - y)
        y1m = meta.origin_xy[1] + (meta.shape[0]-y1) * meta.res
        y2m = meta.origin_xy[1] + (meta.shape[0]-y2) * meta.res
        layer_walls.add(dwg.line(start=(x1m,y1m), end=(x2m,y2m)))
    # doors
    layer_doors = dwg.add(dwg.g(id="DOORS", stroke="red", fill="none", stroke_width=0.02))
    for (cx,cy,span) in doors:
        xm, ym = px_to_m((cx,cy), meta)
        ym = meta.origin_xy[1] + (meta.shape[0]-cy) * meta.res
        r = max(0.05, span * meta.res * 0.3)
        layer_doors.add(dwg.circle(center=(xm, ym), r=r))
    return dwg.tostring().encode("utf-8")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Input")
    mode = st.radio("Data source", ["Upload", "Demo"], index=0)
    if mode == "Upload":
        up = st.file_uploader("Point cloud (.pcd, .ply, .npz, .csv)", type=["pcd","ply","npz","csv","txt"])
    else:
        n_rooms = st.slider("Demo rooms", 2, 6, 3)
        demo_seed = st.number_input("Demo seed", 0, 9999, 42)

    st.header("Preprocess")
    voxel = st.slider("Voxel size (m)", 0.01, 0.10, 0.03, 0.01)
    out_nb = st.slider("Outlier neighbors", 5, 50, 20, 1)
    out_std = st.slider("Outlier std ratio", 0.5, 4.0, 2.0, 0.1)

    st.header("Height Band (m)")
    zmin = st.number_input("z_min", 0.0, 1.0, 0.10, 0.05)
    zmax = st.number_input("z_max", 1.5, 4.0, 2.40, 0.05)

    st.header("Grid/Mask")
    res = st.slider("Grid resolution (m/px)", 0.01, 0.10, 0.03, 0.01)
    dil = st.slider("Wall dilation (px)", 1, 7, 2, 1)

    st.header("Vectorization")
    min_len = st.slider("Min line length (px)", 10, 200, 60, 5)
    hough_thresh = st.slider("Hough threshold", 10, 200, 80, 5)
    angle_merge = st.slider("Merge angle tol (deg)", 1, 15, 5, 1)

    st.header("Door Heuristic")
    door_min = st.slider("Door min width (px)", 10, 120, 20, 2)
    door_max = st.slider("Door max width (px)", 20, 240, 60, 2)

# ---------- Load Data ----------
st.title("LiDAR → 2D Floor-Plan Extractor")
if mode == "Upload" and up is None:
    st.info("Upload a point cloud or switch to Demo.")
    st.stop()

with st.spinner("Loading points"):
    if mode == "Upload":
        pts = load_cloud(up)
    else:
        pts = make_demo(n_rooms=n_rooms, seed=int(demo_seed))
    if pts.ndim != 2 or pts.shape[1] < 3:
        st.error("Expect Nx3 points [x,y,z].")
        st.stop()

orig_n = pts.shape[0]

# ---------- Preprocess ----------
pts = voxel_downsample(pts, leaf=float(voxel))
pts = statistical_outlier(pts, nb=int(out_nb), std_ratio=float(out_std))
ground, rest = ransac_ground(pts, thresh=0.02, iters=500)
band = height_band(rest, zmin=float(zmin), zmax=float(zmax))
xy = band[:, :2]

if xy.shape[0] < 50:
    st.error("Too few points after filtering. Loosen filters.")
    st.stop()

grid, meta = to_grid(xy, res=float(res))
wall = wall_mask_from_points(grid, dil=int(dil))
lines_raw, edges = extract_lines(wall, min_len_px=int(min_len), theta_res=np.pi/180, rho_res=1, thresh=int(hough_thresh))
lines = merge_collinear(lines_raw, angle_tol_deg=int(angle_merge))
doors = detect_doors(wall, door_min_px=int(door_min), door_max_px=int(door_max))
overlay_img = overlay(wall, lines, doors)

# ---------- UI Layout ----------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("3D view")
    # sample for speed
    samp = pts[np.random.choice(len(pts), size=min(60000, len(pts)), replace=False)]
    fig = go.Figure(data=[go.Scatter3d(
        x=samp[:,0], y=samp[:,1], z=samp[:,2],
        mode='markers', marker=dict(size=1))])
    fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                      margin=dict(l=0,r=0,t=0,b=0), height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Input: {orig_n:,} pts → after filters: {pts.shape[0]:,} pts. Band: {band.shape[0]:,} pts.")

with col2:
    st.subheader("2D plan layers")
    st.image(wall, caption="Wall mask", use_column_width=True, clamp=True)
    st.image(overlay_img, caption="Vector overlay (green=walls, red≈doors)", use_column_width=True, clamp=True)

st.subheader("Metrics")
wall_pixels = int((wall>0).sum())
st.write({
    "grid_size_px": meta.shape,
    "grid_resolution_m_per_px": meta.res,
    "wall_pixels": wall_pixels,
    "lines_detected": len(lines),
    "door_candidates": len(doors),
})

# ---------- Exports ----------
svg_bytes = export_svg(lines, doors, meta)
st.download_button("Download SVG plan", data=svg_bytes, file_name="plan.svg", mime="image/svg+xml")

# also export occupancy as PNG
ok, png_bytes = cv2.imencode(".png", wall)
if ok:
    st.download_button("Download wall mask PNG", data=png_bytes.tobytes(), file_name="wall_mask.png", mime="image/png")

# export thin CSV of lines in meters
def lines_to_csv(lines, meta):
    rows = ["x1_m,y1_m,x2_m,y2_m"]
    for (x1,y1,x2,y2) in lines:
        x1m,y1m = px_to_m((x1,y1), meta)
        x2m,y2m = px_to_m((x2,y2), meta)
        y1m = meta.origin_xy[1] + (meta.shape[0]-y1) * meta.res
        y2m = meta.origin_xy[1] + (meta.shape[0]-y2) * meta.res
        rows.append(f"{x1m:.3f},{y1m:.3f},{x2m:.3f},{y2m:.3f}")
    return ("\n".join(rows)).encode("utf-8")

csv_bytes = lines_to_csv(lines, meta)
st.download_button("Download wall lines CSV", data=csv_bytes, file_name="walls.csv", mime="text/csv")

st.divider()
st.markdown("**Notes**")
st.markdown(
"- Tune z-band to isolate wall heights.\n"
"- Increase dilation if walls look broken.\n"
"- Set line length higher to remove clutter.\n"
"- Door detection is heuristic. Calibrate door min/max in px using your grid resolution."
)
