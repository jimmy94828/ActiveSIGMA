import argparse, json, os, re
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

########################## helper    ######################################

# 0.28209479177 是 SH0 的常數。
def sh_dc_to_rgb(dc):
    # dc: shape (N,3)
    # return (dc * 0.28209479177 + 0.5).clip(0, 1)
    #C0 = 0.28209479177387814
    #rgb = dc * C0 + 0.5
    #return rgb
	# 方法1: 標準 SH 轉換 (通常適用於標準 3DGS)
    C0 = 0.28209479177387814
    rgb_standard = dc * C0 + 0.5

    # 檢查標準轉換是否合理
    if rgb_standard.min() >= -0.5 and rgb_standard.max() <= 1.5:
        print(f"[INFO] Using standard SH conversion")
        rgb = np.clip(rgb_standard, 0, 1)
    else:
        # 方法2: 直接正規化 DC 係數到 [0,1]
        print(f"[INFO] DC values out of standard range, using normalization")
        print(f"[INFO] After standard conversion: [{rgb_standard.min():.3f}, {rgb_standard.max():.3f}]")

        # 對每個通道分別正規化，保持顏色比例
        rgb = np.zeros_like(dc)
        for i in range(3):
            ch_min, ch_max = dc[:, i].min(), dc[:, i].max()
            if ch_max > ch_min:
                rgb[:, i] = (dc[:, i] - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, i] = 0.5
        print(f"[INFO] After normalization: [{rgb.min():.3f}, {rgb.max():.3f}]")

    return rgb.astype(np.float32)

from plyfile import PlyData
def load_ply_pointcloud(ply_path, max_points=None, voxel_size=None):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    pts = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    names = v.data.dtype.names
    
    if ('f_dc_0' in names) and ('f_dc_1' in names) and ('f_dc_2' in names):
        dc = np.vstack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).T.astype(np.float32)
        colors = sh_dc_to_rgb(dc)
    else:
        colors = np.ones((pts.shape[0], 3), dtype=np.float32) * 0.5
        print("gray")

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]; colors = colors[idx]

    if voxel_size is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)

    return pts, colors

def remove_ceiling_points_auto(pcd, min_floor_h=None, max_ceiling_h=None, floor_margin=0.15, ceiling_margin=0.15):
    if len(pcd.points) == 0:
        return pcd
    pts = np.asarray(pcd.points)
    y = pts[:,1]
    floor_est = np.percentile(y, 15) if min_floor_h is None else float(min_floor_h)
    ceil_est  = np.percentile(y, 85) if max_ceiling_h is None else float(max_ceiling_h)
    lower_th = floor_est + floor_margin
    upper_th = ceil_est - ceiling_margin
    mask = (y > lower_th) & (y < upper_th)
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    print(f"[INFO] keep {mask.sum()}/{len(mask)} points after floor/ceiling removal (floor_est={floor_est:.3f}, ceil_est={ceil_est:.3f})")
    return filtered

########################## Load Data ##########################

def load_points(points_path, colors_path=None):
    pts = np.load(points_path)  # (N,3) in [0..255] domain or already world-ish
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points shape must be (N,3), got {pts.shape}")

    # scale: apartment_0 = points * (10000/255)
    pts = pts.astype(np.float32) * (10000.0 / 255.0)

    colors = None
    if colors_path:
        colors = np.load(colors_path)  # (N,3), either [0,1] or [0,255]
        if colors.max() > 1.0:
            colors = colors.astype(np.float32) / 255.0
        colors = np.clip(colors, 0.0, 1.0)
        if colors.shape != pts.shape:
            raise ValueError(f"colors shape must match points: {colors.shape} vs {pts.shape}")
    return pts, colors

def load_semantic_colors(xlsx_path):
    df = pd.read_excel(xlsx_path, usecols=["Color_Code (R,G,B)", "Name"]) # read col ["Color_Code (R,G,B)", "Name"]
    pattern = re.compile(r"\d+")
    semantic = {}

    for _, row in df.iterrows():
        name = str(row["Name"]).strip().lower()
        nums = pattern.findall(str(row["Color_Code (R,G,B)"]))
        if len(nums) >= 3:
            semantic[name] = [int(nums[0]), int(nums[1]), int(nums[2])]

    return semantic # { 'object':[R,G,B] }

######################## Point Cloud Processing ########################

def remove_ceiling_points(
    pcd, semantic_dict, margin=3, 
    ceiling_height=0.125, offset=0.2,

) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    print(f'Max Y: ', max(points[:, 1]))
    print(f'Min Y: ', min(points[:, 1]))
    print(f'Max Y: ', (points[:, 1].mean()))
    floor_height = min(points[:, 1])

    # Get object mask
    ceiling_rgb = np.array(semantic_dict["ceiling"], dtype=np.uint8)
    floor_rgb   = np.array(semantic_dict["floor"], dtype=np.uint8)
    colors_rgb = (colors * 255.0).round().astype(np.uint8)
    
    dist_ceiling = np.linalg.norm(colors_rgb - ceiling_rgb, axis=1)
    dist_floor   = np.linalg.norm(colors_rgb - floor_rgb, axis=1)

    # Remove points higher than ceiling height - offset
    ceiling_threshold = ceiling_height - offset
    floor_threshold = floor_height + 0.5

    mask = (dist_ceiling > margin) & (dist_floor > margin) & (points[:, 1] < ceiling_threshold) & (points[:, 1] > floor_threshold)
    
    # remove ceiling and floor
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return filtered_pcd


def TopDownProjection(points, colors, W=1024, H=1024):
    """
    超簡版頂視投影：
      u(=col) ← z 線性映射到 [0, W-1]
      v(=row) ← x 線性映射到 [H-1, 0]（左上為原點，向下為正）
    回傳 (H,W,3) uint8 影像與最基本的反投影參數。
    """
    pts = np.asarray(points)           # (N,3) -> x,y,z
    cols = np.asarray(colors)          # (N,3) RGB in [0,1] 或 [0,255]
    if cols.dtype != np.uint8:
        cols = np.clip(cols, 0, 1); cols = (cols * 255).astype(np.uint8)

    x = pts[:, 0]
    z = pts[:, 2]

    # 避免極端情況除以 0
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    if x_max == x_min: x_max = x_min + 1e-6
    if z_max == z_min: z_max = z_min + 1e-6

    # 資料 -> 像素
    u = ((z - z_min) / (z_max - z_min) * (W - 1)).astype(np.int32)
    v = ((1.0 - (x - x_min) / (x_max - x_min)) * (H - 1)).astype(np.int32)

    # 畫到影像（白底）
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    img[v[inb], u[inb]] = cols[inb]

    # 最小必要投影資訊（用於像素<->資料雙向換算）
    project_info = {
        "W": W, "H": H,
        "x_min": x_min, "x_max": x_max,
        "z_min": z_min, "z_max": z_max,
        "to_world": lambda uu, vv: (
            x_min + (1.0 - vv/(H-1)) * (x_max - x_min),             # x
            z_min + (uu/(W-1))       * (z_max - z_min)              # z
        ),
        "to_pixel": lambda xx, zz: (
            int(round((zz - z_min)/(z_max - z_min) * (W - 1))),     # u
            int(round((1.0 - (xx - x_min)/(x_max - x_min)) * (H - 1)))  # v
        )
    }
    return img, project_info

def save_projection_outputs(img, project_info, img_path="map_sparse.png", info_path="map_meta.json"):
    # save sparse img
    plt.imsave(img_path, img)

    # save meta data
    serializable = {
        "W": int(project_info["W"]),
        "H": int(project_info["H"]),
        "x_min": float(project_info["x_min"]),
        "x_max": float(project_info["x_max"]),
        "z_min": float(project_info["z_min"]),
        "z_max": float(project_info["z_max"]),
        "pixel_origin": "top-left",
        "axes_mapping": {"u_from": "z", "v_from": "x"},
        "formulas": {
            "u = (z - z_min) / (z_max - z_min) * (W-1)": True,
            "v = (1 - (x - x_min)/(x_max - x_min)) * (H-1)": True
        }
    }
    with open(info_path, "w") as f:
        json.dump(serializable, f, indent=2)
    return img_path, info_path
    

def save_scatter_pixel_space(u, v, colors, W, H, out_path="map.png",
                             point_px=3, dpi=200, bg_color=(1,1,1)):
    """
    在像素座標系(左上原點)用 scatter 畫點：
      - u: (N,) 列座標（x 軸）
      - v: (N,) 行座標（y 軸，往下為正）
      - colors: (N,3) in [0,1] 或 [0,255]
    """
    u = np.asarray(u); v = np.asarray(v); c = np.asarray(colors)
    if c.max() > 1.0: c = c / 255.0

    # Matplotlib 的 scatter 尺寸單位是「點(point)」，非像素；
    # 1 point = 1/72 inch，所以要把像素大小換成 point：
    s_points = (point_px * 72.0 / dpi) ** 2  # scatter 的 s 是面積(點^2)

    fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax = plt.axes([0,0,1,1], facecolor=bg_color)  # 無邊框
    ax.scatter(u, v, s=s_points, c=c, marker='s')  # 用正方形 marker 比像素視覺一致
    ax.set_xlim(-0.5, W-0.5)
    ax.set_ylim(H-0.5, -0.5)  # 反轉 y，讓(0,0)在左上
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    Image.open(out_path).convert("RGB").save(out_path)
    return out_path    

def generate_map(points_path='/mnt/HDD3/dow904/ricky/physical-ai25/hw2_robot/office4/params.ply'):
    semantic_path = '/mnt/HDD3/dow904/ricky/physical-ai25/hw2/color_coding_semantic_segmentation_classes.xlsx'
    
    # === load semantic map ===
    pts, colors = load_ply_pointcloud(points_path)
    print(f'PTS SHAPE: {pts.shape}')
    print(f'COLOR SHAPE: {colors.shape}')
    
    # === load semantic dict ===
    semantic_dict = load_semantic_colors(semantic_path)
    
    # === create point map ===
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # === Remove Floor & Ceiling ===
    pcd = remove_ceiling_points_auto(pcd)
    
    # === visualize color map ===
    o3d.visualization.draw_geometries([pcd], window_name="Semantic Color Map")
    W = 1920
    H = 1440
    img, project_info = TopDownProjection(np.asarray(pcd.points), np.asarray(pcd.colors),W=W,H=H)
    save_projection_outputs(img, project_info)

    # get uv
    x = np.asarray(pcd.points)[:,0]
    z = np.asarray(pcd.points)[:,2]

    to_pixel = project_info["to_pixel"]
    uv = np.array([to_pixel(xx, zz) for xx, zz in zip(x, z)])
    u = uv[:,0]; v = uv[:,1]

    # save scatter for better visualization
    save_scatter_pixel_space(u, v, np.asarray(pcd.colors), W, H,
                            out_path="map.png",
                            point_px=5, dpi=200)

if __name__ == '__main__':
    generate_map()