import os
import cv2
import numpy as np

# Import effect functions from test_effects
import sys
sys.path.append(r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\scripts")
from test_effects import effect_original, effect_holographic, effect_heatmap, effect_lidar, effect_gaussian_splat, effect_blob

EFFECTS = [
    effect_original,       # Frame 1: Original
    effect_holographic,    # Frame 2: Holographic
    effect_heatmap,        # Frame 3: Heatmap
    effect_lidar,          # Frame 4: LiDAR
    effect_gaussian_splat, # Frame 5: Gaussian Splat
    effect_blob            # Frame 6: Blob
]

EFFECT_NAMES = [
    "1. ORIGINAL",
    "2. HOLOGRAPHIC",
    "3. HEATMAP",
    "4. LIDAR",
    "5. 3D GAUSSIAN SPLAT",
    "6. METABALL BLOB"
]

# 9:16 Vertical Dimensions (Standard Instagram Reels: 1080x1920)
W, H = 1080, 1920
FOV_V = 52.0
F_LEN = (H / 2) / np.tan(np.radians(FOV_V / 2))
K = np.array([
    [F_LEN, 0, W / 2 + 60], # slight center adjustment
    [0, F_LEN, H / 2 - 40],
    [0, 0, 1]
])

TARGET_POINT = np.array([0.2, 0.95, -3.6])
UP_VECTOR = np.array([0.0, 1.0, 0.0])

def get_camera_extrinsics(cam_pos, target):
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, UP_VECTOR)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    R_cv = np.vstack([right, -cam_up, forward])
    t_cv = -R_cv @ cam_pos
    return R_cv, t_cv

def project_point(pt, R_cv, t_cv):
    p_cam = R_cv @ pt + t_cv
    if p_cam[2] <= 0.1:
        return None
    p_img = K @ p_cam
    return np.array([p_img[0] / p_img[2], p_img[1] / p_img[2]])

def test_vertical_composite(theta_deg=38.0):
    src_p = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\data_dp\frames_combined"
    
    theta_rad = np.radians(theta_deg)
    cam_dist = 9.8
    cam_height = 2.8
    cam_pos = np.array([
        np.sin(theta_rad) * cam_dist,
        cam_height,
        np.cos(theta_rad) * cam_dist - 3.6
    ])
    R_cv, t_cv = get_camera_extrinsics(cam_pos, TARGET_POINT)

    # 1. Dark atmospheric obsidian canvas with subtle vertical gradient
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        val = int(8 + 8 * (1.0 - y / H))
        canvas[y, :] = (val, int(val * 0.8), int(val * 0.7))

    # 2. 3D floor perspective grid
    grid_color = (38, 32, 26)
    grid_accent = (80, 68, 48)
    for gz in np.linspace(-12.0, 2.0, 32):
        p1 = project_point(np.array([-6.0, 0.0, gz]), R_cv, t_cv)
        p2 = project_point(np.array([6.0, 0.0, gz]), R_cv, t_cv)
        if p1 is not None and p2 is not None:
            col = grid_accent if abs(gz) < 0.1 or abs(gz + 8.0) < 0.1 else grid_color
            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 1, cv2.LINE_AA)

    for gx in np.linspace([-5.0, 5.0, 21][0], [-5.0, 5.0, 21][1], 21):
        p1 = project_point(np.array([gx, 0.0, -12.0]), R_cv, t_cv)
        p2 = project_point(np.array([gx, 0.0, 2.0]), R_cv, t_cv)
        if p1 is not None and p2 is not None:
            col = grid_accent if abs(gx) < 0.1 else grid_color
            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 1, cv2.LINE_AA)

    half_w = 2.38
    y_bottom = 0.0
    y_top = 2.0
    src_pts = np.float32([[0, 0], [890, 0], [890, 374], [0, 374]])
    
    TIME_DEPTH = 1.40
    DELAY_FRAMES = 10
    BASE_FRAME = 220
    OPACITY_LADDER = [1.0, 0.90, 0.78, 0.66, 0.55, 0.45]

    for k in range(5, -1, -1):
        f_idx = max(0, BASE_FRAME - k * DELAY_FRAMES)
        f_path = os.path.join(src_p, f"frame_{f_idx:04d}.png")
        if not os.path.exists(f_path):
            continue
        raw_rgba = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        
        effect_fn = EFFECTS[k]
        styled_rgba = effect_fn(raw_rgba)

        z_k = -k * TIME_DEPTH
        corners_3d = [
            np.array([-half_w, y_top, z_k]),
            np.array([half_w, y_top, z_k]),
            np.array([half_w, y_bottom, z_k]),
            np.array([-half_w, y_bottom, z_k])
        ]
        corners_2d = [project_point(pt, R_cv, t_cv) for pt in corners_3d]
        if any(c is None for c in corners_2d):
            continue

        dst_pts = np.float32(corners_2d)
        H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(styled_rgba, H_mat, (W, H), flags=cv2.INTER_LINEAR)

        w_rgb = warped[:, :, :3].astype(np.float32)
        w_alpha = (warped[:, :, 3].astype(np.float32) / 255.0) * OPACITY_LADDER[k]
        alpha_3 = w_alpha[:, :, None]

        canvas = (w_rgb * alpha_3 + canvas.astype(np.float32) * (1.0 - alpha_3)).astype(np.uint8)

    # Clean Mobile HUD optimized for Instagram Reels Safe Zones
    # Top zone (120px - 320px)
    cv2.putText(canvas, "FIGHT THROUGH TIME", (W // 2 - 340, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "4D MOTION SCULPTURE", (W // 2 - 320, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PAST  <==============  TIME AXIS  ==============>  FUTURE", (W // 2 - 420, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 180, 200), 1, cv2.LINE_AA)

    # Effect Legend in mid-lower section
    legend_y = H - 360
    cv2.line(canvas, (80, legend_y), (W - 80, legend_y), (60, 70, 85), 2, cv2.LINE_AA)
    cv2.putText(canvas, "6 TEMPORAL VISUALIZATIONS (Z = TIME x 1.4m)", (W // 2 - 310, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "1: ORIGINAL  |  2: HOLOGRAPHIC  |  3: HEATMAP", (W // 2 - 360, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "4: LiDAR  |  5: 3D GAUSSIAN SPLAT  |  6: METABALL BLOB", (W // 2 - 410, legend_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 240, 255), 1, cv2.LINE_AA)

    out_p = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\samples_new\reels_9x16_preview.png"
    cv2.imwrite(out_p, canvas)
    print("Saved 9:16 vertical preview to:", out_p)

if __name__ == "__main__":
    test_vertical_composite(45.0)
