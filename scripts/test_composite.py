import os
import cv2
import numpy as np
import time

# Import effect functions from test_effects
from test_effects import effect_original, effect_holographic, effect_heatmap, effect_lidar, effect_gaussian_splat, effect_blob

EFFECTS = [
    effect_original,       # Frame 1 (Current pose, k=0)
    effect_holographic,    # Frame 2 (k=1)
    effect_heatmap,        # Frame 3 (k=2)
    effect_lidar,          # Frame 4 (k=3)
    effect_gaussian_splat, # Frame 5 (k=4)
    effect_blob            # Frame 6 (k=5)
]

EFFECT_NAMES = [
    "ORIGINAL (CURRENT)",
    "HOLOGRAPHIC (T-0.4s)",
    "HEATMAP (T-0.8s)",
    "LIDAR (T-1.2s)",
    "GAUSSIAN SPLAT (T-1.6s)",
    "METABALL BLOB (T-2.0s)"
]

W, H = 1280, 720
FOV = 42.0
F_LEN = (W / 2) / np.tan(np.radians(FOV / 2))
K = np.array([
    [F_LEN, 0, W / 2],
    [0, F_LEN, H / 2],
    [0, 0, 1]
])

TARGET_POINT = np.array([0.0, 0.95, -3.8])
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

def test_composite_frame(theta_deg=45.0):
    src_p = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\data_dp\frames_combined"
    
    # 45 deg angle camera
    theta_rad = np.radians(theta_deg)
    cam_dist = 8.5
    cam_height = 1.7
    cam_pos = np.array([
        np.sin(theta_rad) * cam_dist,
        cam_height,
        np.cos(theta_rad) * cam_dist - 3.8
    ])
    R_cv, t_cv = get_camera_extrinsics(cam_pos, TARGET_POINT)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (8, 6, 5)

    # 3D Grid
    grid_color = (32, 28, 22)
    grid_accent = (65, 55, 38)
    for gz in np.linspace(-12.0, 1.5, 28):
        p1 = project_point(np.array([-5.0, 0.0, gz]), R_cv, t_cv)
        p2 = project_point(np.array([5.0, 0.0, gz]), R_cv, t_cv)
        if p1 is not None and p2 is not None:
            col = grid_accent if abs(gz) < 0.1 or abs(gz + 9.0) < 0.1 else grid_color
            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 1, cv2.LINE_AA)

    for gx in np.linspace(-5.0, 5.0, 21):
        p1 = project_point(np.array([gx, 0.0, -12.0]), R_cv, t_cv)
        p2 = project_point(np.array([gx, 0.0, 1.5]), R_cv, t_cv)
        if p1 is not None and p2 is not None:
            col = grid_accent if abs(gx) < 0.1 else grid_color
            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 1, cv2.LINE_AA)

    half_w = 2.38
    y_bottom = 0.0
    y_top = 2.0
    src_pts = np.float32([[0, 0], [890, 0], [890, 374], [0, 374]])
    
    TIME_DEPTH = 1.45
    DELAY_FRAMES = 10
    BASE_FRAME = 220 # intense combat frame
    OPACITY_LADDER = [1.0, 0.88, 0.76, 0.65, 0.54, 0.44]

    # Render back to front: k = 5 down to 0
    for k in range(5, -1, -1):
        f_idx = max(0, BASE_FRAME - k * DELAY_FRAMES)
        f_path = os.path.join(src_p, f"frame_{f_idx:04d}.png")
        if not os.path.exists(f_path):
            continue
        raw_rgba = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        
        # Apply specific effect for this temporal layer
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

        # Draw a sleek marker label at the head of each slice
        head_top_2d = project_point(np.array([0.0, y_top + 0.12, z_k]), R_cv, t_cv)
        if head_top_2d is not None:
            lx, ly = int(head_top_2d[0]), int(head_top_2d[1])
            if 0 < lx < W and 0 < ly < H:
                tag_color = (0, 240, 255) if k == 0 else (180, 200, 220)
                cv2.circle(canvas, (lx, ly), 3, tag_color, -1, cv2.LINE_AA)
                cv2.putText(canvas, f"F{k+1}: {EFFECT_NAMES[k].split(' ')[0]}", (lx - 25, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, tag_color, 1, cv2.LINE_AA)

    # HUD
    cv2.putText(canvas, "4D MOTION SCULPTURE — 6 MULTI-EFFECT TEMPORAL STATES", (36, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "1:ORIGINAL | 2:HOLOGRAPHIC | 3:HEATMAP | 4:LIDAR | 5:GAUSSIAN SPLAT | 6:BLOB", (36, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv2.LINE_AA)

    out_p = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\samples_new\composite_4d_6effects.png"
    cv2.imwrite(out_p, canvas)
    print("Saved composite 4D preview to:", out_p)

if __name__ == "__main__":
    test_composite_frame(45.0)
