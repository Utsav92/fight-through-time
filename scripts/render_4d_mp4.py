import os
import sys
import cv2
import numpy as np
import subprocess
import time
import shutil

BASE_DIR = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time"
FRAMES_DIR = os.path.join(BASE_DIR, "data", "frames_combined")
TEMP_RENDER_DIR = os.path.join(BASE_DIR, "render_frames")
os.makedirs(TEMP_RENDER_DIR, exist_ok=True)

DOWNLOADS_DIR = r"C:\Users\Utsav\Downloads"
FINAL_MP4_DOWNLOADS = os.path.join(DOWNLOADS_DIR, "FIGHT_THROUGH_TIME_4D_MOTION_SCULPTURE.mp4")
FINAL_MP4_LOCAL = os.path.join(BASE_DIR, "FIGHT_THROUGH_TIME_4D_MOTION_SCULPTURE.mp4")

# Output Specifications
W, H = 1920, 1080
FPS = 25.0
TOTAL_FRAMES = 450 # 18 seconds of cinematic 4D motion sculpture

# Camera Intrinsics
FOV = 42.0
F_LEN = (W / 2) / np.tan(np.radians(FOV / 2))
K = np.array([
    [F_LEN, 0, W / 2],
    [0, F_LEN, H / 2],
    [0, 0, 1]
])

TARGET_POINT = np.array([0.0, 0.95, -3.8])
UP_VECTOR = np.array([0.0, 1.0, 0.0])

HISTORY_COUNT = 7
DELAY_FRAMES = 10
TIME_DEPTH = 1.35
OPACITY_LADDER = [1.0, 0.82, 0.68, 0.55, 0.43, 0.32, 0.22]

AURA_A = np.array([255, 229, 0], dtype=np.float32)  # Bruce Lee: Cyan BGR
AURA_B = np.array([0, 145, 255], dtype=np.float32)  # Chuck Norris: Amber BGR

# Preload frames in range [360, 530]
print("Pre-caching segmented frames into RAM...")
cached_frames = {}
cached_alphas = {}
for f_idx in range(360, 531):
    p = os.path.join(FRAMES_DIR, f"frame_{f_idx:04d}.png")
    if os.path.exists(p):
        img_rgba = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        cached_frames[f_idx] = img_rgba[:, :, :3].astype(np.float32)
        cached_alphas[f_idx] = (img_rgba[:, :, 3].astype(np.float32) / 255.0)

print(f"Loaded {len(cached_frames)} frames.")

# Precompute static noise buffer for particulate dispersion
np.random.seed(42)
STATIC_NOISE = np.random.uniform(0.68, 1.0, (H, W)).astype(np.float32)
X_GRID = np.tile(np.arange(W), (H, 1))

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

print("Rendering 4D motion sculpture cinematic sequence...")
t_start = time.time()

half_w = 2.185
y_bottom = 0.0
y_top = 2.0
src_pts = np.float32([[0, 0], [874, 0], [874, 400], [0, 400]])

for render_i in range(TOTAL_FRAMES):
    # Camera Choreography across 450 frames:
    if render_i <= 60:
        # Act 1: 0° front chronophotography view
        theta_deg = 0.0
        cam_dist = 8.8
        cam_height = 1.0
        prog = render_i / 60.0
        src_video_frame = int(360 + prog * 35)
        phase_label = "0° CHRONOPHOTOGRAPHY"
    elif render_i <= 180:
        # Act 2: 0° -> 45° 4D Spatial Reveal
        prog = (render_i - 60) / 120.0
        smooth_prog = prog * prog * (3 - 2 * prog)
        theta_deg = smooth_prog * 45.0
        cam_dist = 8.8 - smooth_prog * 0.3
        cam_height = 1.0 + smooth_prog * 0.7
        src_video_frame = int(395 + prog * 85)
        phase_label = f"{theta_deg:.0f}° 4D SPATIAL REVEAL"
    elif render_i <= 250:
        # Act 3: 45° -> 72° Time Profile
        prog = (render_i - 180) / 70.0
        smooth_prog = prog * prog * (3 - 2 * prog)
        theta_deg = 45.0 + smooth_prog * 27.0 # Clamped to 72° for optimal perspective
        cam_dist = 8.5
        cam_height = 1.7 + smooth_prog * 0.3
        src_video_frame = int(480 + prog * 35)
        phase_label = f"{theta_deg:.0f}° TIME AXIS PROFILE"
    elif render_i <= 360:
        # Act 4: FREEZE 4D — SPATIAL ORBIT at F515 Impact
        prog = (render_i - 250) / 110.0
        smooth_prog = prog * prog * (3 - 2 * prog)
        theta_deg = 72.0 - smooth_prog * 57.0 # Orbits 72° -> 15°
        cam_dist = 8.5 - np.sin(prog * np.pi) * 0.8
        cam_height = 2.0 - smooth_prog * 0.6
        src_video_frame = 515 # Frozen climax
        phase_label = "FREEZE 4D — SPATIAL ORBIT"
    else:
        # Act 5: 4D Release & dynamic recovery
        prog = (render_i - 360) / 90.0
        smooth_prog = prog * prog * (3 - 2 * prog)
        theta_deg = 15.0 + smooth_prog * 20.0
        cam_dist = 8.2
        cam_height = 1.4
        src_video_frame = int(515 + prog * 15)
        phase_label = "4D MOTION RESOLUTION"

    # Compute camera extrinsics
    theta_rad = np.radians(theta_deg)
    cam_pos = np.array([
        np.sin(theta_rad) * cam_dist,
        cam_height,
        np.cos(theta_rad) * cam_dist - 3.8
    ])
    R_cv, t_cv = get_camera_extrinsics(cam_pos, TARGET_POINT)

    # 1. Canvas: deep obsidian background
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (8, 6, 5)

    # 2. Subtle 3D perspective ground grid
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

    # 3. Render 7 Historical Temporal Poses (k = 6 down to 0)
    for k in range(HISTORY_COUNT - 1, -1, -1):
        f_num = src_video_frame - k * DELAY_FRAMES
        f_num = max(360, min(530, f_num))
        if f_num not in cached_frames:
            continue

        raw_rgb = cached_frames[f_num]
        raw_alpha = cached_alphas[f_num]
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

        warped_rgb = cv2.warpPerspective(raw_rgb, H_mat, (W, H), flags=cv2.INTER_LINEAR)
        warped_alpha = cv2.warpPerspective(raw_alpha, H_mat, (W, H), flags=cv2.INTER_LINEAR)

        effective_alpha = warped_alpha * OPACITY_LADDER[k]

        if k > 0:
            mid_x = int(np.mean([c[0] for c in corners_2d]))
            weight_b = np.clip((X_GRID - (mid_x - 120)) / 240.0, 0.0, 1.0)[:, :, None]
            aura = AURA_A * (1.0 - weight_b) + AURA_B * weight_b

            tint = 0.22 + (k / 6.0) * 0.40
            warped_rgb = warped_rgb * (1.0 - tint) + aura * tint

            if k >= 3:
                effective_alpha *= STATIC_NOISE

        alpha_3 = effective_alpha[:, :, None]
        canvas = (warped_rgb * alpha_3 + canvas.astype(np.float32) * (1.0 - alpha_3)).astype(np.uint8)

    # 4. Cinematic Minimal Typography (No 3D time axis line!)
    cv2.putText(canvas, "FIGHT THROUGH TIME — 4D MOTION SCULPTURE", (50, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "BRUCE LEE vs. CHUCK NORRIS (1972) | ROMAN COLOSSEUM", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 170), 1, cv2.LINE_AA)

    src_sec = src_video_frame / FPS
    cv2.putText(canvas, f"CAMERA: {phase_label}", (W - 460, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FRAME: {src_video_frame} | TIME: {src_sec:.2f}s | STATES: 7 | DELAY: 0.40s", (W - 460, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 175, 190), 1, cv2.LINE_AA)

    cv2.line(canvas, (50, H - 45), (W - 50, H - 45), (40, 45, 55), 1, cv2.LINE_AA)
    cv2.putText(canvas, "FIGHTER A: BRUCE LEE (COOL TEMPORAL AURA)", (50, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 229, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "FIGHTER B: CHUCK NORRIS (WARM TEMPORAL AURA)", (390, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 145, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DEPTH: Z = TIME x 1.35m | OPACITY: 100% -> 22% DISPERSION", (W - 480, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 150, 165), 1, cv2.LINE_AA)

    frame_out_path = os.path.join(TEMP_RENDER_DIR, f"frame_{render_i:04d}.png")
    cv2.imwrite(frame_out_path, canvas)

    if render_i % 75 == 0 or render_i == TOTAL_FRAMES - 1:
        print(f"Rendered frame {render_i:04d}/{TOTAL_FRAMES} ({phase_label})")

dt = time.time() - t_start
print(f"Rendered all {TOTAL_FRAMES} frames in {dt:.1f}s ({(TOTAL_FRAMES/dt):.1f} FPS)!")

# FFmpeg Encode to MP4
print(f"Encoding MP4 to {FINAL_MP4_LOCAL}...")
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", os.path.join(TEMP_RENDER_DIR, "frame_%04d.png"),
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    FINAL_MP4_LOCAL
]
subprocess.run(ffmpeg_cmd, check=True)

# Copy directly to Downloads
shutil.copyfile(FINAL_MP4_LOCAL, FINAL_MP4_DOWNLOADS)
file_size_mb = os.path.getsize(FINAL_MP4_DOWNLOADS) / (1024 * 1024)
print(f"SUCCESS: MP4 delivered to Downloads: {FINAL_MP4_DOWNLOADS} ({file_size_mb:.2f} MB)")
