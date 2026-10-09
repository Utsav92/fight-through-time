import os
import sys
import cv2
import numpy as np
import subprocess
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

sys.path.append(r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\scripts")
from test_effects import (
    effect_original,
    effect_holographic,
    effect_heatmap,
    effect_lidar,
    effect_gaussian_splat,
    effect_blob
)

BASE_DIR = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time"
FRAMES_DIR = os.path.join(BASE_DIR, "data", "frames_combined")
DOWNLOADS_DIR = r"C:\Users\Utsav\Downloads"

# 9:16 Vertical Native Resolution
RENDER_W, RENDER_H = 720, 1280
OUTPUT_W, OUTPUT_H = 1080, 1920
FPS = 25.0
TOTAL_FRAMES = 400 # 16.0 seconds

# Vertical Camera Intrinsics
FOV_V = 52.0
F_LEN = (RENDER_H / 2) / np.tan(np.radians(FOV_V / 2))
K = np.array([
    [F_LEN, 0, RENDER_W / 2 + 35],
    [0, F_LEN, RENDER_H / 2 - 20],
    [0, 0, 1]
])

TARGET_POINT = np.array([0.15, 0.95, -3.6])
UP_VECTOR = np.array([0.0, 1.0, 0.0])

HISTORY_COUNT = 6
DELAY_FRAMES = 10
TIME_DEPTH = 1.35
OPACITY_LADDER = [1.0, 0.90, 0.78, 0.66, 0.55, 0.45]

EFFECT_FUNCS = [
    effect_original,       # Frame 1: Original
    effect_holographic,    # Frame 2: Holographic
    effect_heatmap,        # Frame 3: Heatmap
    effect_lidar,          # Frame 4: LiDAR
    effect_gaussian_splat, # Frame 5: Gaussian Splat
    effect_blob            # Frame 6: Blob
]

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

def render_bruce_lee_reels(out_filename="BRUCE_LEE_4D_MULTI_EFFECTS_REELS_9x16.mp4"):
    local_mp4 = os.path.join(BASE_DIR, out_filename)
    downloads_mp4 = os.path.join(DOWNLOADS_DIR, out_filename)

    print("Pre-caching Bruce Lee segmented frames into RAM...")
    raw_frames = {}
    for f_idx in range(360, 531):
        p = os.path.join(FRAMES_DIR, f"frame_{f_idx:04d}.png")
        if os.path.exists(p):
            raw_frames[f_idx] = cv2.imread(p, cv2.IMREAD_UNCHANGED)

    print(f"Loaded {len(raw_frames)} frames.")
    styled_cache = {}

    print(f"Launching FFmpeg encoder (scaling to {OUTPUT_W}x{OUTPUT_H} 9:16 Reels)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{RENDER_W}x{RENDER_H}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-vf", f"scale={OUTPUT_W}:{OUTPUT_H}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        local_mp4
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    half_w = 2.185
    y_bottom = 0.0
    y_top = 2.0
    src_pts = np.float32([[0, 0], [874, 0], [874, 400], [0, 400]])
    pool = ThreadPoolExecutor(max_workers=6)

    t_start = time.time()
    for render_i in range(TOTAL_FRAMES):
        # 5-Act Vertical Camera Choreography for Bruce Lee:
        if render_i <= 50:
            # Act 1: 0° front view (Chronophotography baseline)
            theta_deg = 0.0
            cam_dist = 9.4
            cam_height = 2.0
            prog = render_i / 50.0
            src_video_frame = int(360 + prog * 35)
            phase_label = "0° CHRONOPHOTOGRAPHY"
        elif render_i <= 160:
            # Act 2: 0° -> 38° 4D Spatial Reveal
            prog = (render_i - 50) / 110.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = smooth_prog * 38.0
            cam_dist = 9.4 + smooth_prog * 0.2
            cam_height = 2.0 + smooth_prog * 0.8
            src_video_frame = int(395 + prog * 85)
            phase_label = f"{theta_deg:.0f}° 4D SPATIAL REVEAL"
        elif render_i <= 230:
            # Act 3: 38° -> 56° Profile of Time
            prog = (render_i - 160) / 70.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = 38.0 + smooth_prog * 18.0
            cam_dist = 9.6 - smooth_prog * 0.3
            cam_height = 2.8 + smooth_prog * 0.2
            src_video_frame = int(480 + prog * 35)
            phase_label = f"{theta_deg:.0f}° TIME AXIS PROFILE"
        elif render_i <= 330:
            # Act 4: FREEZE 4D — SPATIAL ORBIT at climax kick
            prog = (render_i - 230) / 100.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = 56.0 - smooth_prog * 45.0
            cam_dist = 9.4 - np.sin(prog * np.pi) * 0.6
            cam_height = 3.0 - smooth_prog * 0.7
            src_video_frame = 515
            phase_label = "FREEZE 4D — SPATIAL ORBIT"
        else:
            # Act 5: 4D Release & recovery
            prog = (render_i - 330) / 70.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = 11.0 + smooth_prog * 24.0
            cam_dist = 9.2
            cam_height = 2.3
            src_video_frame = int(515 + prog * 15)
            phase_label = "4D MOTION RESOLUTION"

        theta_rad = np.radians(theta_deg)
        cam_pos = np.array([
            np.sin(theta_rad) * cam_dist,
            cam_height,
            np.cos(theta_rad) * cam_dist - 3.6
        ])
        R_cv, t_cv = get_camera_extrinsics(cam_pos, TARGET_POINT)

        # 1. Canvas: deep obsidian background
        canvas = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        canvas[:] = (8, 6, 5)

        # 2. 3D floor perspective grid
        grid_color = (36, 30, 24)
        grid_accent = (75, 62, 42)
        for gz in np.linspace(-12.0, 2.0, 32):
            p1 = project_point(np.array([-6.0, 0.0, gz]), R_cv, t_cv)
            p2 = project_point(np.array([6.0, 0.0, gz]), R_cv, t_cv)
            if p1 is not None and p2 is not None:
                col = grid_accent if abs(gz) < 0.1 or abs(gz + 8.0) < 0.1 else grid_color
                cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 1, cv2.LINE_AA)

        for gx in np.linspace(-5.0, 5.0, 21):
            p1 = project_point(np.array([gx, 0.0, -12.0]), R_cv, t_cv)
            p2 = project_point(np.array([gx, 0.0, 2.0]), R_cv, t_cv)
            if p1 is not None and p2 is not None:
                col = grid_accent if abs(gx) < 0.1 else grid_color
                cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 1, cv2.LINE_AA)

        # 3. Retrieve or compute 6 styled temporal frames
        to_compute = []
        for k in range(HISTORY_COUNT):
            f_num = max(360, min(530, src_video_frame - k * DELAY_FRAMES))
            key = (k, f_num)
            if key not in styled_cache and f_num in raw_frames:
                to_compute.append((k, f_num, EFFECT_FUNCS[k], raw_frames[f_num]))

        if to_compute:
            tasks = [pool.submit(fn, raw) for _, _, fn, raw in to_compute]
            for (k, f_num, _, _), task in zip(to_compute, tasks):
                styled_cache[(k, f_num)] = task.result()

        # Evict stale cached frames to keep RAM low
        stale_keys = [k for k in list(styled_cache.keys()) if k[1] < src_video_frame - 65]
        for sk in stale_keys:
            del styled_cache[sk]

        # Render 6 Historical Poses back-to-front
        for k in range(HISTORY_COUNT - 1, -1, -1):
            f_num = max(360, min(530, src_video_frame - k * DELAY_FRAMES))
            key = (k, f_num)
            if key not in styled_cache:
                continue

            styled_rgba = styled_cache[key]
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
            warped = cv2.warpPerspective(styled_rgba, H_mat, (RENDER_W, RENDER_H), flags=cv2.INTER_LINEAR)

            w_a = (warped[:, :, 3].astype(np.float32) / 255.0) * OPACITY_LADDER[k]
            mask = w_a > 0.02
            if np.any(mask):
                a_pix = w_a[mask, None]
                c_pix = canvas[mask].astype(np.float32)
                w_pix = warped[mask, :3].astype(np.float32)
                canvas[mask] = np.clip(w_pix * a_pix + c_pix * (1.0 - a_pix), 0, 255).astype(np.uint8)

        # 4. Sleek Typography for Instagram Reels Safe Zone
        cv2.putText(canvas, "BRUCE LEE vs CHUCK NORRIS", (RENDER_W // 2 - 245, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "4D MOTION SCULPTURE", (RENDER_W // 2 - 195, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 235, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "PAST  <====  TIME AXIS  ====>  FUTURE", (RENDER_W // 2 - 210, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 180, 200), 1, cv2.LINE_AA)

        # Bottom Info Bar
        legend_y = RENDER_H - 240
        cv2.line(canvas, (40, legend_y), (RENDER_W - 40, legend_y), (60, 70, 85), 1, cv2.LINE_AA)
        cv2.putText(canvas, "6 TEMPORAL STATES (Z = TIME x 1.35m)", (RENDER_W // 2 - 190, legend_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "1:ORIGINAL | 2:HOLO | 3:HEATMAP", (RENDER_W // 2 - 175, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "4:LIDAR | 5:3D GAUSSIAN | 6:BLOB", (RENDER_W // 2 - 175, legend_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 240, 255), 1, cv2.LINE_AA)

        proc.stdin.write(canvas.tobytes())

        if render_i % 40 == 0 or render_i == TOTAL_FRAMES - 1:
            print(f"Rendered frame {render_i:04d}/{TOTAL_FRAMES} ({phase_label}) - Active cache: {len(styled_cache)}")

    pool.shutdown()
    proc.stdin.close()
    proc.wait()

    dt = time.time() - t_start
    print(f"Render & encode completed in {dt:.1f}s ({(TOTAL_FRAMES/dt):.1f} FPS)!")

    shutil.copyfile(local_mp4, downloads_mp4)
    file_size_mb = os.path.getsize(downloads_mp4) / (1024 * 1024)
    print(f"SUCCESS: Bruce Lee 9:16 Reels delivered to Downloads: {downloads_mp4} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    render_bruce_lee_reels()
