import os
import sys
import cv2
import numpy as np
import subprocess
import time
import shutil

BASE_DIR = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time"
FRAMES_DIR = os.path.join(BASE_DIR, "data_dp", "frames_combined")
DOWNLOADS_DIR = r"C:\Users\Utsav\Downloads"
FINAL_MP4_DOWNLOADS = os.path.join(DOWNLOADS_DIR, "DEADPOOL_WOLVERINE_4D_MOTION_SCULPTURE.mp4")
FINAL_MP4_LOCAL = os.path.join(BASE_DIR, "DEADPOOL_WOLVERINE_4D_MOTION_SCULPTURE.mp4")

# Output Specs
W, H = 1280, 720
FPS = 25.0
TOTAL_FRAMES = 405 # Entire clip: 16.19 seconds

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
DELAY_FRAMES = 10 # 0.40 seconds
TIME_DEPTH = 1.35 # meters per 0.4s
OPACITY_LADDER = [1.0, 0.82, 0.68, 0.55, 0.43, 0.32, 0.22]

AURA_DEADPOOL = np.array([40, 40, 255], dtype=np.float32)  # Deadpool Crimson BGR
AURA_WOLVERINE = np.array([0, 185, 255], dtype=np.float32) # Wolverine Gold BGR

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

def render_movie():
    print("Pre-caching segmented frames into RAM...")
    cached_rgba = {}
    for f_idx in range(TOTAL_FRAMES):
        p = os.path.join(FRAMES_DIR, f"frame_{f_idx:04d}.png")
        if os.path.exists(p):
            cached_rgba[f_idx] = cv2.imread(p, cv2.IMREAD_UNCHANGED)

    print(f"Loaded {len(cached_rgba)} frames from disk.")

    np.random.seed(42)
    STATIC_NOISE = np.random.uniform(0.68, 1.0, (H, W)).astype(np.float32)
    X_GRID = np.tile(np.arange(W), (H, 1))

    # Launch FFmpeg stream encoder
    print(f"Launching FFmpeg encoder to {FINAL_MP4_LOCAL}...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        FINAL_MP4_LOCAL
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    half_w = 2.38 # 890:374 aspect ratio
    y_bottom = 0.0
    y_top = 2.0
    src_pts = np.float32([[0, 0], [890, 0], [890, 374], [0, 374]])

    t_start = time.time()
    for render_i in range(TOTAL_FRAMES):
        # Camera Choreography across the entire 405 frames:
        if render_i <= 60:
            # Act 1: 0° front view (Chronophotography baseline)
            theta_deg = 0.0
            cam_dist = 8.8
            cam_height = 1.0
            phase_label = "0° CHRONOPHOTOGRAPHY"
        elif render_i <= 180:
            # Act 2: 0° -> 45° 4D Spatial Reveal
            prog = (render_i - 60) / 120.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = smooth_prog * 45.0
            cam_dist = 8.8 - smooth_prog * 0.3
            cam_height = 1.0 + smooth_prog * 0.7
            phase_label = f"{theta_deg:.0f}° 4D SPATIAL REVEAL"
        elif render_i <= 260:
            # Act 3: 45° -> 70° Profile of Time
            prog = (render_i - 180) / 80.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = 45.0 + smooth_prog * 25.0
            cam_dist = 8.5
            cam_height = 1.7 + smooth_prog * 0.3
            phase_label = f"{theta_deg:.0f}° TIME AXIS PROFILE"
        elif render_i <= 340:
            # Act 4: FREEZE 4D — SPATIAL ORBIT around peak clash
            prog = (render_i - 260) / 80.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = 70.0 - smooth_prog * 55.0
            cam_dist = 8.5 - np.sin(prog * np.pi) * 0.8
            cam_height = 2.0 - smooth_prog * 0.6
            phase_label = "FREEZE 4D — SPATIAL ORBIT"
        else:
            # Act 5: 4D Release & resolution
            prog = (render_i - 340) / 65.0
            smooth_prog = prog * prog * (3 - 2 * prog)
            theta_deg = 15.0 + smooth_prog * 20.0
            cam_dist = 8.2
            cam_height = 1.4
            phase_label = "4D COMBAT RESOLUTION"

        src_video_frame = render_i # Entire video mapped 1:1 to time!

        theta_rad = np.radians(theta_deg)
        cam_pos = np.array([
            np.sin(theta_rad) * cam_dist,
            cam_height,
            np.cos(theta_rad) * cam_dist - 3.8
        ])
        R_cv, t_cv = get_camera_extrinsics(cam_pos, TARGET_POINT)

        # 1. Deep obsidian canvas
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (8, 6, 5)

        # 2. Subtle 3D floor perspective grid
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
            f_num = max(0, src_video_frame - k * DELAY_FRAMES)
            if f_num not in cached_rgba:
                continue

            rgba = cached_rgba[f_num]
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
            warped = cv2.warpPerspective(rgba, H_mat, (W, H), flags=cv2.INTER_LINEAR)

            w_rgb = warped[:, :, :3].astype(np.float32)
            w_alpha = (warped[:, :, 3].astype(np.float32) / 255.0) * OPACITY_LADDER[k]

            if k > 0:
                mid_x = int(np.mean([c[0] for c in corners_2d]))
                weight_w = np.clip((X_GRID - (mid_x - 60)) / 120.0, 0.0, 1.0)[:, :, None]
                aura = AURA_DEADPOOL * (1.0 - weight_w) + AURA_WOLVERINE * weight_w

                tint = 0.22 + (k / 6.0) * 0.38
                w_rgb = w_rgb * (1.0 - tint) + aura * tint

                if k >= 3:
                    w_alpha *= STATIC_NOISE

            alpha_3 = w_alpha[:, :, None]
            canvas = (w_rgb * alpha_3 + canvas.astype(np.float32) * (1.0 - alpha_3)).astype(np.uint8)

        # 4. Minimal HUD Typography
        cv2.putText(canvas, "DEADPOOL & WOLVERINE — 4D MOTION SCULPTURE", (36, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "THE VOID (2024) | ENTIRE 16.2s CLIP | 7 TEMPORAL STATES", (36, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 170), 1, cv2.LINE_AA)

        src_sec = src_video_frame / FPS
        cv2.putText(canvas, f"CAMERA: {phase_label}", (W - 360, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FRAME: {src_video_frame:03d}/{TOTAL_FRAMES} | TIME: {src_sec:.2f}s | DELAY: 0.40s", (W - 360, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 175, 190), 1, cv2.LINE_AA)

        cv2.line(canvas, (36, H - 34), (W - 36, H - 34), (40, 45, 55), 1, cv2.LINE_AA)
        cv2.putText(canvas, "FIGHTER A: DEADPOOL (CRIMSON AURA / DUAL KATANAS)", (36, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40, 40, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "FIGHTER B: WOLVERINE (GOLD AURA / ADAMANTIUM CLAWS)", (380, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 185, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "DEPTH: Z = TIME x 1.35m | OPACITY: 100% -> 22%", (W - 360, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 150, 165), 1, cv2.LINE_AA)

        proc.stdin.write(canvas.tobytes())

        if render_i % 50 == 0 or render_i == TOTAL_FRAMES - 1:
            print(f"Rendered frame {render_i:04d}/{TOTAL_FRAMES} ({phase_label})")

    proc.stdin.close()
    proc.wait()

    dt = time.time() - t_start
    print(f"Render & encode completed in {dt:.1f}s ({(TOTAL_FRAMES/dt):.1f} FPS)!")

    shutil.copyfile(FINAL_MP4_LOCAL, FINAL_MP4_DOWNLOADS)
    file_size_mb = os.path.getsize(FINAL_MP4_DOWNLOADS) / (1024 * 1024)
    print(f"SUCCESS: MP4 delivered to Downloads: {FINAL_MP4_DOWNLOADS} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    render_movie()
