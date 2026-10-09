import os
import cv2
import numpy as np
import time
import concurrent.futures
from rembg import remove, new_session

BASE_DIR = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time"
VIDEO_PATH = r"C:\Users\Utsav\Downloads\07.10.2026_04.29.48_REC.mp4"
OUT_DIR = os.path.join(BASE_DIR, "data_dp", "frames_combined")
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Total video frames: {total_frames}, FPS: {fps:.2f}, Res: {w}x{h}")

# Pre-extract all source frames into memory / list to prevent video seeking lock
frames_dict = {}
for i in range(total_frames):
    out_file = os.path.join(OUT_DIR, f"frame_{i:04d}.png")
    if not (os.path.exists(out_file) and os.path.getsize(out_file) > 1000):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_bgr = cap.read()
        if ret:
            frames_dict[i] = frame_bgr

cap.release()
print(f"{len(frames_dict)} frames remaining to segment.")

# Initialize 2 dedicated rembg sessions for 2 workers
session1 = new_session("u2net")
session2 = new_session("u2net")
sessions = [session1, session2]

completed = total_frames - len(frames_dict)
t_start = time.time()

def worker_task(item):
    f_idx, worker_id = item
    out_file = os.path.join(OUT_DIR, f"frame_{f_idx:04d}.png")
    frame_bgr = frames_dict[f_idx]
    sess = sessions[worker_id]
    
    rgba = np.array(remove(frame_bgr, session=sess), copy=True)
    cv2.imwrite(out_file, rgba)
    return f_idx

tasks = [(f_idx, idx % 2) for idx, f_idx in enumerate(sorted(frames_dict.keys()))]

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    for f_idx in executor.map(worker_task, tasks):
        completed += 1
        if completed % 25 == 0 or completed == total_frames:
            elapsed = time.time() - t_start
            done_in_run = completed - (total_frames - len(frames_dict))
            rate = done_in_run / max(0.1, elapsed)
            rem = (total_frames - completed) / max(0.01, rate)
            print(f"Segmented {completed}/{total_frames} frames ({completed/total_frames*100:.1f}%) | Est. remaining: {rem/60:.1f}m")

print("All frames segmented successfully!")
