import os
import sys
import cv2
import numpy as np
import json
import time
from rembg import remove, new_session
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_PATH = r"C:\Users\Utsav\Downloads\07.10.2026_04.25.15_REC.mp4"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAMES_COMB_DIR = os.path.join(DATA_DIR, "frames_combined")
FRAMES_A_DIR = os.path.join(DATA_DIR, "frames_a")
FRAMES_B_DIR = os.path.join(DATA_DIR, "frames_b")
SPRITES_DIR = os.path.join(DATA_DIR, "sprites")

MODEL_TASK_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

START_FRAME = 360
END_FRAME = 530
FPS = 25.028

print(f"Loading rembg u2net session...")
rembg_session = new_session("u2net")

print(f"Loading MediaPipe Pose Landmarker...")
base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
pose_detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(VIDEO_PATH)

telemetry = {
    "title": "FIGHT THROUGH TIME — INTERACTIVE 4D MOTION SCULPTURE",
    "source_video": "07.10.2026_04.25.15_REC.mp4",
    "fighters": {
        "fighter_a": "Bruce Lee (Jeet Kune Do / Dark Silhouette / Cool Temporal Family)",
        "fighter_b": "Chuck Norris (Karate / Light Gi / Warm Temporal Family)"
    },
    "fps": FPS,
    "start_frame": START_FRAME,
    "end_frame": END_FRAME,
    "total_frames": END_FRAME - START_FRAME + 1,
    "default_history_count": 7,
    "default_temporal_delay_sec": 0.40,
    "default_temporal_delay_frames": 10,
    "default_temporal_depth": 1.5,
    "key_actions": [],
    "frames": []
}

# Action markers ground truth detection based on combat choreography:
ACTION_MARKERS = {
    360: {"type": "PUNCH", "label": "Punch Preparation (Bruce Lee advance)", "intensity": 0.7},
    375: {"type": "BLOCK", "label": "High Block (Chuck Norris forearm defense)", "intensity": 0.75},
    395: {"type": "KICK", "label": "Kick Preparation (Chamber stance)", "intensity": 0.8},
    420: {"type": "KICK", "label": "High Roundhouse Kick Extension", "intensity": 0.95},
    438: {"type": "DODGE", "label": "Evasive Slip & Low Duck", "intensity": 0.85},
    462: {"type": "BLOCK", "label": "Wing Chun Trapping & Deflection", "intensity": 0.8},
    485: {"type": "PUNCH", "label": "Rapid Punch Flurry", "intensity": 0.9},
    515: {"type": "IMPACT", "label": "Body Strike Impact Clash", "intensity": 1.0},
    528: {"type": "RECOVERY", "label": "Impact Recovery & Step Back", "intensity": 0.6}
}

print(f"Processing frames {START_FRAME} to {END_FRAME} ({END_FRAME - START_FRAME + 1} frames total)...")

# Keep track of previous fighter centroids to maintain identity tracking
prev_cx_a = 300.0
prev_cx_b = 560.0

all_frames_data = []

for frame_idx in range(START_FRAME, END_FRAME + 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        print(f"Warning: Frame {frame_idx} could not be read.")
        break
        
    t_start = time.time()
    h, w, _ = frame_bgr.shape
    
    # Clean input: mask watermark strictly
    clean_input = frame_bgr.copy()
    clean_input[0:65, 0:250] = 0
    
    # Background removal
    rgba = np.array(remove(clean_input, session=rembg_session), copy=True)
    alpha = rgba[:, :, 3]
    alpha[0:70, 0:260] = 0
    rgba[:, :, 3] = alpha
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha > 50).astype(np.uint8))
    
    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 600:
            cx, cy = centroids[i]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            comps.append({
                "label": i, "area": area, "cx": cx, "cy": cy,
                "bbox": [int(x), int(y), int(cw), int(ch)]
            })
            
    comps.sort(key=lambda c: c["cx"])
    
    fighter_a_mask = np.zeros((h, w), dtype=np.uint8)
    fighter_b_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_a = None
    bbox_b = None
    
    if len(comps) == 0:
        pass
    elif len(comps) == 1:
        c = comps[0]
        # Match with closest previous centroid or split
        mid_x = int(c["cx"])
        labels_mask = (labels == c["label"]).astype(np.uint8) * 255
        fighter_a_mask[:, :mid_x] = labels_mask[:, :mid_x]
        fighter_b_mask[:, mid_x:] = labels_mask[:, mid_x:]
        bbox_a = [c["bbox"][0], c["bbox"][1], mid_x - c["bbox"][0], c["bbox"][3]]
        bbox_b = [mid_x, c["bbox"][1], c["bbox"][0] + c["bbox"][2] - mid_x, c["bbox"][3]]
    else:
        # Match to fighter A (Bruce Lee, historically left/lower cx) and Fighter B
        fighter_a_mask = (labels == comps[0]["label"]).astype(np.uint8) * 255
        fighter_b_mask = (labels == comps[1]["label"]).astype(np.uint8) * 255
        bbox_a = comps[0]["bbox"]
        bbox_b = comps[1]["bbox"]
        prev_cx_a = comps[0]["cx"]
        prev_cx_b = comps[1]["cx"]
        for extra in comps[2:]:
            dist_a = abs(extra["cx"] - prev_cx_a)
            dist_b = abs(extra["cx"] - prev_cx_b)
            extra_mask = (labels == extra["label"]).astype(np.uint8) * 255
            if dist_a < dist_b:
                fighter_a_mask = cv2.bitwise_or(fighter_a_mask, extra_mask)
            else:
                fighter_b_mask = cv2.bitwise_or(fighter_b_mask, extra_mask)
                
    rgba_a = rgba.copy()
    rgba_a[:, :, 3] = np.minimum(rgba_a[:, :, 3], fighter_a_mask)
    
    rgba_b = rgba.copy()
    rgba_b[:, :, 3] = np.minimum(rgba_b[:, :, 3], fighter_b_mask)
    
    # Save isolated images
    comb_path = os.path.join(FRAMES_COMB_DIR, f"frame_{frame_idx:04d}.png")
    a_path = os.path.join(FRAMES_A_DIR, f"frame_{frame_idx:04d}.png")
    b_path = os.path.join(FRAMES_B_DIR, f"frame_{frame_idx:04d}.png")
    
    cv2.imwrite(comb_path, rgba)
    cv2.imwrite(a_path, rgba_a)
    cv2.imwrite(b_path, rgba_b)
    
    # Pose estimation
    def get_joints(bbox):
        if not bbox or bbox[2] < 15 or bbox[3] < 15:
            return None
        x, y, bw, bh = bbox
        pad = 25
        y1, y2 = max(0, y - pad), min(h, y + bh + pad)
        x1, x2 = max(0, x - pad), min(w, x + bw + pad)
        crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        res = pose_detector.detect(mp_img)
        if not res.pose_landmarks or len(res.pose_landmarks) == 0:
            return None
        lms = res.pose_landmarks[0]
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        def lm_to_global(lm):
            gx = x1 + lm.x * crop_w
            gy = y1 + lm.y * crop_h
            # Map full frame to 3D coordinate system:
            # X: [-2.2, 2.2], Y: [0.0, 2.0], Z: forward/back offset
            norm_x = round((gx / w - 0.5) * 4.4, 4)
            norm_y = round((1.0 - gy / h) * 2.0, 4)
            norm_z = round(-lm.z * 0.5, 4)
            return [norm_x, norm_y, norm_z]
            
        return {
            "head": lm_to_global(lms[0]),
            "left_hand": lm_to_global(lms[15]),
            "right_hand": lm_to_global(lms[16]),
            "left_foot": lm_to_global(lms[27]),
            "right_foot": lm_to_global(lms[28]),
            "torso": lm_to_global(lms[23])
        }

    joints_a = get_joints(bbox_a)
    joints_b = get_joints(bbox_b)
    
    # Calculate collision / contact points
    contacts = []
    if joints_a and joints_b:
        # Check hand-to-hand, hand-to-torso, foot-to-torso, foot-to-foot distances
        pairs = [
            ("HAND_CLASH", joints_a["left_hand"], joints_b["right_hand"]),
            ("HAND_CLASH", joints_a["right_hand"], joints_b["left_hand"]),
            ("KICK_STRIKE", joints_a["left_foot"], joints_b["torso"]),
            ("KICK_STRIKE", joints_b["right_foot"], joints_a["torso"]),
            ("PUNCH_IMPACT", joints_a["right_hand"], joints_b["head"]),
            ("PUNCH_IMPACT", joints_b["right_hand"], joints_a["head"]),
        ]
        for ctype, pt_a, pt_b in pairs:
            dist = np.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)
            if dist < 0.45: # Proximity in normalized space
                mid_x = round((pt_a[0] + pt_b[0]) * 0.5, 4)
                mid_y = round((pt_a[1] + pt_b[1]) * 0.5, 4)
                mid_z = round((pt_a[2] + pt_b[2]) * 0.5, 4)
                intensity = round(float(max(0.2, 1.0 - dist / 0.45)), 3)
                contacts.append({
                    "type": ctype,
                    "pos": [mid_x, mid_y, mid_z],
                    "intensity": intensity
                })

    # Sample 3D point cloud particles (downsampled foreground)
    # Step of 4 pixels gives crisp silhouettes (~600-900 points per fighter)
    fg_indices = np.where(alpha > 80)
    pts_sample = []
    if len(fg_indices[0]) > 0:
        step = 6
        sub_y = fg_indices[0][::step]
        sub_x = fg_indices[1][::step]
        for py, px in zip(sub_y, sub_x):
            nx = round((px / w - 0.5) * 4.4, 3)
            ny = round((1.0 - py / h) * 2.0, 3)
            b, g, r = frame_bgr[py, px]
            is_fighter_b = 1 if px > (prev_cx_a + prev_cx_b) * 0.5 else 0
            pts_sample.append([nx, ny, int(r), int(g), int(b), is_fighter_b])
            
    # Check if this frame has an action marker
    key_action = None
    if frame_idx in ACTION_MARKERS:
        key_action = ACTION_MARKERS[frame_idx]
        telemetry["key_actions"].append({
            "frame": frame_idx,
            "time": round((frame_idx - START_FRAME) / FPS, 2),
            **key_action
        })
        
    frame_record = {
        "frame": frame_idx,
        "time": round((frame_idx - START_FRAME) / FPS, 3),
        "abs_time": round(frame_idx / FPS, 3),
        "bbox_a": bbox_a,
        "bbox_b": bbox_b,
        "joints_a": joints_a,
        "joints_b": joints_b,
        "contacts": contacts,
        "key_action": key_action,
        "point_count": len(pts_sample),
        "points": pts_sample # [x, y, r, g, b, is_fighter_b]
    }
    
    all_frames_data.append(frame_record)
    dt = time.time() - t_start
    if frame_idx % 10 == 0 or frame_idx == START_FRAME:
        print(f"Frame {frame_idx:04d} ({frame_record['time']:.2f}s) processed in {dt:.2f}s | Points: {len(pts_sample)} | Contacts: {len(contacts)}")

telemetry["frames"] = all_frames_data
cap.release()

# Save telemetry json
telemetry_path = os.path.join(DATA_DIR, "fight_telemetry.json")
with open(telemetry_path, "w") as f:
    json.dump(telemetry, f)
print(f"Saved complete telemetry to {telemetry_path}")

# Build compact sprite sheets for instant 60 FPS WebGL texture sampling!
# We pack 10x10 = 100 frames per atlas at 218x100 resolution (half aspect)
print("Generating sprite atlases for Three.js WebGL viewer...")
sprite_w = 218
sprite_h = 100
grid_cols = 10
grid_rows = 10
frames_per_sheet = grid_cols * grid_rows

sheet_idx = 0
for start_i in range(0, len(all_frames_data), frames_per_sheet):
    sheet_frames = all_frames_data[start_i:start_i + frames_per_sheet]
    atlas = np.zeros((grid_rows * sprite_h, grid_cols * sprite_w, 4), dtype=np.uint8)
    atlas_a = np.zeros((grid_rows * sprite_h, grid_cols * sprite_w, 4), dtype=np.uint8)
    atlas_b = np.zeros((grid_rows * sprite_h, grid_cols * sprite_w, 4), dtype=np.uint8)
    
    for sub_i, fr in enumerate(sheet_frames):
        f_idx = fr["frame"]
        col = sub_i % grid_cols
        row = sub_i // grid_cols
        
        c_img = cv2.imread(os.path.join(FRAMES_COMB_DIR, f"frame_{f_idx:04d}.png"), cv2.IMREAD_UNCHANGED)
        a_img = cv2.imread(os.path.join(FRAMES_A_DIR, f"frame_{f_idx:04d}.png"), cv2.IMREAD_UNCHANGED)
        b_img = cv2.imread(os.path.join(FRAMES_B_DIR, f"frame_{f_idx:04d}.png"), cv2.IMREAD_UNCHANGED)
        
        c_small = cv2.resize(c_img, (sprite_w, sprite_h), interpolation=cv2.INTER_AREA)
        a_small = cv2.resize(a_img, (sprite_w, sprite_h), interpolation=cv2.INTER_AREA)
        b_small = cv2.resize(b_img, (sprite_w, sprite_h), interpolation=cv2.INTER_AREA)
        
        y0, y1 = row * sprite_h, (row + 1) * sprite_h
        x0, x1 = col * sprite_w, (col + 1) * sprite_w
        
        atlas[y0:y1, x0:x1] = c_small
        atlas_a[y0:y1, x0:x1] = a_small
        atlas_b[y0:y1, x0:x1] = b_small
        
    cv2.imwrite(os.path.join(SPRITES_DIR, f"atlas_combined_{sheet_idx}.png"), atlas)
    cv2.imwrite(os.path.join(SPRITES_DIR, f"atlas_fighter_a_{sheet_idx}.png"), atlas_a)
    cv2.imwrite(os.path.join(SPRITES_DIR, f"atlas_fighter_b_{sheet_idx}.png"), atlas_b)
    sheet_idx += 1

print(f"Generated {sheet_idx} sprite atlas sheets!")
print("Dataset build complete!")
