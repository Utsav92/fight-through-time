import os
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
MODEL_TASK_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

print("Initializing rembg session...")
rembg_session = new_session("u2net")

print("Initializing mediapipe pose landmarker...")
base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
pose_detector = vision.PoseLandmarker.create_from_options(options)

def process_single_frame(frame_idx, frame_bgr):
    h, w, _ = frame_bgr.shape
    
    # 1. Mask out watermark area before rembg (top-left: y < 65, x < 250)
    clean_input = frame_bgr.copy()
    clean_input[0:65, 0:250] = 0
    
    # 2. rembg foreground isolation
    rgba = np.array(remove(clean_input, session=rembg_session), copy=True)
    alpha = rgba[:, :, 3]
    
    # Mask top-left watermark strictly
    alpha[0:70, 0:260] = 0
    rgba[:, :, 3] = alpha
    
    # 3. Fighter separation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha > 50).astype(np.uint8))
    
    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 800:
            cx, cy = centroids[i]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            comps.append({
                "label": i, "area": area, "cx": cx, "cy": cy,
                "bbox": [x, y, cw, ch]
            })
            
    # Sort left to right
    comps.sort(key=lambda c: c["cx"])
    
    fighter_a_mask = np.zeros((h, w), dtype=np.uint8)
    fighter_b_mask = np.zeros((h, w), dtype=np.uint8)
    
    bbox_a = None
    bbox_b = None
    
    if len(comps) == 1:
        # If single merged blob, split horizontally around centroid
        c = comps[0]
        cx = int(c["cx"])
        labels_mask = (labels == c["label"]).astype(np.uint8) * 255
        fighter_a_mask[:, :cx] = labels_mask[:, :cx]
        fighter_b_mask[:, cx:] = labels_mask[:, cx:]
        bbox_a = [c["bbox"][0], c["bbox"][1], cx - c["bbox"][0], c["bbox"][3]]
        bbox_b = [cx, c["bbox"][1], c["bbox"][0] + c["bbox"][2] - cx, c["bbox"][3]]
    elif len(comps) >= 2:
        # Check pants / lower torso brightness or assign left=A, right=B
        # Typically comps[0] is Bruce Lee (left), comps[1] is Chuck Norris (right)
        fighter_a_mask = (labels == comps[0]["label"]).astype(np.uint8) * 255
        fighter_b_mask = (labels == comps[1]["label"]).astype(np.uint8) * 255
        bbox_a = comps[0]["bbox"]
        bbox_b = comps[1]["bbox"]
        # If more than 2, merge remaining small pieces into nearest
        for extra in comps[2:]:
            dist_a = abs(extra["cx"] - comps[0]["cx"])
            dist_b = abs(extra["cx"] - comps[1]["cx"])
            extra_mask = (labels == extra["label"]).astype(np.uint8) * 255
            if dist_a < dist_b:
                fighter_a_mask = cv2.bitwise_or(fighter_a_mask, extra_mask)
            else:
                fighter_b_mask = cv2.bitwise_or(fighter_b_mask, extra_mask)
                
    # Create isolated RGBA images
    rgba_a = rgba.copy()
    rgba_a[:, :, 3] = np.minimum(rgba_a[:, :, 3], fighter_a_mask)
    
    rgba_b = rgba.copy()
    rgba_b[:, :, 3] = np.minimum(rgba_b[:, :, 3], fighter_b_mask)
    
    # 4. Joint tracking using mediapipe on crops
    def get_joints(bbox):
        if not bbox or bbox[2] < 10 or bbox[3] < 10:
            return None
        x, y, bw, bh = bbox
        pad = 20
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
        
        # Landmark indices:
        # 0: nose, 11: left_shoulder, 12: right_shoulder,
        # 15: left_wrist, 16: right_wrist, 23: left_hip, 24: right_hip,
        # 27: left_ankle, 28: right_ankle
        def lm_to_global(lm):
            gx = x1 + lm.x * crop_w
            gy = y1 + lm.y * crop_h
            # Convert to normalized 3D space: X in [-1.5, 1.5], Y in [0, 2.0]
            norm_x = (gx / w - 0.5) * 3.0
            norm_y = (1.0 - gy / h) * 1.8
            return [round(norm_x, 4), round(norm_y, 4), round(lm.z, 4)]
            
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
    
    return {
        "frame": frame_idx,
        "rgba_combined": rgba,
        "rgba_a": rgba_a,
        "rgba_b": rgba_b,
        "bbox_a": bbox_a,
        "bbox_b": bbox_b,
        "joints_a": joints_a,
        "joints_b": joints_b
    }

cap = cv2.VideoCapture(VIDEO_PATH)
test_frames = [300, 360, 420, 515]
for tf in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, tf)
    ret, frame = cap.read()
    if not ret:
        continue
    t0 = time.time()
    res = process_single_frame(tf, frame)
    t1 = time.time()
    cv2.imwrite(f"samples/test_comb_{tf}.png", res["rgba_combined"])
    cv2.imwrite(f"samples/test_a_{tf}.png", res["rgba_a"])
    cv2.imwrite(f"samples/test_b_{tf}.png", res["rgba_b"])
    print(f"Frame {tf} processed in {t1-t0:.2f}s:")
    print("  Joints A:", "OK" if res["joints_a"] else "None")
    print("  Joints B:", "OK" if res["joints_b"] else "None")
    if res["joints_a"]:
        print("    Fighter A hands:", res["joints_a"]["left_hand"], res["joints_a"]["right_hand"])
    if res["joints_b"]:
        print("    Fighter B hands:", res["joints_b"]["left_hand"], res["joints_b"]["right_hand"])
cap.release()
print("Test completed successfully!")
