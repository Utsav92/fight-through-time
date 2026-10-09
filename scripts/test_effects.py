import os
import cv2
import numpy as np

def effect_original(rgba):
    return rgba.copy()

def effect_holographic(rgba):
    """Hologram: Vibrant cyan/teal glow, horizontal scanlines, chromatic aberration, edge glow."""
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # 1. Fast chromatic aberration
    shift = 4
    shifted_rgb = np.empty_like(rgb)
    shifted_rgb[:, :, 0] = np.roll(rgb[:, :, 0], -shift, axis=1) # Blue
    shifted_rgb[:, :, 1] = rgb[:, :, 1]                          # Green
    shifted_rgb[:, :, 2] = np.roll(rgb[:, :, 2], shift, axis=1)  # Red

    # 2. Electric Cyan Tint via 1D LUT or fast float
    lum = cv2.cvtColor(shifted_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Cyan: [255, 235, 30]
    holo_rgb = np.empty_like(rgb, dtype=np.float32)
    holo_rgb[:, :, 0] = lum * 255.0
    holo_rgb[:, :, 1] = lum * 235.0
    holo_rgb[:, :, 2] = lum * 35.0

    # 3. Fast 1D scanlines
    y_lines = (0.70 + 0.30 * np.sin(np.arange(h) * 0.95)).astype(np.float32)
    y_lines[np.arange(h) % 3 == 0] *= 0.55
    holo_rgb *= y_lines[:, None, None]

    # 4. Fast edge glow via morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, kernel).astype(np.float32) / 255.0
    holo_rgb[:, :, 0] += edge * 255.0 * 0.8
    holo_rgb[:, :, 1] += edge * 235.0 * 0.8
    holo_rgb[:, :, 2] += edge * 40.0 * 0.8

    out_rgb = np.clip(holo_rgb, 0, 255).astype(np.uint8)
    out_alpha = np.clip(alpha.astype(np.float32) * 0.92 + edge * 200.0, 0, 255).astype(np.uint8)
    return np.dstack([out_rgb, out_alpha])

def effect_heatmap(rgba):
    """Heatmap: Thermal infrared / FLIR / Turbo colormap showing combat energy."""
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_TURBO)
    hot_mask = (enhanced > 190)[:, :, None].astype(np.float32)
    thermal = np.clip(thermal.astype(np.float32) * (1.0 - hot_mask * 0.2) + np.array([255, 255, 255]) * hot_mask * 0.5, 0, 255).astype(np.uint8)

    return np.dstack([thermal, alpha])

def effect_lidar(rgba):
    """LiDAR: Discrete point-cloud scanner with laser elevation gradient and scan dots."""
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    out_canvas = np.zeros((h, w, 4), dtype=np.uint8)
    step = 7 # optimal density for genuine LiDAR scanner look & high speed

    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()

    mask_vals = alpha[y_flat, x_flat]
    valid = mask_vals > 50

    vx = x_flat[valid]
    vy = y_flat[valid]

    norm_y = 1.0 - (vy / float(h))
    
    # Fast vectorized color LUT
    colors = np.zeros((len(vy), 3), dtype=np.uint8)
    is_lower = norm_y < 0.5
    t_low = norm_y[is_lower] * 2.0
    colors[is_lower, 0] = (1.0 - t_low) * 40 + t_low * 255 # B
    colors[is_lower, 1] = (1.0 - t_low) * 255 + t_low * 230 # G
    colors[is_lower, 2] = (1.0 - t_low) * 60 + t_low * 0    # R

    t_high = (norm_y[~is_lower] - 0.5) * 2.0
    colors[~is_lower, 0] = (1.0 - t_high) * 255 + t_high * 0
    colors[~is_lower, 1] = (1.0 - t_high) * 230 + t_high * 220
    colors[~is_lower, 2] = (1.0 - t_high) * 0 + t_high * 255

    # Draw laser dots
    for i in range(len(vx)):
        cx, cy = int(vx[i]), int(vy[i])
        b, g, r = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        cv2.circle(out_canvas, (cx, cy), 2, (b, g, r, 255), -1, cv2.LINE_4)

    # Laser scanlines
    sweep_y = np.arange(0, h, 14)
    for sy in sweep_y:
        row_alpha = alpha[sy, :]
        active_x = np.where(row_alpha > 50)[0]
        if len(active_x) > 0:
            cv2.line(out_canvas, (int(active_x[0]), sy), (int(active_x[-1]), sy), (220, 255, 120, 100), 1, cv2.LINE_4)

    return out_canvas

def effect_gaussian_splat(rgba):
    """Gaussian Splat: 3D Gaussian Splatting representation with oriented elliptical radiance splats."""
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]

    out_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    alpha_mask = np.zeros((h, w), dtype=np.uint8)
    step = 10 # distinct, visible Gaussian splats
    np.random.seed(42)

    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()

    mask_vals = alpha[y_flat, x_flat]
    valid = mask_vals > 50

    vx = x_flat[valid]
    vy = y_flat[valid]
    sampled_bgr = rgb[vy, vx]

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.degrees(np.arctan2(gy[vy, vx], gx[vy, vx])) + 90.0

    jit_x = np.clip(vx + np.random.randint(-2, 3, len(vx)), 0, w - 1)
    jit_y = np.clip(vy + np.random.randint(-2, 3, len(vy)), 0, h - 1)

    # Draw Gaussian splat ellipses
    for i in range(len(vx)):
        cx, cy = int(jit_x[i]), int(jit_y[i])
        b, g, r = int(sampled_bgr[i, 0]), int(sampled_bgr[i, 1]), int(sampled_bgr[i, 2])

        ax1 = int(np.random.randint(9, 14))
        ax2 = int(np.random.randint(4, 8))
        ang = float(angles[i])

        # Outer soft envelope
        cv2.ellipse(out_canvas, (cx, cy), (ax1, ax2), ang, 0, 360, (b, g, r), -1, cv2.LINE_AA)
        cv2.ellipse(alpha_mask, (cx, cy), (ax1, ax2), ang, 0, 360, 220, -1, cv2.LINE_AA)

        # Concentrated core
        b_c = min(255, int(b * 1.35))
        g_c = min(255, int(g * 1.35))
        r_c = min(255, int(r * 1.35))
        cv2.ellipse(out_canvas, (cx, cy), (max(2, ax1 // 2), max(1, ax2 // 2)), ang, 0, 360, (b_c, g_c, r_c), -1, cv2.LINE_4)

    # Add floating satellite splats
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    rim_zone = cv2.subtract(cv2.dilate(alpha, kernel), alpha)
    rim_y, rim_x = np.where((rim_zone > 50) & (np.random.uniform(0, 1, (h, w)) > 0.992))
    for ry, rx in zip(rim_y, rim_x):
        b = int(np.random.randint(180, 255))
        g = int(np.random.randint(150, 230))
        r = int(np.random.randint(60, 200))
        ax1 = int(np.random.randint(5, 10))
        ax2 = int(np.random.randint(2, 6))
        ang = float(np.random.randint(0, 180))
        cv2.ellipse(out_canvas, (int(rx), int(ry)), (ax1, ax2), ang, 0, 360, (b, g, r), -1, cv2.LINE_4)
        cv2.ellipse(alpha_mask, (int(rx), int(ry)), (ax1, ax2), ang, 0, 360, 180, -1, cv2.LINE_4)

    splat_rgb = cv2.GaussianBlur(out_canvas, (5, 5), 1.5)
    splat_a = cv2.GaussianBlur(alpha_mask, (5, 5), 1.5)
    return np.dstack([splat_rgb, splat_a])

def effect_blob(rgba):
    """Blob: Organic fluid metaball / SDF plasma mass with molten energy core."""
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    # Downscale for instant morphological operations
    small_alpha = cv2.resize(alpha, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(small_alpha, kernel, iterations=2)
    blob_mask = cv2.GaussianBlur(dilated, (15, 15), 5.0)
    
    _, thresh = cv2.threshold(blob_mask, 50, 255, cv2.THRESH_BINARY)
    smooth_blob_s = cv2.GaussianBlur(thresh, (9, 9), 3.0)

    # Upscale back to full resolution
    smooth_blob = cv2.resize(smooth_blob_s, (w, h), interpolation=cv2.INTER_LINEAR)

    # Signed Distance Field via Distance Transform
    dist = cv2.distanceTransform((smooth_blob > 40).astype(np.uint8), cv2.DIST_L2, 3)
    max_d = np.max(dist) if np.max(dist) > 0 else 1.0
    norm_dist = np.clip(dist / (max_d * 0.8), 0.0, 1.0)

    # Vectorized plasma color map
    c_edge = np.array([160.0, 30.0, 90.0], dtype=np.float32) # Deep violet
    c_mid = np.array([210.0, 40.0, 220.0], dtype=np.float32)  # Magenta
    c_core = np.array([255.0, 230.0, 200.0], dtype=np.float32) # Cyan-white

    t = norm_dist[:, :, None]
    color_map = np.where(t < 0.5,
                         (1.0 - t * 2.0) * c_edge + (t * 2.0) * c_mid,
                         (1.0 - (t - 0.5) * 2.0) * c_mid + ((t - 0.5) * 2.0) * c_core)

    # Equipotential contour rings
    rings = (np.sin(dist * 0.45) > 0.85).astype(np.float32)[:, :, None]
    color_map = np.clip(color_map + rings * 100.0, 0, 255)

    blob_alpha = (smooth_blob.astype(np.float32) / 255.0 * 235.0).astype(np.uint8)
    return np.dstack([color_map.astype(np.uint8), blob_alpha])

def main():
    p = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\data_dp\frames_combined\frame_0150.png"
    src = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if src is None:
        print("Failed to load source frame")
        return

    print("Generating effects on frame 150...")
    f1 = effect_original(src)
    f2 = effect_holographic(src)
    f3 = effect_heatmap(src)
    f4 = effect_lidar(src)
    f5 = effect_gaussian_splat(src)
    f6 = effect_blob(src)

    # Composite each onto a dark background to inspect
    bg_color = np.array([8, 6, 5], dtype=np.uint8)
    def to_bgr(f):
        c = np.tile(bg_color, (f.shape[0], f.shape[1], 1))
        a = (f[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        return (f[:, :, :3].astype(np.float32) * a + c.astype(np.float32) * (1.0 - a)).astype(np.uint8)

    labels = ["1. ORIGINAL", "2. HOLOGRAPHIC", "3. HEATMAP", "4. LIDAR", "5. GAUSSIAN SPLAT", "6. BLOB"]
    rendered = [to_bgr(f) for f in [f1, f2, f3, f4, f5, f6]]

    for i, (img, lbl) in enumerate(zip(rendered, labels)):
        cv2.putText(img, lbl, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 240, 255), 2, cv2.LINE_AA)

    # 2 rows x 3 cols preview grid
    row1 = np.hstack([rendered[0], rendered[1], rendered[2]])
    row2 = np.hstack([rendered[3], rendered[4], rendered[5]])
    preview = np.vstack([row1, row2])

    out_p = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time\samples_new\effects_comparison.png"
    cv2.imwrite(out_p, preview)
    print("Saved preview to:", out_p)

if __name__ == "__main__":
    main()
