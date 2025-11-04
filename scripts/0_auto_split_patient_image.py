import os
import json
import tifffile as tiff
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
INPUT_PATH = "image/pHGG_1991_3149_Scan1-001.qptiff"   # your Akoya QPTIFF file
SERVER_JSON = "image/server.json"                      # JSON metadata file
OUTPUT_DIR = Path("data/patients_split")

THRESH_VALUE = 30       # intensity threshold for black gap
MIN_GAP_HEIGHT = 500    # min height (pixels) for dark region (gap)
THUMBNAIL_SCALE = 0.03  # scale factor for small preview images

# -------------------------------------------------------
# Prepare output folders
# -------------------------------------------------------
for sub in ["", "thumbnails"]:
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# 1️⃣ Load channel names from server.json
# -------------------------------------------------------
print(f"📄 Reading channel metadata from: {SERVER_JSON}")
with open(SERVER_JSON, "r") as f:
    meta = json.load(f)

channels_meta = meta["metadata"]["channels"]
channel_map = {i: ch["name"] for i, ch in enumerate(channels_meta)}

print(f"✅ Loaded {len(channel_map)} channels:")
for i, n in channel_map.items():
    print(f"  {i}: {n}")

# -------------------------------------------------------
# 2️⃣ Load QPTIFF image
# -------------------------------------------------------
print(f"\n🧠 Loading QPTIFF image: {INPUT_PATH}")
img = tiff.imread(INPUT_PATH)

# Convert (C, H, W) → (H, W, C)
if img.ndim == 3 and img.shape[0] < img.shape[-1]:
    img = np.transpose(img, (1, 2, 0))

H, W, C = img.shape
print(f"Image shape: {H} x {W} x {C}")

# -------------------------------------------------------
# 3️⃣ Detect black gap between two patient regions
# -------------------------------------------------------
gray = img[..., 0].astype(np.float32)  # use first channel for intensity
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, mask = cv2.threshold(gray_norm, THRESH_VALUE, 255, cv2.THRESH_BINARY)
profile = np.mean(mask, axis=1)
gap_rows = profile < 5  # rows that are mostly black

start, end, max_len = None, None, 0
curr_start = None
for i, is_black in enumerate(gap_rows):
    if is_black and curr_start is None:
        curr_start = i
    elif not is_black and curr_start is not None:
        length = i - curr_start
        if length > max_len and length > MIN_GAP_HEIGHT:
            max_len = length
            start, end = curr_start, i
        curr_start = None

if start is None:
    raise RuntimeError("❌ No clear dark gap found between patient regions!")

split_row = (start + end) // 2
print(f"🩸 Detected dark gap between rows {start}–{end}. Splitting at {split_row}")

# -------------------------------------------------------
# 4️⃣ Split top and bottom regions (patients)
# -------------------------------------------------------
patient_info = {
    "patient_3149": img[:split_row, :, :],
    "patient_1991": img[split_row:, :, :]
}

# -------------------------------------------------------
# 5️⃣ Save each patient's channels separately
# -------------------------------------------------------
for patient, crop_img in patient_info.items():
    patient_dir = OUTPUT_DIR / patient
    patient_thumb_dir = OUTPUT_DIR / "thumbnails" / patient
    patient_dir.mkdir(parents=True, exist_ok=True)
    patient_thumb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🧬 Saving channels for {patient} ...")

    for i, marker in channel_map.items():
        if i >= crop_img.shape[-1]:
            print(f"⚠️ Skipping channel {marker}: index {i} beyond image depth")
            continue
        out_path = patient_dir / f"{marker.replace('/', '_')}.tif"
        tiff.imwrite(out_path, crop_img[..., i].astype(np.float32))

        # Create thumbnail
        thumb = cv2.resize(crop_img[..., i], (0, 0), fx=THUMBNAIL_SCALE, fy=THUMBNAIL_SCALE)
        thumb_path = patient_thumb_dir / f"{marker.replace('/', '_')}_thumb.jpg"
        cv2.imwrite(str(thumb_path), thumb)

    print(f"✅ Channels & thumbnails saved for {patient}")

print("\n🎉 Done! All channels saved per patient successfully.")
