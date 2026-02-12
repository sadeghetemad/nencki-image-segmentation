import json
import tifffile as tiff
import numpy as np
import cv2
from pathlib import Path

# ---------------- config ----------------
BASE_DIR = Path().resolve().parent
IMAGE_DIR = BASE_DIR / "image"
SERVER_DIR = IMAGE_DIR / "server"
OUT_DIR = BASE_DIR / "splitted_images"

THUMB_SCALE = 0.03

# markers
NUCLEUS_MARKERS = []
CELL_MARKERS = ["TMEM119", "CD68", "CD45", "GPNMB"]
SELECTED_MARKERS = set(NUCLEUS_MARKERS + CELL_MARKERS)

# slide list
SLIDES = {
    "2506": "bottom",
    "2503": "upper",
    "4155": "bottom",
    "4152": "upper",
    "4891": "upper",
    "4892": "bottom",
    "5029": "full",
    "5727": "upper",
    "5731": "bottom",
    "5730": "upper",
    "5733": "bottom",
    "4156": "upper",
    "17827": "bottom",
    "5113": "bottom",
    "1287": "upper",
    "732": "bottom",
    "728": "upper",
    "930": "bottom",
    "926": "upper",
    "1684": "upper",
    "1686": "bottom",
    "2087": "upper",
    "2089": "bottom",
    "1991": "bottom",
    "3149": "upper",
}

# manual split overrides (ratio of height)
MANUAL_SPLITS = {
    "1287": 0.2,   
}


(OUT_DIR / "thumbnails").mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def load_channels_for_biopsy(biopsy_id: str):
    servers = list(SERVER_DIR.glob(f"*{biopsy_id}*.json"))
    if not servers:
        raise FileNotFoundError(f"No server json found for biopsy {biopsy_id}")

    with open(servers[0]) as f:
        meta = json.load(f)

    return {
        i: c["name"]
        for i, c in enumerate(meta["metadata"]["channels"])
    }

def tissue_aware_split(img, channel_indices, region, biopsy_id):
    """
    img: (H, W, C)
    region: upper | bottom | full
    """

    if region == "full":
        return img

    h = img.shape[0]

    # --- manual override ---
    if biopsy_id in MANUAL_SPLITS:
        ratio = MANUAL_SPLITS[biopsy_id]
        split_y = int(h * ratio)

        if region == "upper":
            return img[:split_y]
        else:
            return img[split_y:]

    # --- automatic tissue-aware split ---
    signal = np.sum(img[..., channel_indices], axis=-1)
    thresh = np.percentile(signal, 95)
    mask = signal > thresh

    if not mask.any():
        mid = h // 2
        return img[:mid] if region == "upper" else img[mid:]

    ys = np.where(mask.any(axis=1))[0]
    y_min, y_max = ys.min(), ys.max()
    tissue_mid = (y_min + y_max) // 2

    if region == "upper":
        return img[y_min:tissue_mid]
    else:
        return img[tissue_mid:y_max]


# ---------------- main loop ----------------
for biopsy, region in SLIDES.items():
    print(f"\nProcessing biopsy {biopsy} ({region})")

    try:
        channels = load_channels_for_biopsy(biopsy)
    except Exception as e:
        print(f" server load failed → {e}")
        continue

    out_base = OUT_DIR / f"biopsy_{biopsy}" / region
    if out_base.exists() and any(out_base.glob("*.tif")):
        print(" already processed → skip")
        continue

    files = list(IMAGE_DIR.glob(f"*{biopsy}*.qptiff"))
    if not files:
        print(" image file not found → skip")
        continue

    img = tiff.imread(files[0])

    # ensure (H, W, C)
    if img.shape[0] < img.shape[-1]:
        img = img.transpose(1, 2, 0)
    
    # hard fix for slide 5113: discard top 20% and skip smart splitting
    if biopsy == "5113":
        h = img.shape[0]
        img = img[int(0.2 * h):]
        region = "full" 


    # selected channel indices
    selected_channel_indices = [
        i for i, ch in channels.items()
        if ch in SELECTED_MARKERS and i < img.shape[-1]
    ]

    if not selected_channel_indices:
        print(" no selected markers found → skip")
        continue

    crop = tissue_aware_split(
        img,
        selected_channel_indices,
        region,
        biopsy
        )

    out_base.mkdir(parents=True, exist_ok=True)
    thumb_base = OUT_DIR / "thumbnails" / f"biopsy_{biopsy}" / region
    thumb_base.mkdir(parents=True, exist_ok=True)

    for i, ch in channels.items():
        if ch not in SELECTED_MARKERS:
            continue
        if i >= crop.shape[-1]:
            continue

        name = ch.replace("/", "_")

        # save full-resolution channel
        tiff.imwrite(
            out_base / f"{name}.tif",
            crop[..., i].astype(np.float16),
            compression="zlib"
        )

        # save thumbnail
        thumb = cv2.resize(
            crop[..., i],
            (0, 0),
            fx=THUMB_SCALE,
            fy=THUMB_SCALE
        )
        thumb = cv2.normalize(thumb, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(thumb_base / f"{name}.jpg"), thumb)

    print(" saved")

print("\nAll done.")
