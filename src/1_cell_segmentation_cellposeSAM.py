import os, gc, json, time, psutil, torch
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.filters import threshold_otsu, sobel
from skimage.morphology import remove_small_objects
from cellpose import models

# =======================================================
# CONFIG
# =======================================================
BASE_DIR = Path("splitted_images")
OUTPUT_DIR = Path("data/features_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_MARKERS = ["TMEM119", "CD68", "CD45", "GPNMB"]

MAX_INTENSITY_WORKERS = min(8, os.cpu_count() or 4)
INT_TILE_SIZE = 6000
CP_TILE_OVERLAP = 0.2

# =======================================================
# LOGGING
# =======================================================
def log(msg, level="INFO"):
    cpu = psutil.cpu_percent(None)
    mem = psutil.virtual_memory().percent
    print(
        f"[{time.strftime('%H:%M:%S')}] "
        f"[{level}] {msg} | CPU {cpu:.1f}% MEM {mem:.1f}%",
        flush=True
    )

# =======================================================
# IMAGE SCORING (for best segmentation marker)
# =======================================================
def score_image(img):
    img = img.astype(np.float32)
    img -= img.min()
    mx = img.max()
    if mx > 0:
        img /= mx

    thr = threshold_otsu(img)
    fg = remove_small_objects(img > thr, 32)

    contrast = np.percentile(img, 99) - np.percentile(img, 1)
    sharpness = np.median(sobel(img))
    fg_frac = float(fg.mean())

    ok = 1.0 if (0.05 < fg_frac < 0.6) else 0.0
    return (contrast * 0.6 + sharpness * 0.4) * ok

def pick_best_cell_marker(region_dir):
    log("Selecting best cell marker for segmentation", "STEP")
    best_score, best_path = -1, None

    for m in CELL_MARKERS:
        f = region_dir / f"{m}.tif"
        if not f.exists():
            continue
        try:
            img = tiff.imread(f)
            if img.ndim == 3:
                img = img.max(axis=0)
            s = score_image(img)
            log(f"{m}: score={s:.3f}", "SUB")
            if s > best_score:
                best_score, best_path = s, f
        except Exception as e:
            log(f"Scoring failed {m}: {e}", "WARN")

    if best_path is not None:
        log(f"Best segmentation marker: {best_path.stem}", "DONE")
    return best_path

# =======================================================
# CELLPOSE (CELL ONLY)
# =======================================================
def run_cellpose_cells(cell_img):
    log(f"Cellpose CELL input shape {cell_img.shape}", "SUB")
    model = models.Cellpose(model_type="cyto", gpu=torch.cuda.is_available())
    t0 = time.time()

    try:
        masks, *_ = model.eval(
            cell_img.astype(np.float32),
            channels=[0, 0],
            diameter=40,
            normalize=True,
            tile=True,
            tile_overlap=CP_TILE_OVERLAP
        )
    except TypeError:
        masks, *_ = model.eval(
            cell_img.astype(np.float32),
            channels=[0, 0],
            diameter=40,
            normalize=True
        )

    log(f"Cellpose CELL done in {time.time()-t0:.1f}s", "DONE")
    del model
    torch.cuda.empty_cache()
    return masks.astype(np.uint32)

# =======================================================
# GEOMETRY
# =======================================================
def geometry_from_mask(mask):
    m = mask.astype(np.int64, copy=False)
    max_id = int(m.max())
    if max_id == 0:
        return pd.DataFrame(columns=["Object ID", "area", "centroid-0", "centroid-1"])

    area = np.bincount(m.ravel(), minlength=max_id + 1)
    ys, xs = np.indices(m.shape)
    sum_y = np.bincount(m.ravel(), weights=ys.ravel(), minlength=max_id + 1)
    sum_x = np.bincount(m.ravel(), weights=xs.ravel(), minlength=max_id + 1)

    labels = np.arange(1, max_id + 1)
    valid = area[labels] > 0

    return pd.DataFrame({
        "Object ID": labels[valid],
        "area": area[labels][valid],
        "centroid-0": sum_y[labels][valid] / area[labels][valid],
        "centroid-1": sum_x[labels][valid] / area[labels][valid],
    })

# =======================================================
# INTENSITY
# =======================================================
def fast_intensity(mask, img, tile_size=INT_TILE_SIZE, median_cap=2000):
    sums = defaultdict(float)
    counts = defaultdict(int)
    samples = defaultdict(list)
    rng = np.random.default_rng(0)

    h, w = mask.shape
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            m = mask[y:y+tile_size, x:x+tile_size]
            if m.max() == 0:
                continue
            i = img[y:y+tile_size, x:x+tile_size]

            for lbl in np.unique(m[m > 0]):
                pix = i[m == lbl]
                sums[lbl] += pix.sum()
                counts[lbl] += pix.size

                s = samples[lbl]
                remain = median_cap - len(s)
                if remain > 0:
                    take = pix if pix.size <= remain else rng.choice(pix, remain, False)
                    s.extend(take.tolist())

    return pd.DataFrame({
        "LabelID": list(counts.keys()),
        "MeanIntensity": [sums[k]/counts[k] for k in counts],
        "MedianIntensity": [np.median(samples[k]) for k in counts]
    })

def safe_tiff_read(path, dtype=None):
    try:
        arr = tiff.memmap(path)
    except Exception:
        arr = tiff.imread(path)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr

def compute_marker(marker_path, mask_path, cache_dir):
    log(f"Intensity {marker_path.name}", "SUB")
    cache = cache_dir / f"{marker_path.stem}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    img = safe_tiff_read(marker_path, np.float32)
    mask = safe_tiff_read(mask_path)

    df = fast_intensity(mask, img)

    result = {
        "marker": marker_path.stem,
        "df": df.to_dict(orient="list")
    }

    with open(cache, "w") as f:
        json.dump(result, f)

    del img, mask, df
    gc.collect()
    return result

# =======================================================
# MAIN
# =======================================================
if __name__ == "__main__":

    for biopsy_dir in BASE_DIR.glob("biopsy_*"):
        for region_dir in biopsy_dir.iterdir():
            if not region_dir.is_dir():
                continue

            sample_id = f"{biopsy_dir.name}_{region_dir.name}"
            log(f"=== START {sample_id} ===", "STEP")

            out_csv = OUTPUT_DIR / f"{sample_id}.csv"
            if out_csv.exists():
                log("Already processed → skip", "WARN")
                continue

            cache_dir = OUTPUT_DIR / f"{sample_id}_cache"
            cache_dir.mkdir(exist_ok=True)

            cell_mask_p = OUTPUT_DIR / f"{sample_id}_cell_mask.tif"

            seg_marker = pick_best_cell_marker(region_dir)
            if seg_marker is None:
                continue

            cell_img = tiff.imread(seg_marker)
            if cell_img.ndim == 3:
                cell_img = cell_img.max(axis=0)

            if cell_mask_p.exists():
                cell_masks = tiff.imread(cell_mask_p)
            else:
                cell_masks = run_cellpose_cells(cell_img)
                tiff.imwrite(cell_mask_p, cell_masks)

            cell_df = geometry_from_mask(cell_masks)
            cell_df["ROI"] = region_dir.name
            cell_df["Image"] = sample_id

            tasks = []
            for f in region_dir.glob("*.tif"):
                if f.stem in CELL_MARKERS:
                    tasks.append((f, cell_mask_p, cache_dir))

            results = []
            with ProcessPoolExecutor(MAX_INTENSITY_WORKERS) as ex:
                for fut in as_completed([ex.submit(compute_marker, *t) for t in tasks]):
                    results.append(fut.result())

            df = cell_df.copy()
            for r in results:
                tmp = pd.DataFrame(r["df"])
                for stat in ["MeanIntensity", "MedianIntensity"]:
                    col = f"{r['marker']}: {stat.replace('Intensity','')}"
                    df[col] = df["Object ID"].map(
                        pd.Series(tmp[stat].values, index=tmp["LabelID"])
                    )

            df.to_csv(out_csv, index=False)
            log(f"Saved {out_csv}", "DONE")
            gc.collect()

            log(f"=== END {sample_id} ===", "DONE")

    log("ALL BIOPSIES DONE", "DONE")
