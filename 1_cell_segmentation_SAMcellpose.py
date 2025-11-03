import os
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from cellpose import models
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import remove_small_objects
import torch, gc, multiprocessing, time, psutil, json
from collections import defaultdict

# =======================================================
# CONFIGURATION
# =======================================================
BASE_DIR = Path("data/patients_split")
OUTPUT_DIR = Path("data/features_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NUCLEUS_MARKERS = ["DAPI", "H3K27M", "Ki67"]
NUCLEUS_MARKERS = ["DAPI", "H3K27M"]
# CELL_MARKERS = [
#     "GLUT1", "GFAP", "CD163", "SOX4", "CD31", "TMEM119", "HLA-A",
#     "CD68", "CD20", "CD4", "CD8", "GPNMB", "PD-L1", "CD45RO", "SMA",
#     "CD45", "CD3e", "HLA-DR", "PDGFRA", "NESTIN", "LGALS3", "PD-1",
#     "MOG", "SPP1", "ALDH1L1", "FOXP3"
# ]

CELL_MARKERS = [
    "GLUT1", "TMEM119", "CD68", "CD45", "LGALS3", "SPP1", "CD31"
]

# =======================================================
# LOGGING
# =======================================================
def log(msg):
    cpu, mem = psutil.cpu_percent(None), psutil.virtual_memory().percent
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | CPU {cpu:.1f}% MEM {mem:.1f}%")

# =======================================================
# IMAGE QUALITY SCORING
# =======================================================
def score_image(img):
    img = img.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img /= img.max()

    thr = threshold_otsu(img)
    fg = img > thr
    fg = remove_small_objects(fg, 16)

    contrast = np.percentile(img, 99) - np.percentile(img, 1)
    sharpness = np.median(sobel(img))
    fg_frac = fg.mean()

    score = (contrast * 0.6 + sharpness * 0.4) * (0.1 < fg_frac < 0.5)
    return score

def pick_best_marker(patient_dir, marker_list):
    best_score, best_path = -1, None
    for marker in marker_list:
        for f in patient_dir.glob(f"*{marker}*.tif"):
            try:
                img = tiff.imread(f)
                if img.ndim == 3:
                    img = img.max(axis=0)
                s = score_image(img)
                if s > best_score:
                    best_score, best_path = s, f
            except Exception as e:
                log(f"Error scoring {f}: {e}")
    if best_path:
        log(f"Best channel selected: {best_path.name} (score={best_score:.3f})")
    return best_path

# =======================================================
# CELLPOSE MODELS
# =======================================================
def run_cellpose_cyto3_dual(cyto_img, nuc_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Running Cellpose dual-channel model on {device}")
    model = models.CellposeModel(pretrained_model="cpsam", device=device, gpu=(device.type=="cuda"))
    img = np.stack([cyto_img, nuc_img], axis=-1).astype(np.float32)
    masks, flows, styles = model.eval(img, diameter=None, batch_size=8, normalize=True)
    cell_mask = masks.astype(np.uint32)
    log(f"✅ Cell segmentation done → {cell_mask.max()} cells")
    return cell_mask

def run_cellpose_nuclei(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"🧠 Running Cellpose nuclei model on {device}")
    model = models.CellposeModel(pretrained_model="cpsam", device=device, gpu=(device.type=="cuda"))
    masks, flows, styles = model.eval(img, diameter=None, batch_size=8, normalize=True)
    log(f"✅ Nucleus segmentation done → {masks.max()} nuclei")
    return masks.astype(np.uint32)

# =======================================================
# INTENSITY CALCULATION
# =======================================================
def fast_intensity_with_median(mask, img, tile_size=8192):
    h, w = mask.shape
    n_labels = mask.max()
    sums = np.zeros(n_labels + 1, dtype=np.float64)
    counts = np.zeros(n_labels + 1, dtype=np.int64)
    pixel_values = defaultdict(list)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            m_tile = mask[y:y + tile_size, x:x + tile_size]
            i_tile = img[y:y + tile_size, x:x + tile_size]
            valid = m_tile > 0
            if not np.any(valid): continue
            flat_mask = m_tile[valid].ravel()
            flat_img = i_tile[valid].ravel()
            np.add.at(sums, flat_mask, flat_img)
            np.add.at(counts, flat_mask, 1)
            for lbl in np.unique(flat_mask):
                vals = flat_img[flat_mask == lbl]
                if len(vals) > 5000:
                    vals = np.random.choice(vals, 5000, replace=False)
                pixel_values[lbl].extend(vals.tolist())
        gc.collect()

    means = np.zeros_like(sums, dtype=np.float32)
    medians = np.zeros_like(sums, dtype=np.float32)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    for lbl, vals in pixel_values.items():
        if len(vals) > 0:
            medians[lbl] = np.median(vals)

    df = pd.DataFrame({
        "LabelID": np.arange(1, len(means)),
        "MeanIntensity": means[1:],
        "MedianIntensity": medians[1:]
    })
    return df

def compute_marker_intensities_light(marker_file, mask, cache_dir):
    name = marker_file.stem.strip()
    cache_file = cache_dir / f"{name}_intensity.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    start = time.time()
    log(f"Computing intensities for {name} ...")
    img = tiff.imread(marker_file).astype(np.float32)
    mask_df = fast_intensity_with_median(mask, img, tile_size=12000)

    all_labels = np.unique(mask)
    all_labels = all_labels[all_labels > 0]
    mask_df = mask_df.set_index("LabelID").reindex(all_labels, fill_value=0).reset_index()

    result = {"marker": name, "cell_df": mask_df.to_dict(orient="list")}
    with open(cache_file, "w") as f:
        json.dump(result, f)
    log(f"✅ Done {name} in {time.time()-start:.1f}s")
    return result

# =======================================================
# MAIN PIPELINE
# =======================================================
for patient_dir in BASE_DIR.glob("patient_*"):
    log(f"🚀 Start {patient_dir.name}")
    out_csv = OUTPUT_DIR / f"{patient_dir.name}_measurements.csv"
    cache_dir = OUTPUT_DIR / f"{patient_dir.name}_cache"
    cache_dir.mkdir(exist_ok=True)

    if out_csv.exists():
        log(f"Already processed → skipping {patient_dir.name}")
        continue

    nuc_path = pick_best_marker(patient_dir, NUCLEUS_MARKERS)
    cyto_path = pick_best_marker(patient_dir, CELL_MARKERS)

    if nuc_path is None or cyto_path is None:
        log("❌ Missing nucleus or cyto marker → skip patient")
        continue

    nuc_img = tiff.imread(nuc_path)
    cyto_img = tiff.imread(cyto_path)

    # 🔹 Segmentation (dual channel)
    cell_mask_p = OUTPUT_DIR / f"{patient_dir.name}_cell_mask.tif"
    if cell_mask_p.exists():
        log("Loading cached cell mask")
        cell_masks = tiff.imread(cell_mask_p)
    else:
        cell_masks = run_cellpose_cyto3_dual(cyto_img, nuc_img)
        tiff.imwrite(cell_mask_p, cell_masks)

    # 🔹 Nucleus segmentation (global once)
    nuc_mask_p = OUTPUT_DIR / f"{patient_dir.name}_nucleus_mask.tif"
    if nuc_mask_p.exists():
        log("Loading cached nucleus mask")
        nuc_mask_global = tiff.imread(nuc_mask_p)
    else:
        nuc_mask_global = run_cellpose_nuclei(nuc_img)
        tiff.imwrite(nuc_mask_p, nuc_mask_global)

    # 🔹 Cell features
    def safe_regionprops(mask, img=None, tile_size=4096):
        h, w = mask.shape
        props_list = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                m_tile = mask[y:y+tile_size, x:x+tile_size]
                if np.all(m_tile == 0): continue
                i_tile = img[y:y+tile_size, x:x+tile_size] if img is not None else None
                props = regionprops_table(m_tile, intensity_image=i_tile,
                                          properties=("label","area","centroid","solidity","perimeter"))
                props_list.append(pd.DataFrame(props))
            gc.collect()
        df = pd.concat(props_list, ignore_index=True)
        df = df.groupby("label", as_index=False).first()
        return df

    cell_df = safe_regionprops(cell_masks, cyto_img)
    cell_df.rename(columns={"label": "CellID"}, inplace=True)

    # =======================================================
    # MARKER INTENSITY CALCULATION
    # =======================================================
    results = []
    for marker_file in patient_dir.glob("*.tif"):
        marker_name = marker_file.stem.strip()

        if marker_name in NUCLEUS_MARKERS:
            mask_to_use = nuc_mask_global
            mask_type = "Nucleus"
        elif marker_name in CELL_MARKERS:
            mask_to_use = cell_masks
            mask_type = "Cell"
        else:
            log(f"⚠️ Marker {marker_name} not in lists → skipping")
            continue

        result = compute_marker_intensities_light(marker_file, mask_to_use, cache_dir)
        result["mask_type"] = mask_type
        results.append(result)

    # 🔹 Merge results
    log("Building final table...")
    df = cell_df.copy()
    df.rename(columns={
        "CellID": "Object ID",
        "centroid-1": "Centroid X µm",
        "centroid-0": "Centroid Y µm",
        "area": "Cell: Area µm^2",
        "perimeter": "Cell: Length µm",
        "solidity": "Cell: Solidity",
    }, inplace=True)
    df["Cell: Circularity"] = 4*np.pi*df["Cell: Area µm^2"]/(df["Cell: Length µm"]**2+1e-9)
    df["Image"] = f"{patient_dir.name}.qptiff"
    df["Object type"] = "Cell"
    df["ROI"] = "ROI-1"

    for r in results:
        m = r["marker"]
        mask_type = r.get("mask_type", "Cell")
        tmp = pd.DataFrame(r["cell_df"])
        mean_series = pd.Series(tmp["MeanIntensity"].values, index=tmp["LabelID"])
        median_series = pd.Series(tmp["MedianIntensity"].values, index=tmp["LabelID"])
        df[f"{mask_type}: {m}: Mean"] = df["Object ID"].map(mean_series)
        df[f"{mask_type}: {m}: Median"] = df["Object ID"].map(median_series)

    cols_first = [
        "Image","Object ID","Object type","ROI",
        "Centroid X µm","Centroid Y µm",
        "Cell: Area µm^2","Cell: Length µm",
        "Cell: Circularity","Cell: Solidity"
    ]
    df = df[[c for c in cols_first if c in df.columns] +
            [c for c in df.columns if c not in cols_first]]

    df.to_csv(out_csv, index=False)
    log(f"💾 Saved {out_csv} ({df.shape[0]}×{df.shape[1]})")

    del df, cell_df, cell_masks, nuc_mask_global
    gc.collect()

log("✅ All patients processed successfully with auto-channel selection.")
