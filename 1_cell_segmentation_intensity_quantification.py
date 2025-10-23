import os
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from cellpose import models
from skimage.measure import regionprops_table
from scipy.spatial import cKDTree
from tqdm import tqdm
from joblib import Parallel, delayed
import torch, gc, multiprocessing, time, psutil, json

# =======================================================
# CONFIGURATION
# =======================================================
BASE_DIR = Path("data/patients_split")
OUTPUT_DIR = Path("data/features_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUCLEUS_MARKERS = ["DAPI", "H3K27M", "Ki67"]
CELL_MARKERS = [
    "GLUT1", "GFAP", "CD163", "SOX4", "CD31", "TMEM119", "HLA-A",
    "CD68", "CD20", "CD4", "CD8", "GPNMB", "PD-L1", "CD45RO", "SMA",
    "CD45", "CD3e", "HLA-DR", "PDGFRA", "NESTIN", "LGALS3", "PD-1",
    "MOG", "SPP1", "ALDH1L1", "FOXP3", "CD68"
]

N_CORES = max(1, int(multiprocessing.cpu_count() * 0.75))
CHUNK_SIZE = 3  # how many markers to process per batch in parallel
print(f"🧠 Detected {multiprocessing.cpu_count()} cores → using {N_CORES} (chunk size = {CHUNK_SIZE})\n")

# =======================================================
# LOGGING UTIL
# =======================================================
def log(msg):
    cpu, mem = psutil.cpu_percent(None), psutil.virtual_memory().percent
    print(f"[{time.strftime('%H:%M:%S')}] 🧩 {msg} | CPU {cpu:.1f}% MEM {mem:.1f}%")

# =======================================================
# SEGMENTATION
# =======================================================
def run_cellpose_segmentation(img, model_type="nuclei"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if img.ndim == 3:
        img = img[..., 0]
    log(f"Running Cellpose ({model_type}) on {device}")
    model = models.CellposeModel(gpu=(device.type == "cuda"), device=device, model_type=model_type)
    masks, _, _ = model.eval(img, diameter=None, channels=None, batch_size=1, tile=True, tile_overlap=0.1)
    log(f"Done {model_type} → {masks.max()} objects")
    return masks.astype(np.uint16)

# =======================================================
# MATCH NUCLEUS ↔ CELL
# =======================================================
def match_nucleus_to_cell(nuc_df, cell_df):
    nuc_xy = np.c_[nuc_df["centroid-0"], nuc_df["centroid-1"]]
    cell_xy = np.c_[cell_df["centroid-0"], cell_df["centroid-1"]]
    idx = cKDTree(cell_xy).query(nuc_xy, k=1)[1]
    return {int(nuc_df["NucleusID"].iloc[i]): int(cell_df["CellID"].iloc[idx[i]]) for i in range(len(idx))}

# =======================================================
# GPU INTENSITY FUNCTION
# =======================================================
def gpu_intensity(mask, img, compute_median=True, chunk_size=20000):
    """Compute per-label intensity stats on GPU (CuPy) using A6000."""
    try:
        import cupy as cp
        xp = cp
        log("🚀 Using GPU (CuPy) for intensity calculation")
    except ImportError:
        xp = np
        log("⚠️ CuPy not found — falling back to CPU")

    mask_gpu = xp.asarray(mask)
    img_gpu = xp.asarray(img, dtype=xp.float32)

    n_labels = int(mask_gpu.max())
    means = xp.zeros(n_labels, dtype=xp.float32)
    medians = xp.zeros(n_labels, dtype=xp.float32) if compute_median else None
    labels = xp.arange(1, n_labels + 1)

    for i in range(0, n_labels, chunk_size):
        sub_labels = labels[i:i + chunk_size]
        for lbl in sub_labels:
            vals = img_gpu[mask_gpu == lbl]
            if vals.size > 0:
                means[lbl - 1] = vals.mean()
                if compute_median:
                    medians[lbl - 1] = xp.median(vals)

        if xp.__name__ == "cupy":
            xp.cuda.Device().synchronize()
        gc.collect()

    means = xp.asnumpy(means)
    medians = xp.asnumpy(medians) if compute_median else None

    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()

    del mask_gpu, img_gpu
    gc.collect()
    return means, medians

# =======================================================
# GPU-OPTIMIZED INTENSITY WRAPPER
# =======================================================
def compute_marker_intensities_light(marker_file, cell_masks, nuc_masks, cache_dir):
    """GPU-optimized (A6000) mean/median intensity computation with caching."""
    name = marker_file.stem.strip()
    cache_file = cache_dir / f"{name}_intensity.json"

    if cache_file.exists():
        log(f"🔁 Loading cached intensities for {name}")
        with open(cache_file, "r") as f:
            return json.load(f)

    start = time.time()
    log(f"🧮 Computing intensities for {name} (GPU mode)...")
    img = tiff.imread(marker_file).astype(np.float32)

    cell_mean, cell_median = gpu_intensity(cell_masks, img, compute_median=True)
    nuc_mean, nuc_median = gpu_intensity(nuc_masks, img, compute_median=True)

    result = {
        "marker": name,
        "cell_mean": cell_mean.tolist(),
        "cell_median": cell_median.tolist(),
        "nuc_mean": nuc_mean.tolist(),
        "nuc_median": nuc_median.tolist(),
    }

    with open(cache_file, "w") as f:
        json.dump(result, f)

    del img
    torch.cuda.empty_cache()
    gc.collect()

    log(f"✅ Finished {name} in {time.time()-start:.1f}s")
    return result

# =======================================================
# MAIN PIPELINE
# =======================================================
for patient_dir in BASE_DIR.glob("patient_*"):
    log(f"🚀 Start {patient_dir.name}")
    out_csv = OUTPUT_DIR / f"{patient_dir.name}_features.csv"
    cache_dir = OUTPUT_DIR / f"{patient_dir.name}_cache"
    cache_dir.mkdir(exist_ok=True)

    nuc_mask_p = OUTPUT_DIR / f"{patient_dir.name}_nucleus_mask.tif"
    cell_mask_p = OUTPUT_DIR / f"{patient_dir.name}_cell_mask.tif"
    cell_props_csv = OUTPUT_DIR / f"{patient_dir.name}_cell_props.csv"
    nuc_props_csv = OUTPUT_DIR / f"{patient_dir.name}_nuc_props.csv"

    if out_csv.exists():
        log(f"✅ Already processed → skipping {patient_dir.name}")
        continue

    # nucleus marker
    nuc_path = None
    for n in NUCLEUS_MARKERS:
        f = list(patient_dir.glob(f"*{n}*.tif"))
        if f:
            nuc_path = f[0]
            log(f"Using nucleus ref {n}")
            break
    if nuc_path is None:
        log("⚠️ No nucleus marker → skip")
        continue
    nuc_img = tiff.imread(nuc_path)

    # nucleus segmentation
    nuc_masks = tiff.imread(nuc_mask_p) if nuc_mask_p.exists() else run_cellpose_segmentation(nuc_img, "nuclei")
    if not nuc_mask_p.exists():
        tiff.imwrite(nuc_mask_p, nuc_masks)

    # cell segmentation
    cell_ref = None
    for m in CELL_MARKERS:
        f = list(patient_dir.glob(f"*{m}*.tif"))
        if f:
            cell_ref = tiff.imread(f[0])
            log(f"Using {m} for cell segmentation")
            break
    if cell_ref is None:
        log("⚠️ No cyto marker → skip")
        continue
    cell_masks = tiff.imread(cell_mask_p) if cell_mask_p.exists() else run_cellpose_segmentation(cell_ref, "cyto2")
    if not cell_mask_p.exists():
        tiff.imwrite(cell_mask_p, cell_masks)

    # regionprops caching
    if cell_props_csv.exists() and nuc_props_csv.exists():
        log("📂 Loading cached regionprops")
        cell_df = pd.read_csv(cell_props_csv)
        nuc_df = pd.read_csv(nuc_props_csv)
    else:
        log("📏 Computing regionprops (once)")
        nuc_p = regionprops_table(nuc_masks, intensity_image=nuc_img,
                                  properties=("label", "area", "centroid", "solidity", "perimeter"))
        cell_p = regionprops_table(cell_masks, intensity_image=cell_ref,
                                   properties=("label", "area", "centroid", "solidity", "perimeter"))
        nuc_df = pd.DataFrame(nuc_p).rename(columns={"label": "NucleusID"})
        cell_df = pd.DataFrame(cell_p).rename(columns={"label": "CellID"})
        nuc_df.to_csv(nuc_props_csv, index=False)
        cell_df.to_csv(cell_props_csv, index=False)
        log("💾 Cached regionprops")

    # match nucleus↔cell
    mapping = match_nucleus_to_cell(nuc_df, cell_df)
    nuc_df["CellID"] = nuc_df["NucleusID"].map(mapping)
    log(f"Mapped {len(nuc_df)} nuclei → {len(cell_df)} cells")

    # intensity (GPU + chunked parallel)
    log("🧮 Intensity computation (GPU + chunked parallel)")
    markers = list(patient_dir.glob("*.tif"))
    results = []
    chunks = [markers[i:i+CHUNK_SIZE] for i in range(0, len(markers), CHUNK_SIZE)]

    for chunk_idx, chunk in enumerate(chunks, 1):
        log(f"⚙️  Processing chunk {chunk_idx}/{len(chunks)} ({len(chunk)} markers)...")
        chunk_results = Parallel(n_jobs=min(N_CORES, len(chunk)), verbose=0)(
            delayed(compute_marker_intensities_light)(m, cell_masks, nuc_masks, cache_dir)
            for m in chunk
        )
        results.extend(chunk_results)
        gc.collect()
        torch.cuda.empty_cache()

    # merge results
    log("📊 Merging results...")
    df = cell_df.copy()
    df["Image"] = f"{patient_dir.name}.qptiff"
    df["Object type"] = "Cell"
    for r in results:
        m = r["marker"]
        df[f"Cell:{m}:Mean"] = r["cell_mean"]
        df[f"Cell:{m}:Median"] = r["cell_median"]
        df[f"Nuc:{m}:Mean"] = r["nuc_mean"]
        df[f"Nuc:{m}:Median"] = r["nuc_median"]

    df.to_csv(out_csv, index=False)
    log(f"✅ Saved {out_csv} ({df.shape[0]}×{df.shape[1]})")

    del df, nuc_df, cell_df, nuc_masks, cell_masks
    gc.collect()
    torch.cuda.empty_cache()
    log(f"🏁 Done {patient_dir.name}\n")

log("🎯 All patients processed successfully.")