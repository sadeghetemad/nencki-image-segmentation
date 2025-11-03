import os, gc, json, time, psutil, torch, tifffile as tiff
import numpy as np, pandas as pd, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import remove_small_objects
from cellpose import models

# =======================================================
# CONFIGURATION
# =======================================================
BASE_DIR = Path("data/patients_split")
OUTPUT_DIR = Path("data/features_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUCLEUS_MARKERS = ["DAPI", "H3K27M"]
CELL_MARKERS = ["GLUT1", "TMEM119", "CD68", "CD45", "LGALS3", "SPP1", "CD31"]

MAX_INTENSITY_WORKERS = 16  # ← parallel CPU workers for intensity
TILE_SIZE = 6000            # ← smaller tiles for huge gigapixel images

# =======================================================
def log(msg):
    cpu, mem = psutil.cpu_percent(None), psutil.virtual_memory().percent
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | CPU {cpu:.1f}% MEM {mem:.1f}%")

# =======================================================
# IMAGE SCORING
# =======================================================
def score_image(img):
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0: img /= img.max()
    thr = threshold_otsu(img)
    fg = remove_small_objects(img > thr, 16)
    contrast = np.percentile(img, 99) - np.percentile(img, 1)
    sharpness = np.median(sobel(img))
    fg_frac = fg.mean()
    return (contrast * 0.6 + sharpness * 0.4) * (0.1 < fg_frac < 0.5)

def pick_best_marker(patient_dir, markers):
    best_score, best_path = -1, None
    for marker in markers:
        for f in patient_dir.glob(f"*{marker}*.tif"):
            try:
                img = tiff.imread(f)
                if img.ndim == 3: img = img.max(axis=0)
                s = score_image(img)
                if s > best_score:
                    best_score, best_path = s, f
            except Exception as e:
                log(f"Error scoring {f}: {e}")
    if best_path:
        log(f"✅ Best channel: {best_path.name} (score={best_score:.3f})")
    return best_path

# =======================================================
# CELLPOSE SEGMENTATION
# =======================================================
def run_cellpose_cyto3_dual(cyto_img, nuc_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Running Cellpose dual-channel model on {device}")
    model = models.CellposeModel(pretrained_model="cpsam", device=device, gpu=(device.type=="cuda"))
    img = np.stack([cyto_img, nuc_img], axis=-1).astype(np.float32)
    masks, _, _ = model.eval(img, diameter=None, batch_size=4, normalize=True)
    log(f"✅ Cell segmentation done → {masks.max()} cells")
    return masks.astype(np.uint32)

def run_cellpose_nuclei(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"🧠 Running Cellpose nuclei model on {device}")
    model = models.CellposeModel(pretrained_model="cpsam", device=device, gpu=(device.type=="cuda"))
    masks, _, _ = model.eval(img, diameter=None, batch_size=4, normalize=True)
    log(f"✅ Nucleus segmentation done → {masks.max()} nuclei")
    return masks.astype(np.uint32)

# =======================================================
# INTENSITY CALCULATION (parallel)
# =======================================================
def fast_intensity_with_median(mask, img, tile_size=TILE_SIZE):
    h, w = mask.shape
    n_labels = mask.max()
    sums = np.zeros(n_labels + 1, dtype=np.float64)
    counts = np.zeros(n_labels + 1, dtype=np.int64)
    pixel_values = defaultdict(list)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            m_tile, i_tile = mask[y:y+tile_size,x:x+tile_size], img[y:y+tile_size,x:x+tile_size]
            valid = m_tile > 0
            if not np.any(valid): continue
            flat_mask, flat_img = m_tile[valid], i_tile[valid]
            np.add.at(sums, flat_mask, flat_img)
            np.add.at(counts, flat_mask, 1)
            for lbl in np.unique(flat_mask):
                vals = flat_img[flat_mask==lbl]
                if len(vals)>5000: vals=np.random.choice(vals,5000,replace=False)
                pixel_values[lbl].extend(vals.tolist())
        gc.collect()
    means=np.zeros_like(sums,dtype=np.float32)
    medians=np.zeros_like(sums,dtype=np.float32)
    valid=counts>0
    means[valid]=sums[valid]/counts[valid]
    for lbl,vals in pixel_values.items():
        if len(vals): medians[lbl]=np.median(vals)
    return pd.DataFrame({
        "LabelID": np.arange(1,len(means)),
        "MeanIntensity": means[1:],
        "MedianIntensity": medians[1:]
    })

def compute_marker_intensity(marker_file, mask_path, mask_type, cache_dir):
    import tifffile as tiff, pandas as pd, numpy as np, gc, json, time
    name = Path(marker_file).stem
    cache_file = cache_dir / f"{name}_intensity.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    start = time.time()
    img = tiff.imread(marker_file).astype(np.float32)
    mask = tiff.imread(mask_path)
    df = fast_intensity_with_median(mask, img)
    all_labels = np.unique(mask)[1:]
    df = df.set_index("LabelID").reindex(all_labels, fill_value=0).reset_index()
    result = {"marker": name, "mask_type": mask_type, "cell_df": df.to_dict(orient="list")}
    with open(cache_file, "w") as f: json.dump(result, f)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Done {name} ({mask_type}) in {time.time()-start:.1f}s")
    del df, mask, img; gc.collect()
    return result

# =======================================================
# MAIN PIPELINE (Sequential patients, parallel markers)
# =======================================================
if __name__ == "__main__":
    for patient_dir in BASE_DIR.glob("patient_*"):
        log(f"🚀 Start {patient_dir.name}")
        out_csv = OUTPUT_DIR / f"{patient_dir.name}_measurements.csv"
        cache_dir = OUTPUT_DIR / f"{patient_dir.name}_cache"
        cache_dir.mkdir(exist_ok=True)

        if out_csv.exists():
            log(f"Already processed → skipping {patient_dir.name}")
            continue

        cell_mask_p = OUTPUT_DIR / f"{patient_dir.name}_cell_mask.tif"
        nuc_mask_p = OUTPUT_DIR / f"{patient_dir.name}_nucleus_mask.tif"

        # --- segmentation (only if masks missing)
        if cell_mask_p.exists() and nuc_mask_p.exists():
            log("Found existing masks → skip segmentation")
            cell_masks = tiff.imread(cell_mask_p)
            nuc_mask_global = tiff.imread(nuc_mask_p)
            cyto_img = np.zeros_like(cell_masks, dtype=np.float32)
        else:
            nuc_path = pick_best_marker(patient_dir, NUCLEUS_MARKERS)
            cyto_path = pick_best_marker(patient_dir, CELL_MARKERS)
            if nuc_path is None or cyto_path is None:
                log("❌ Missing markers → skip patient")
                continue
            nuc_img = tiff.imread(nuc_path)
            cyto_img = tiff.imread(cyto_path)
            cell_masks = run_cellpose_cyto3_dual(cyto_img, nuc_img)
            tiff.imwrite(cell_mask_p, cell_masks)
            nuc_mask_global = run_cellpose_nuclei(nuc_img)
            tiff.imwrite(nuc_mask_p, nuc_mask_global)

        # --- cell geometry features
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
            return df.groupby("label", as_index=False).first()

        cell_df = safe_regionprops(cell_masks, cyto_img)
        cell_df.rename(columns={"label":"CellID"}, inplace=True)

        # --- parallel intensity
        tasks=[]
        for f in patient_dir.glob("*.tif"):
            name=f.stem
            if name in NUCLEUS_MARKERS: tasks.append((f, nuc_mask_p, "Nucleus", cache_dir))
            elif name in CELL_MARKERS: tasks.append((f, cell_mask_p, "Cell", cache_dir))
        results=[]
        if tasks:
            log(f"⚙️ Computing intensities with {MAX_INTENSITY_WORKERS} CPU workers...")
            with ProcessPoolExecutor(max_workers=MAX_INTENSITY_WORKERS) as executor:
                futures={executor.submit(compute_marker_intensity,*t):t[0].name for t in tasks}
                for fut in as_completed(futures):
                    try: results.append(fut.result())
                    except Exception as e: log(f"❌ Error {futures[fut]}: {e}")

        # --- merge results
        df = cell_df.copy()
        df.rename(columns={
            "CellID":"Object ID","centroid-1":"Centroid X µm","centroid-0":"Centroid Y µm",
            "area":"Cell: Area µm^2","perimeter":"Cell: Length µm","solidity":"Cell: Solidity"
        }, inplace=True)
        df["Cell: Circularity"]=4*np.pi*df["Cell: Area µm^2"]/(df["Cell: Length µm"]**2+1e-9)
        df["Image"]=f"{patient_dir.name}.qptiff"
        df["Object type"]="Cell"
        df["ROI"]="ROI-1"

        for r in results:
            m=r["marker"]; mt=r.get("mask_type","Cell")
            tmp=pd.DataFrame(r["cell_df"])
            df[f"{mt}: {m}: Mean"]=df["Object ID"].map(pd.Series(tmp["MeanIntensity"].values,index=tmp["LabelID"]))
            df[f"{mt}: {m}: Median"]=df["Object ID"].map(pd.Series(tmp["MedianIntensity"].values,index=tmp["LabelID"]))

        df.to_csv(out_csv, index=False)
        log(f"💾 Saved {out_csv} ({df.shape[0]}×{df.shape[1]})")
        del df, cell_df, cell_masks, nuc_mask_global
        gc.collect()

    log("🎯 All patients processed successfully (single-patient mode, parallel intensities).")
