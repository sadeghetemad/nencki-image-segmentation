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

N_CORES = 2
CHUNK_SIZE = 1
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
    masks, _, _ = model.eval(img, diameter=None, channels=None, batch_size=1)
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
# FAST INTENSITY
# =======================================================
# def fast_intensity(mask, img):
#     """Compute mean intensity for each label using regionprops_table (C-optimized)."""
#     props = regionprops_table(mask, intensity_image=img, properties=("label", "mean_intensity"))
#     df = pd.DataFrame(props).rename(columns={"label": "LabelID", "mean_intensity": "MeanIntensity"})
#     return df

def fast_intensity(mask, img, tile_size=8192):
    """
    Compute mean intensity per label using tile-wise accumulation.
    Efficient for huge masks (prevents MemoryError).
    """
    h, w = mask.shape
    labels = np.unique(mask)
    labels = labels[labels > 0]
    n_labels = mask.max()

    sums = np.zeros(n_labels + 1, dtype=np.float64)
    counts = np.zeros(n_labels + 1, dtype=np.int64)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            m_tile = mask[y:y + tile_size, x:x + tile_size]
            i_tile = img[y:y + tile_size, x:x + tile_size]

            valid = m_tile > 0
            if not np.any(valid):
                continue

            flat_mask = m_tile[valid].ravel()
            flat_img = i_tile[valid].ravel()

            # accumulate sums and counts
            np.add.at(sums, flat_mask, flat_img)
            np.add.at(counts, flat_mask, 1)

        gc.collect()

    means = np.zeros_like(sums, dtype=np.float32)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]

    df = pd.DataFrame({
        "LabelID": np.arange(1, len(means)),
        "MeanIntensity": means[1:]
    })
    return df


# =======================================================
# INTENSITY WRAPPER (FAST + CACHED)
# =======================================================
def compute_marker_intensities_light(marker_file, cell_masks, nuc_masks, cache_dir):
    name = marker_file.stem.strip()
    cache_file = cache_dir / f"{name}_intensity.json"

    if cache_file.exists():
        log(f"🔁 Loading cached intensities for {name}")
        with open(cache_file, "r") as f:
            return json.load(f)

    start = time.time()
    log(f"🧮 Computing intensities for {name} (FAST mode)...")
    img = tiff.imread(marker_file).astype(np.float32)

    # mean intensity per label
    cell_int = fast_intensity(cell_masks, img, tile_size=12000)
    nuc_int = fast_intensity(nuc_masks, img, tile_size=12000)

    result = {
        "marker": name,
        "cell_df": cell_int.to_dict(orient="list"),
        "nuc_df": nuc_int.to_dict(orient="list"),
    }

    with open(cache_file, "w") as f:
        json.dump(result, f)

    del img
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
    def safe_regionprops(mask, img=None, tile_size=4096):
        """Compute regionprops in tiles to reduce memory usage."""
        from skimage.measure import regionprops_table
        h, w = mask.shape
        props_list = []

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                m_tile = mask[y:y+tile_size, x:x+tile_size]
                if np.all(m_tile == 0):
                    continue
                i_tile = img[y:y+tile_size, x:x+tile_size] if img is not None else None

                props = regionprops_table(
                    m_tile,
                    intensity_image=i_tile,
                    properties=("label", "area", "centroid", "solidity", "perimeter")
                )
                props_list.append(pd.DataFrame(props))

            gc.collect()

        if not props_list:
            return pd.DataFrame(columns=["label", "area", "centroid-0", "centroid-1", "solidity", "perimeter"])

        df = pd.concat(props_list, ignore_index=True)
        df = df.groupby("label", as_index=False).first()  # remove duplicates
        return df


    if cell_props_csv.exists() and nuc_props_csv.exists():
        log("📂 Loading cached regionprops")
        cell_df = pd.read_csv(cell_props_csv)
        nuc_df = pd.read_csv(nuc_props_csv)
    else:
        log("📏 Computing regionprops safely (tile-wise)...")
        nuc_df = safe_regionprops(nuc_masks, nuc_img, tile_size=4096)
        cell_df = safe_regionprops(cell_masks, cell_ref, tile_size=4096)
        nuc_df.rename(columns={"label": "NucleusID"}, inplace=True)
        cell_df.rename(columns={"label": "CellID"}, inplace=True)
        nuc_df.to_csv(nuc_props_csv, index=False)
        cell_df.to_csv(cell_props_csv, index=False)
        log("💾 Cached regionprops safely")


    # if cell_props_csv.exists() and nuc_props_csv.exists():
    #     log("📂 Loading cached regionprops")
    #     cell_df = pd.read_csv(cell_props_csv)
    #     nuc_df = pd.read_csv(nuc_props_csv)
    # else:
    #     log("📏 Computing regionprops (once)")
    #     nuc_p = regionprops_table(nuc_masks, intensity_image=nuc_img,
    #                               properties=("label", "area", "centroid", "solidity", "perimeter"))
    #     cell_p = regionprops_table(cell_masks, intensity_image=cell_ref,
    #                                properties=("label", "area", "centroid", "solidity", "perimeter"))
    #     nuc_df = pd.DataFrame(nuc_p).rename(columns={"label": "NucleusID"})
    #     cell_df = pd.DataFrame(cell_p).rename(columns={"label": "CellID"})
    #     nuc_df.to_csv(nuc_props_csv, index=False)
    #     cell_df.to_csv(cell_props_csv, index=False)
    #     log("💾 Cached regionprops")

    # match nucleus↔cell
    mapping = match_nucleus_to_cell(nuc_df, cell_df)
    nuc_df["CellID"] = nuc_df["NucleusID"].map(mapping)
    log(f"Mapped {len(nuc_df)} nuclei → {len(cell_df)} cells")

    # intensity (FAST + cached)
    log("🧮 Intensity computation (FAST parallel mode)")
    markers = list(patient_dir.glob("*.tif"))
    results = []
    chunks = [markers[i:i+CHUNK_SIZE] for i in range(0, len(markers), CHUNK_SIZE)]

    for chunk_idx, chunk in enumerate(chunks, 1):
        log(f"⚙️ Processing chunk {chunk_idx}/{len(chunks)} ({len(chunk)} markers)...")
        chunk_results = Parallel(n_jobs=min(N_CORES, len(chunk)), verbose=0)(
            delayed(compute_marker_intensities_light)(m, cell_masks, nuc_masks, cache_dir)
            for m in chunk
        )
        results.extend(chunk_results)
        gc.collect()

    # =======================================================
    # MERGE RESULTS (FIXED VERSION)
    # =======================================================
        # =======================================================
    # BUILD FINAL TABLE (PER PATIENT)
    # =======================================================
    log("📊 Building final result table...")

    # 1️⃣ Base DataFrame from cell properties
    df = cell_df.copy()
    df.rename(columns={
        "CellID": "Object ID",
        "centroid-1": "Centroid X µm",
        "centroid-0": "Centroid Y µm",
        "area": "Cell: Area µm^2",
        "perimeter": "Cell: Length µm",
        "solidity": "Cell: Solidity",
    }, inplace=True)

    # Add basic columns
    df["Image"] = f"{patient_dir.name}.qptiff"
    df["Object type"] = "Cell"
    df["Name"] = ""
    df["Classification"] = ""
    df["Parent"] = ""
    df["ROI"] = "ROI-1"

    # 2️⃣ Add mean and median intensities for each marker
    for r in results:
        m = r["marker"]

        # Convert to DataFrames
        cell_tmp = pd.DataFrame(r["cell_df"])
        nuc_tmp = pd.DataFrame(r["nuc_df"])

        # Compute median too (based on mean proxy here)
        cell_df_tmp = pd.DataFrame({
            "CellID": cell_tmp["LabelID"],
            f"Cell:{m}:Mean": cell_tmp["MeanIntensity"],
            f"Cell:{m}:Median": cell_tmp["MeanIntensity"],  # placeholder (same as mean)
        })
        nuc_df_tmp = pd.DataFrame({
            "NucleusID": nuc_tmp["LabelID"],
            f"Nucleus:{m}:Mean": nuc_tmp["MeanIntensity"],
            f"Nucleus:{m}:Median": nuc_tmp["MeanIntensity"],  # placeholder
        })

        # Merge with nucleus-cell mapping
        inv_map = {v: k for k, v in mapping.items()}
        df["NucleusID"] = df["Object ID"].map(inv_map)

        df = df.merge(cell_df_tmp, left_on="Object ID", right_on="CellID", how="left").drop(columns=["CellID"])
        df = df.merge(nuc_df_tmp, on="NucleusID", how="left")

    # 3️⃣ Clean up and reorder columns
    drop_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Sort columns (optional)
    first_cols = [
        "Image", "Object ID", "Object type", "Name", "Classification",
        "Parent", "ROI", "Centroid X µm", "Centroid Y µm",
        "Cell: Area µm^2", "Cell: Length µm", "Cell: Solidity"
    ]
    df = df[[c for c in first_cols if c in df.columns] +
            [c for c in df.columns if c not in first_cols]]

    # 4️⃣ Save final table
    out_csv = OUTPUT_DIR / f"{patient_dir.name}_features_final.csv"
    df.to_csv(out_csv, index=False)
    log(f"✅ Saved {out_csv} ({df.shape[0]}×{df.shape[1]})")

    # cleanup memory
    del df, nuc_df, cell_df, nuc_masks, cell_masks
    gc.collect()


    # log("📊 Merging results...")
    # df = cell_df.copy()
    # df["Image"] = f"{patient_dir.name}.qptiff"
    # df["Object type"] = "Cell"

    # for r in results:
    #     m = r["marker"]
    #     cell_tmp = pd.DataFrame(r["cell_df"]).rename(columns={
    #         "LabelID": "CellID",
    #         "MeanIntensity": f"Cell:{m}:Mean"
    #     })
    #     nuc_tmp = pd.DataFrame(r["nuc_df"]).rename(columns={
    #         "LabelID": "NucleusID",
    #         "MeanIntensity": f"Nuc:{m}:Mean"
    #     })

    #     df = df.merge(cell_tmp, on="CellID", how="left")
    #     df = df.merge(nuc_tmp, on="NucleusID", how="left")

    # df.to_csv(out_csv, index=False)
    # log(f"✅ Saved {out_csv} ({df.shape[0]}×{df.shape[1]})")
    # del df, nuc_df, cell_df, nuc_masks, cell_masks

    # del nuc_df, cell_df, nuc_masks, cell_masks
    # gc.collect()
    # log(f"🏁 Done {patient_dir.name}\n")

log("🎯 All patients processed successfully.")
