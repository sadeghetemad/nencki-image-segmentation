import os
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from cellpose import models
from skimage.measure import regionprops_table
from scipy.spatial import cKDTree
from tqdm import tqdm
import torch

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BASE_DIR = Path("data/patients_split")
OUTPUT_DIR = Path("data/features_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Marker categories
NUCLEUS_MARKERS = ["DAPI", "H3K27M", "Ki67"]
CELL_MARKERS = [
    "GLUT1", "GFAP", "CD163", "SOX4", "CD31", "TMEM119", "HLA-A",
    "CD68", "CD20", "CD4", "CD8", "GPNMB", "PD-L1", "CD45RO", "SMA",
    "CD45", "CD3e", "HLA-DR", "PDGFRA", "NESTIN", "LGALS3", "PD-1",
    "MOG", "SPP1", "ALDH1L1", "FOXP3", "CD68"
] #HLA-DR

# -------------------------------------------------------
# Helper: segment image using Cellpose
# -------------------------------------------------------
# def run_cellpose_segmentation(image, model_type):
#     model = models.CellposeModel(model_type=model_type, gpu=True)
#     masks, _, _, _ = model.eval(image, diameter=None, channels=[0, 0])
#     return masks.astype(np.uint16)

def run_cellpose_segmentation(image, model_type="nuclei"):
    # ✅ Proper torch device object
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert multi-channel to single-channel (Cellpose expects grayscale)
    if image.ndim == 3:
        image = image[..., 0]

    print(f"🧩 Running Cellpose (v4) on {device} using model_type='{model_type}'")

    # Create model (works for v4.0.1+)
    model = models.CellposeModel(gpu=(device.type == "cuda"), device=device, model_type=model_type)

    # Evaluate
    masks, _, _ = model.eval(image, diameter=None, channels=None)
    print(f"✅ Segmentation complete — {masks.max()} objects detected")

    return masks.astype(np.uint16)



# -------------------------------------------------------
# Helper: match nucleus to nearest cell centroid
# -------------------------------------------------------
def match_nucleus_to_cell(nuc_props, cell_props):
    nuc_xy = np.array(list(zip(nuc_props["centroid-0"], nuc_props["centroid-1"])))
    cell_xy = np.array(list(zip(cell_props["centroid-0"], cell_props["centroid-1"])))

    tree = cKDTree(cell_xy)
    dist, idx = tree.query(nuc_xy, k=1)
    mapping = {int(nuc_props["label"][i]): int(cell_props["label"][idx[i]]) for i in range(len(idx))}
    return mapping

# -------------------------------------------------------
# Main processing loop for each patient
# -------------------------------------------------------
for patient_dir in BASE_DIR.glob("patient_*"):
    print(f"\n🧠 Processing {patient_dir.name}")

    # 1️⃣ Load nucleus reference (DAPI preferred)
    nuc_path = None
    for name in NUCLEUS_MARKERS:
        found = list(patient_dir.glob(f"*{name}*.tif"))
        if found:
            nuc_path = found[0]
            print(f"   ➤ Using {name} as nucleus reference")
            break
    if nuc_path is None:
        raise RuntimeError(f"No nucleus marker found in {patient_dir}")

    nuc_img = tiff.imread(nuc_path)

    # 2️⃣ Segment nuclei
    nuc_masks = run_cellpose_segmentation(nuc_img, model_type="nuclei")
    tiff.imwrite(OUTPUT_DIR / f"{patient_dir.name}_nucleus_mask.tif", nuc_masks)
    print(f"   🧩 Nucleus mask saved ({nuc_masks.max()} objects)")

    # 3️⃣ Segment cells (use one strong cytoplasmic marker, e.g., GFAP or CD68)
    cell_ref = None
    for marker in CELL_MARKERS:
        f = list(patient_dir.glob(f"*{marker}*.tif"))
        if f:
            cell_ref = tiff.imread(f[0])
            print(f"   ➤ Using {marker} as cell segmentation reference")
            break
    if cell_ref is None:
        raise RuntimeError("No cytoplasmic marker found for cell segmentation")

    cell_masks = run_cellpose_segmentation(cell_ref, model_type="cyto2")
    tiff.imwrite(OUTPUT_DIR / f"{patient_dir.name}_cell_mask.tif", cell_masks)
    print(f"   🧫 Cell mask saved ({cell_masks.max()} objects)")

    # 4️⃣ Extract region properties
    nuc_props = regionprops_table(
        nuc_masks, intensity_image=nuc_img,
        properties=("label", "area", "centroid", "solidity", "perimeter")
    )
    cell_props = regionprops_table(
        cell_masks, intensity_image=cell_ref,
        properties=("label", "area", "centroid", "solidity", "perimeter")
    )

    nuc_df = pd.DataFrame(nuc_props).rename(columns={"label": "NucleusID"})
    cell_df = pd.DataFrame(cell_props).rename(columns={"label": "CellID"})

    # 5️⃣ Match nucleus ↔ cell
    mapping = match_nucleus_to_cell(nuc_props, cell_props)
    nuc_df["CellID"] = nuc_df["NucleusID"].map(mapping)

    # 6️⃣ Quantify intensity per marker
    print("   ➤ Measuring marker intensities...")
    df_final = cell_df.copy()
    df_final["Image"] = f"{patient_dir.name}.qptiff"
    df_final["Object type"] = "Cell"

    for marker_file in tqdm(list(patient_dir.glob("*.tif"))):
        marker_name = marker_file.stem.strip()
        img = tiff.imread(marker_file)

        mean_vals, med_vals = [], []
        for i in range(1, cell_masks.max() + 1):
            vals = img[cell_masks == i]
            mean_vals.append(np.mean(vals))
            med_vals.append(np.median(vals))
        df_final[f"Cell: {marker_name}: Mean"] = mean_vals
        df_final[f"Cell: {marker_name}: Median"] = med_vals

        mean_nuc, med_nuc = [], []
        for i in range(1, nuc_masks.max() + 1):
            vals = img[nuc_masks == i]
            mean_nuc.append(np.mean(vals))
            med_nuc.append(np.median(vals))
        df_final[f"Nucleus: {marker_name}: Mean"] = np.pad(mean_nuc, (0, len(df_final)-len(mean_nuc)), constant_values=np.nan)
        df_final[f"Nucleus: {marker_name}: Median"] = np.pad(med_nuc, (0, len(df_final)-len(med_nuc)), constant_values=np.nan)

    # 7️⃣ Save CSV
    out_csv = OUTPUT_DIR / f"{patient_dir.name}_features.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"💾 Saved feature table: {out_csv}")

print("\n🎉 Done! Feature tables created for both patients.")
