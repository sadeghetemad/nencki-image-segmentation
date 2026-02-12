# 🧬 Multi-Resolution Cell Image Segmentation and Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides a complete workflow for **cell segmentation on
multi-resolution microscopy images** (e.g., `.qptiff`) followed by
**downstream patient-level feature analysis**.

------------------------------------------------------------------------

## 📂 Repository Structure

├── data/
│ └── features_final/ # Final extracted cellular features
│
├── image/ # Raw microscopy images (.qptiff, etc.)
│
├── notebooks/
│ └── downstream_patient_analysis.ipynb
│ # Feature analysis and visualization
│
├── src/
│ ├── 0_multiple_patient_image_splitter.py
│ │ # Splits large multi-patient whole-slide images
│ │
│ └── 1_cell_segmentation_cellposeSAM.py
│ # Cell segmentation using Cellpose + SAM
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

------------------------------------------------------------------------

## ⚙️ Installation

```bash
# Clone the repository
git clone git@github.com:Occhipinti-Lab/cell-image-segmentation-and-analysis.git
cd cell-image-segmentation-and-analysis

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Upgrade pip (recommended)
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Pipeline Workflow

### 1️⃣ Image Splitting

python src/0_multiple_patient_image_splitter.py

This step: - Splits large images - Organizes patient-level regions -
Prepares input for segmentation

------------------------------------------------------------------------

### 2️⃣ Cell Segmentation

python src/1_cell_segmentation_cellposeSAM.py

This step: - Handles multi-resolution microscopy images - Applies
Cellpose segmentation - Optionally integrates SAM refinement - Generates
cell masks

Final outputs are stored in:

data/features_final/

------------------------------------------------------------------------

### 3️⃣ Downstream Patient Analysis

jupyter notebook notebooks/downstream_patient_analysis.ipynb

This notebook performs: - Mask cleanup - Feature extraction (size,
morphology, intensity) - Patient-level aggregation - Statistical
summaries - Visualization 

------------------------------------------------------------------------

## 🔬 Key Features

-   Multi-resolution QPTIFF support
-   Automated tiling for whole-slide images
-   Deep learning-based segmentation (Cellpose + SAM)
-   Feature quantification
-   Patient-level aggregation

------------------------------------------------------------------------

## 🧠 Requirements

-   Python 3.10+
-   NumPy
-   Pandas
-   Cellpose
-   Jupyter Notebook

See `requirements.txt` for full dependency list.

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests and discussions are welcome.

Please: - Keep scripts modular - Document functions clearly - Clear
notebook outputs before committing

------------------------------------------------------------------------

## 📜 License

MIT License --- © Occhipinti Lab
