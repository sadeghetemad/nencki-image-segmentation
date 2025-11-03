import napari
import numpy as np
import tifffile as tiff


img = 'data/patients_split/patient_3149.tif'
cell_mask = 'data/features_final/patient_3149_cell_mask.tif'
nuc_mask = 'data/features_final/patient_3149_nucleus_mask.tif'

img = tiff.imread(img)
cell_mask = tiff.imread(cell_mask)
nuc_mask = tiff.imread(nuc_mask)


viewer = napari.Viewer()
viewer.add_image(img, name="Original", contrast_limits=[0, np.percentile(img, 99)])
viewer.add_labels(cell_mask, name="Cells", opacity=0.3)
layer_nuc = viewer.add_labels(nuc_mask, name="Nuclei", opacity=0.5)
layer_nuc.color = {1: 'red'}
napari.run()