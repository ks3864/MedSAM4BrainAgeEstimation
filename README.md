# MedSAM

MedSAM is a medical image processing tool. This README covers basic setup, usage, and contribution guidelines.

## Installation
1. Clone the repository: git clone https://github.com/yourusername/MedSAM.git
2. Install dependencies: pip install -r requirements.txt
3. Download the model check point.

## Dataset
In this study, we leveraged neuroimaging and clinical data from 13 publicly available datasets: Alzheimer’s Disease Neuroimaging Initiative (ADNI), Brain Genomics Superstruct Project (BGSP), Neuroimaging in Frontotemporal Dementia (NIFD), Stress and Limbic Modeling (SLIM), Dallas Lifespan Brain Study (dlbs), Southwest Advanced Life-course Development (SALD), Australian Imaging, Biomarkers & Lifestyle Study of Aging (AIBL), Consortium for Reliability and Reproducibility (CoRR), SchizConnect, Information eXtraction from Images (IXI), Open Access Series of Imaging Studies (OAS1 and OAS2), and the Parkinson’s Progression Markers Initiative (PPMI). A total of 2,851 T1w MRI images were included, all aligned to the MNI-152 coordinate system with an affine transformation. This standardized approach maximizes the generalizability of results. The diverse data sources increase the transferability of our findings to future studies and clinical settings. For the experiments, we split the data into train/validation/test with an 8:1:1 ratio.

Download from: https://drive.google.com/drive/u/2/folders/15XyWVGmikFCtrRu-x0sTJCTtWxykZvvn


## Overview
This project contains scripts and a notebook that demonstrate how to:

1. Select specific slices (or a middle slice) from 3D MRI volumes.
2. Preprocess these slices for use with the MedSAM (Segment Anything Medical) model.
3. Extract features from the MedSAM image encoder and save them for further processing or analysis.

## File:  preprocessing_middle_axial.py
### Description:
This script focuses on extracting only the middle axial slice from 3D MRI volumes. It applies the same preprocessing steps (normalization, resizing to 1024×1024, stack to 3 channels) before passing the slice through the MedSAM image encoder. The resulting features are saved as .npy files.

### Key points:
1. Locates the central slice on the axial (Z) dimension automatically.
2. Useful for scenarios where only a single representative slice is needed.
3. Includes a clear usage pattern for hooking onto the MedSAM encoder and saving feature tensors.

## File: preprocessing_11_coronal.py
### Description: 
This script extracts a range of coronal slices around a chosen center index (e.g., +/- 5 slices around slice 125). Each slice is preprocessed and passed through the MedSAM encoder to obtain features, which are saved individually.

### Key points: 
1. Demonstrates how to process multiple adjacent slices (e.g., 120 to 130) in the coronal plane.
2. Creates a folder per subject and stores separate .npy feature files for each slice index.
3. Can be adapted for other slice ranges or different planes.

## File: preprocessing_selected_slices.py
### Description: 
This script loads 3D MRI volumes, extracts specific slices (by index) in multiple views (sagittal, coronal, axial), preprocesses them (normalization, resizing, etc.), then uses MedSAM’s image encoder to get feature embeddings. The features are saved as NumPy arrays for each slice.

### Key points: 
1. Configurable slice selection through the TARGET_SLICES dictionary (e.g., indexes 80 and 125 in sagittal/coronal/axial views).
2. Demonstrates setting up a “forward hook” on the MedSAM image encoder to capture intermediate features.
3. Saves each slice’s features in a subject-specific folder structure.

### File: train_compare_MLPs.ipynb
### Description:
Trains and compares multiple MLP architectures for age prediction from 256-dimensional embeddings. Runs repeated experiments (configurable number of runs) for each model to compute average MAE and standard deviation.

### Three MLP variants:
1. AgeMLP_2Layer
2. AgeMLP_3Layer
3. AgeMLP_3Layer_AttentionBN (adds a multi-head attention layer + BatchNorm)

Uses a CSV with subject filename–age mappings and .npy embedding files.
Implements early stopping and learning rate scheduler based on validation performance.
Logs and prints final average test MAE and validation MAE (mean ± std) across multiple runs.

### How It Works:

Datasets & DataLoaders: 
1. Loads data from /train, /validation, and /test directories (each containing .npy embeddings).
2. Reads the CSV file to retrieve ages for each subject.
3. Pools or reshapes embeddings into a 256-length vector.
Model Training:
1. Each model is instantiated, trained (performing multiple epochs), then validated.
2. Best model checkpoint is saved based on validation MAE.
3. Early stopping halts training if no improvement is seen for a specified patience.

Comparison & Results:
At the end of training, prints a table with average and standard deviation of test MAE across multiple runs per MLP variant.

Usage Notes: • Configure paths, hyperparameters, device selection (CPU/GPU) at the beginning.
1. Adjust EPOCHS, BATCH_SIZE, etc. as needed for your dataset.
2. Set NUM_RUNS to control how many repeated trials are run per model.

## File: train_coronal120-130.ipynb
### Description:
Demonstrates training strategies for slice-based age prediction using coronal slices (indices 120 to 130, inclusive).
Includes multiple approaches:
-   Individual MLP model per slice (then averaging predictions).
-   Early-fusion MLP that concatenates slice embeddings into a single vector.
-   Fusion via averaging or simple multi-head attention strategies.

1. Slice-Specific Training:
-   Each coronal slice is treated as a separate model.
-   Final inference “averages” the predictions of all slice-specific models.
2. Early-Fusion MLP:
-   Concatenates the embeddings for a set of slices (e.g., 11 slices) into one vector.
-   Trains a single MLP on the combined input.
3. Feature Fusion with Multi-head Attention (optional approach):
-   Demonstrates how to fuse slice representations either via a simple average or using an attention mechanism.
4. Code snippet for evaluating the best model on the test set, including a test_model function.

## File: train_4_selected_slices.ipynb
### Description
This Jupyter notebook demonstrates how to train or fine-tune a model (potentially MedSAM or a related model) on selected 2D slices extracted from 3D MRI data. It likely includes code to load the extracted features or slices, define the training pipeline, and run the training loop.
-   Key points:
1. Shows how to incorporate preprocessed slices into a PyTorch (or similar) training routine.
2. Can be adapted to train on multiple slices per 3D volume.