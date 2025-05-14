import torch
import numpy as np
import cv2
import nibabel as nib
from segment_anything import sam_model_registry
import os
from pathlib import Path
from tqdm import tqdm
import gc # Import garbage collector

# Define the feature extraction hook class
class EncoderFeatureExtractor:
    def __init__(self):
        self.features = None

    def hook(self, module, input, output):
        # Detach the tensor to prevent gradient tracking and move to CPU
        self.features = output.detach().cpu() # Shape: (1, 256, 64, 64)

    def clear(self):
        self.features = None

def load_specific_coronal_slices(mri_path, slice_indices):
    """
    Loads specific coronal slice indices from a 3D MRI volume.
    Args:
        mri_path (str or Path): Path to the MRI file.
        slice_indices (list[int]): A list of the specific coronal slice indices to extract.
    Returns:
        dict[int, np.ndarray]: A dictionary mapping the slice index to the loaded 2D slice.
                               Returns an empty dict if loading fails, volume is not 3D,
                               or none of the requested indices are valid.
    """
    slices_dict = {}
    try:
        img = nib.load(mri_path)
        volume = img.get_fdata()
        if volume.ndim != 3:
            print(f"Warning: Expected 3D volume, but got {volume.ndim} dimensions for {mri_path}. Skipping.")
            return {}

        coronal_depth = volume.shape[1] # Coronal dimension is typically the second one

        for index in slice_indices:
            if 0 <= index < coronal_depth:
                # Extract coronal slice: all sagittal, specific coronal index, all axial
                slice_2d = volume[:, index, :]
                # Optional: Rotate if needed for consistent orientation
                # slice_2d = np.rot90(slice_2d, k=1) # Adjust k as needed
                slices_dict[index] = slice_2d.astype(np.float32)
            else:
                print(f"Warning: Requested coronal slice index {index} is out of bounds (0-{coronal_depth-1}) for {mri_path}.")

        return slices_dict

    except Exception as e:
        print(f"Error loading or slicing {mri_path}: {e}")
        return {}

def preprocess_slice(slice_2d, device):
    """
    Preprocesses a 2D grayscale NumPy slice for MedSAM's image encoder.
    Matches the preprocessing in MedSAM's demo.py.

    Args:
        slice_2d (np.ndarray): The 2D grayscale slice (H, W).
        device (torch.device or str): The device to send the tensor to.

    Returns:
        torch.Tensor: The preprocessed tensor (1, 3, 1024, 1024). Returns None if error.
    """
    if slice_2d is None or slice_2d.size == 0:
        return None
    if slice_2d.ndim != 2:
        print(f"Warning: preprocess_slice expected 2D array, got {slice_2d.ndim}D. Skipping.")
        return None

    # 1. Normalize the slice to [0, 1]
    min_val = slice_2d.min()
    max_val = slice_2d.max()
    denominator = max_val - min_val
    slice_norm = (slice_2d - min_val) / np.clip(denominator, a_min=1e-8, a_max=None)

    # 2. Convert to 3 channels by stacking
    slice_3c = np.stack([slice_norm] * 3, axis=-1) # Shape: (H, W, 3)

    # 3. Resize to 1024x1024
    try:
        # Ensure correct dtype for cv2.resize
        if slice_3c.dtype != np.float32:
             slice_3c = slice_3c.astype(np.float32)
        img_resize = cv2.resize(
            slice_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC
        ) # Shape: (1024, 1024, 3)
    except Exception as e:
        print(f"Error during resizing: {e}")
        return None

    img_resize = np.clip(img_resize, 0, 1)

    # 4. Convert to Tensor, permute, add batch dim, move to device
    img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1) # (3, 1024, 1024)
    img_tensor = img_tensor.unsqueeze(0) # (1, 3, 1024, 1024)
    img_tensor = img_tensor.to(device)

    return img_tensor

def extract_and_save_individual_slice_features(mri_path, model, feature_extractor, base_output_dir, device, slice_indices_to_process):
    """
    Loads specific coronal slices for an MRI, preprocesses each, extracts features,
    and saves the features for each slice individually into a subject-specific subfolder.

    Args:
        mri_path (str or Path): Path to the input MRI file.
        model (torch.nn.Module): The loaded MedSAM model.
        feature_extractor (EncoderFeatureExtractor): The hook object.
        base_output_dir (str or Path): The base directory for the current split (e.g., .../train/).
        device (str or torch.device): The computation device.
        slice_indices_to_process (list[int]): List of coronal slice indices to process.
    """
    mri_path = Path(mri_path)
    base_output_dir = Path(base_output_dir)

    # Create a subfolder for this specific MRI file's features
    # Remove '.nii' or '.nii.gz' extension for the folder name
    mri_stem = mri_path.name.replace(".nii.gz", "").replace(".nii", "")
    subject_output_dir = base_output_dir / mri_stem
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if all target slices already exist for this subject to potentially skip
    all_exist = True
    for slice_idx in slice_indices_to_process:
        output_filename = subject_output_dir / f"slice_{slice_idx}.npy"
        if not output_filename.exists():
            all_exist = False
            break
    if all_exist:
        # print(f"All target slices already processed for {mri_stem}. Skipping.") # Optional print
        return

    # 1. Load the required coronal slices
    coronal_slices_dict = load_specific_coronal_slices(mri_path, slice_indices_to_process)
    if not coronal_slices_dict: # Handles empty dict from loading errors or no valid slices found
        print(f"Skipping {mri_path.name} due to slice loading errors or no valid slices found.")
        return

    model.eval() # Ensure model is in evaluation mode

    # 2. Process each loaded slice
    for slice_idx, slice_2d in coronal_slices_dict.items():
        # Define output path for this specific slice
        output_filename = subject_output_dir / f"slice_{slice_idx}.npy"

        # Skip if this individual slice file already exists
        if output_filename.exists():
            continue

        # Preprocess
        preprocessed_tensor = preprocess_slice(slice_2d, device)
        if preprocessed_tensor is None:
            print(f"Skipping slice {slice_idx} of {mri_path.name} due to preprocessing error.")
            continue # Skip this slice, try others for the same subject

        # Extract features
        feature_extractor.clear()
        with torch.no_grad():
            try:
                _ = model.image_encoder(preprocessed_tensor)
            except Exception as e:
                print(f"Error during model forward pass for slice {slice_idx} of {mri_path.name}: {e}")
                continue # Skip this slice

        # Get features (already on CPU from hook)
        current_features = feature_extractor.features # Shape: (1, 256, 64, 64)
        if current_features is None:
            print(f"Failed to extract features for slice {slice_idx} of {mri_path.name}.")
            continue # Skip this slice

        # Save the features for this individual slice
        try:
            # Squeeze the batch dimension before saving if desired, result shape (256, 64, 64)
            # features_to_save = current_features.squeeze(0).numpy()
            # Or save with the batch dimension, shape (1, 256, 64, 64)
            features_to_save = current_features.numpy()
            np.save(output_filename, features_to_save)
            # print(f"Saved features ({features_to_save.shape}) to: {output_filename}") # Optional
        except Exception as e:
            print(f"Error saving features for slice {slice_idx} of {mri_path.name} to {output_filename}: {e}")

        # Clean up tensor and cache
        del preprocessed_tensor, current_features, features_to_save
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_DATA_DIR = Path("/data/kuang/Projects/MedSAM/data/BrainAGE")
    # Changed output dir name to reflect content
    OUTPUT_BASE_DIR = Path("/data/kuang/Projects/MedSAM/data/BrainAGE_preprocessed_coronal_120_130")
    MEDSAM_CKPT_PATH = "medsam_vit_b.pth"
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu" # Use specific GPU if desired
    MRI_EXTENSIONS = ['.nii', '.nii.gz']
    SPLITS = ["train", "test", "validation"]

    # Define the target coronal slices
    CORONAL_SLICE_CENTER = 125
    CORONAL_SLICE_RANGE = 5 # 5 slices before and 5 slices after the center
    TARGET_CORONAL_INDICES = list(range(CORONAL_SLICE_CENTER - CORONAL_SLICE_RANGE,
                                        CORONAL_SLICE_CENTER + CORONAL_SLICE_RANGE + 1)) # e.g., 120 to 130
    # --- /Configuration ---

    print(f"Using device: {DEVICE}")
    print(f"Input data directory: {INPUT_DATA_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Processing Coronal slices: {TARGET_CORONAL_INDICES}")

    # --- Setup (Checks, File Collection, Model Loading, Hook) ---
    if not INPUT_DATA_DIR.is_dir():
        print(f"Error: Input data directory not found at {INPUT_DATA_DIR}")
        exit()
    if not os.path.exists(MEDSAM_CKPT_PATH):
        print(f"Error: MedSAM checkpoint not found at {MEDSAM_CKPT_PATH}")
        exit()

    all_mri_files = []
    output_dirs = {} # Maps split name to its output directory path
    print("Scanning for files...")
    for split in SPLITS:
        split_input_dir = INPUT_DATA_DIR / split
        split_output_dir = OUTPUT_BASE_DIR / split # Define output dir for this split
        output_dirs[split] = split_output_dir

        if not split_input_dir.is_dir():
            print(f"Warning: Input directory for split '{split}' not found: {split_input_dir}")
            continue
        # No need to create split_output_dir here, the saving function will create subject folders inside it

        split_files = []
        for ext in MRI_EXTENSIONS:
            split_files.extend(list(split_input_dir.glob(f'*{ext}')))
        if not split_files:
            print(f"No MRI files found in {split_input_dir} with extensions {MRI_EXTENSIONS}")
        else:
            all_mri_files.extend(split_files)
            print(f"Found {len(split_files)} files in split '{split}'.")

    if not all_mri_files:
        print("Error: No MRI files found in any split directory. Exiting.")
        exit()
    print(f"\nFound a total of {len(all_mri_files)} MRI files to process.")

    print("Loading MedSAM model...")
    try:
        medsam_model = sam_model_registry['vit_b'](checkpoint=MEDSAM_CKPT_PATH)
        medsam_model = medsam_model.to(DEVICE)
        medsam_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    feature_extractor = EncoderFeatureExtractor()
    image_encoder = medsam_model.image_encoder
    hook_handle = image_encoder.register_forward_hook(feature_extractor.hook)
    print(f"Hook registered to {type(image_encoder).__name__}")
    # --- /Setup ---


    # --- Process all files ---
    print(f"\nStarting feature extraction for {len(TARGET_CORONAL_INDICES)} coronal slices per volume...")
    for mri_file_path in tqdm(all_mri_files, desc="Processing dataset"):
        try:
            # Determine the correct output directory based on the input file's split
            split_name = mri_file_path.parent.name
            if split_name not in output_dirs:
                 print(f"Warning: Could not determine split for {mri_file_path}. Skipping.")
                 continue
            correct_output_dir_for_split = output_dirs[split_name]
        except IndexError:
             print(f"Warning: Could not determine split for {mri_file_path}. Skipping.")
             continue

        # Call the modified extraction function to save individual slice features
        extract_and_save_individual_slice_features(
            mri_path=mri_file_path,
            model=medsam_model,
            feature_extractor=feature_extractor,
            base_output_dir=correct_output_dir_for_split, # Pass the split's base output dir
            device=DEVICE,
            slice_indices_to_process=TARGET_CORONAL_INDICES
        )
        # Optional: Add explicit garbage collection more frequently if memory issues persist
        # gc.collect()
        # if DEVICE.startswith('cuda'):
        #     torch.cuda.empty_cache()

    # --- Cleanup ---
    hook_handle.remove()
    print("\nHook removed.")
    print("Batch processing finished.")
    print(f"Preprocessed features saved under: {OUTPUT_BASE_DIR}")