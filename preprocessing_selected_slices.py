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
        self.features = output.detach().cpu()

    def clear(self):
        self.features = None

def load_specific_slice(mri_path, view, index):
    """
    Loads a specific slice from a 3D MRI volume based on view and index.

    Args:
        mri_path (str or Path): Path to the MRI file.
        view (str): The view ('sagittal', 'coronal', or 'axial').
        index (int): The slice index for the specified view.

    Returns:
        np.ndarray: The specified 2D slice. Returns None if loading or slicing fails.
    """
    try:
        img = nib.load(mri_path)
        # Reorient to standard RAS+ orientation if possible
        img = nib.as_closest_canonical(img)
        volume = img.get_fdata()

        if volume.ndim != 3:
            # print(f"Error: Expected 3D volume, got {volume.ndim} dimensions for {mri_path}")
            return None

        # Get slice based on view
        if view == 'sagittal':
            if index < 0 or index >= volume.shape[0]:
                # print(f"Error: Sagittal index {index} out of bounds for shape {volume.shape} in {mri_path}")
                return None
            slice_2d = volume[index, :, :]
            # Optional: Transpose if needed for consistent orientation, e.g., slice_2d = slice_2d.T
        elif view == 'coronal':
            if index < 0 or index >= volume.shape[1]:
                # print(f"Error: Coronal index {index} out of bounds for shape {volume.shape} in {mri_path}")
                return None
            slice_2d = volume[:, index, :]
            # Optional: Transpose if needed, e.g., slice_2d = slice_2d.T
        elif view == 'axial':
            if index < 0 or index >= volume.shape[2]:
                # print(f"Error: Axial index {index} out of bounds for shape {volume.shape} in {mri_path}")
                return None
            slice_2d = volume[:, :, index]
            # Optional: Transpose if needed, e.g., slice_2d = slice_2d.T
        else:
            print(f"Error: Unknown view '{view}' for {mri_path}")
            return None

        # Ensure float32 for consistency before normalization
        # Rotate to make orientation consistent if necessary (e.g., axial slices often need rotation)
        # This might need adjustment based on your specific data orientation needs
        if view == 'axial' or view == 'coronal' or view == 'sagittal': # Apply rotation/transpose consistently if needed
             slice_2d = np.rot90(slice_2d) # Example: Rotate 90 degrees

        return slice_2d.astype(np.float32)

    except Exception as e:
        print(f"Error loading or processing {mri_path} for slice {view}_{index}: {e}")
        return None

def preprocess_slice(slice_2d, device):
    """
    Preprocesses a 2D grayscale NumPy slice for MedSAM's image encoder.
    Matches the preprocessing in MedSAM's demo.py.

    Args:
        slice_2d (np.ndarray): The 2D grayscale slice (H, W).
        device (torch.device or str): The device to send the tensor to.

    Returns:
        torch.Tensor: The preprocessed tensor (1, 3, 1024, 1024).
    """
    if slice_2d is None or slice_2d.size == 0:
        # print("Error: Input slice is empty or None.")
        return None

    # 1. Normalize the slice to [0, 1]
    min_val = slice_2d.min()
    max_val = slice_2d.max()
    denominator = max_val - min_val
    slice_norm = (slice_2d - min_val) / np.clip(denominator, a_min=1e-8, a_max=None)

    # 2. Convert to 3 channels by stacking
    if slice_norm.ndim != 2:
        # print(f"Error: Expected 2D slice for stacking, got {slice_norm.ndim} dimensions.")
        return None
    slice_3c = np.stack([slice_norm] * 3, axis=-1) # Shape: (H, W, 3)

    # 3. Resize to 1024x1024
    try:
        if slice_3c.dtype != np.float32 and slice_3c.dtype != np.uint8:
             slice_3c = slice_3c.astype(np.float32)
        img_resize = cv2.resize(
            slice_3c,
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC
        ) # Shape: (1024, 1024, 3)
    except Exception as e:
        print(f"Error during resizing: {e}")
        return None

    img_resize = np.clip(img_resize, 0, 1)

    # 4. Convert to Tensor, permute, add batch dim, move to device
    img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1) # HWC to CHW
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    img_tensor = img_tensor.to(device)

    return img_tensor

def extract_and_save_features(mri_path, model, feature_extractor, output_base_dir, target_slices, device):
    """
    Loads an MRI, preprocesses specified slices, extracts features for each,
    and saves them into a subject-specific directory.

    Args:
        mri_path (str or Path): Path to the input MRI file.
        model (torch.nn.Module): The loaded MedSAM model.
        feature_extractor (EncoderFeatureExtractor): The hook object.
        output_base_dir (str or Path): Base directory for the split (e.g., .../BrainAGE_preprocessed/train).
        target_slices (dict): Dictionary defining slices to process, e.g., {'sagittal': [80, 125], ...}.
        device (str or torch.device): The computation device.
    """
    mri_path = Path(mri_path)
    output_base_dir = Path(output_base_dir)
    # Create subject-specific directory: use filename without extension as folder name
    subject_id = mri_path.name.replace(".nii.gz", "").replace(".nii", "") # Handle multiple extensions
    subject_output_dir = output_base_dir / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    # print(f"Processing: {mri_path.name} -> {subject_output_dir}") # Optional

    for view, indices in target_slices.items():
        for index in indices:
            output_filename = subject_output_dir / f"slice_{view}_{index}.npy"

            # Skip if feature file for this specific slice already exists
            if output_filename.exists():
                # print(f"Skipping slice {view}_{index} for {mri_path.name}, file exists.") # Optional
                continue

            # 1. Load the specific slice
            slice_data = load_specific_slice(mri_path, view, index)
            if slice_data is None:
                # print(f"Skipping slice {view}_{index} for {mri_path.name} due to loading error.") # Optional
                continue # Skip to the next slice index

            # 2. Preprocess the slice
            preprocessed_tensor = preprocess_slice(slice_data, device)
            if preprocessed_tensor is None:
                # print(f"Skipping slice {view}_{index} for {mri_path.name} due to preprocessing error.") # Optional
                continue # Skip to the next slice index

            # 3. Clear previous features and run forward pass
            feature_extractor.clear()
            model.eval()
            with torch.no_grad():
                try:
                    _ = model.image_encoder(preprocessed_tensor)
                except Exception as e:
                    print(f"Error during model forward pass for {mri_path.name}, slice {view}_{index}: {e}")
                    # Clean up tensor before potentially skipping
                    del preprocessed_tensor
                    gc.collect()
                    if device.startswith('cuda'): torch.cuda.empty_cache()
                    continue # Skip to the next slice index

            # 4. Get features from the hook
            encoder_features = feature_extractor.features
            if encoder_features is None:
                # print(f"Failed to extract features for {mri_path.name}, slice {view}_{index}.") # Optional
                # Clean up tensor
                del preprocessed_tensor
                gc.collect()
                if device.startswith('cuda'): torch.cuda.empty_cache()
                continue # Skip to the next slice index

            # 5. Save the features
            try:
                np.save(output_filename, encoder_features.numpy())
                # print(f"Saved features for slice {view}_{index} to: {output_filename}") # Optional
            except Exception as e:
                print(f"Error saving features for {mri_path.name}, slice {view}_{index} to {output_filename}: {e}")

            # Clean up tensor and cache after processing each slice
            del preprocessed_tensor, encoder_features, slice_data
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    INPUT_DATA_DIR = Path("/data/kuang/Projects/MedSAM/data/BrainAGE")
    # Adjust output directory name to reflect multi-slice nature if desired
    OUTPUT_BASE_DIR = Path("/data/kuang/Projects/MedSAM/data/BrainAGE_preprocessed_multi_slice")
    MEDSAM_CKPT_PATH = "medsam_vit_b.pth"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MRI_EXTENSIONS = ['.nii', '.nii.gz']
    SPLITS = ["train", "test", "validation"]

    # Define the target slices for each view
    TARGET_SLICES = {
        'sagittal': [80, 125],
        'coronal': [125],
        'axial': [80]
        # Add more slices/views as needed
    }
    # --- /Configuration ---

    print(f"Using device: {DEVICE}")
    print(f"Input data directory: {INPUT_DATA_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Target slices: {TARGET_SLICES}")

    if not INPUT_DATA_DIR.is_dir():
        print(f"Error: Input data directory not found at {INPUT_DATA_DIR}")
        exit()
    if not os.path.exists(MEDSAM_CKPT_PATH):
        print(f"Error: MedSAM checkpoint not found at {MEDSAM_CKPT_PATH}")
        exit()

    # --- Collect all files and prepare output directories ---
    all_mri_files = []
    output_dirs = {} # Maps split name to its base output directory
    print("Scanning for files...")
    for split in SPLITS:
        split_input_dir = INPUT_DATA_DIR / split
        split_output_dir = OUTPUT_BASE_DIR / split
        output_dirs[split] = split_output_dir

        if not split_input_dir.is_dir():
            print(f"Warning: Input directory for split '{split}' not found: {split_input_dir}")
            continue

        # Only create the base output directory for the split here
        split_output_dir.mkdir(parents=True, exist_ok=True)

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

    # --- Load Model and Set up Hook ---
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

    # --- Process all files ---
    print("\nStarting feature extraction for specified slices...")
    for mri_file_path in tqdm(all_mri_files, desc="Processing MRI files"):
        try:
            # Determine the base output directory for the file's split
            split_name = mri_file_path.parent.name
            if split_name not in output_dirs:
                 print(f"Warning: Could not determine split for {mri_file_path}. Skipping.")
                 continue
            correct_output_base_dir = output_dirs[split_name]
        except IndexError:
             print(f"Warning: Could not determine split for {mri_file_path}. Skipping.")
             continue

        # Call the function to process all target slices for this MRI file
        extract_and_save_features(
            mri_path=mri_file_path,
            model=medsam_model,
            feature_extractor=feature_extractor,
            output_base_dir=correct_output_base_dir, # Pass the split's base output dir
            target_slices=TARGET_SLICES,
            device=DEVICE
        )
        # Optional: Add explicit garbage collection after each file if memory issues persist
        # gc.collect()
        # if DEVICE.startswith('cuda'): torch.cuda.empty_cache()


    # --- Cleanup ---
    hook_handle.remove()
    print("\nHook removed.")
    # Explicitly delete model and clear cache
    del medsam_model, image_encoder, feature_extractor
    gc.collect()
    if DEVICE.startswith('cuda'):
        torch.cuda.empty_cache()
    print("Multi-slice preprocessing finished.")
