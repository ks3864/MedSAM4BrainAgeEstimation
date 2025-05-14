import torch
import numpy as np
import cv2
import nibabel as nib
from segment_anything import sam_model_registry
import os
from pathlib import Path
from tqdm import tqdm

# Define the feature extraction hook class
class EncoderFeatureExtractor:
    def __init__(self):
        self.features = None

    def hook(self, module, input, output):
        # Detach the tensor to prevent gradient tracking and move to CPU
        self.features = output.detach().cpu()

    def clear(self):
        self.features = None

def load_middle_axial_slice(mri_path):
    """
    Loads the middle axial slice of a 3D MRI volume.
    Args:
        mri_path (str or Path): Path to the MRI file.
    Returns:
        np.ndarray: The middle axial slice (2D). Returns None if loading fails.
    """
    try:
        img = nib.load(mri_path)
        volume = img.get_fdata()
        if volume.ndim != 3:
            # print(f"Error: Expected 3D volume, but got {volume.ndim} dimensions for {mri_path}") # Optional: reduce verbosity
            return None
        # Ensure the slice index is valid
        middle_index = volume.shape[2] // 2
        if middle_index < 0 or middle_index >= volume.shape[2]:
             # print(f"Error: Invalid middle slice index {middle_index} for shape {volume.shape} in {mri_path}") # Optional: reduce verbosity
             return None
        slice_2d = volume[:, :, middle_index]  # shape [H, W]
        # Ensure float32 for consistency before normalization
        return slice_2d.astype(np.float32)
    except Exception as e:
        print(f"Error loading or processing {mri_path}: {e}")
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
        # print("Error: Input slice is empty or None.") # Optional: reduce verbosity
        return None

    # 1. Normalize the slice to [0, 1]
    min_val = slice_2d.min()
    max_val = slice_2d.max()
    denominator = max_val - min_val
    # Use a small epsilon to avoid division by zero for constant slices
    slice_norm = (slice_2d - min_val) / np.clip(denominator, a_min=1e-8, a_max=None)

    # 2. Convert to 3 channels by stacking
    # Ensure the input is 2D before stacking
    if slice_norm.ndim != 2:
        # print(f"Error: Expected 2D slice for stacking, got {slice_norm.ndim} dimensions.") # Optional: reduce verbosity
        return None
    slice_3c = np.stack([slice_norm] * 3, axis=-1) # Shape: (H, W, 3)

    # 3. Resize to 1024x1024
    try:
        # Ensure input to resize is float32 or uint8
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

    # Ensure it's still [0, 1] after resize (interpolation might slightly change values)
    img_resize = np.clip(img_resize, 0, 1)

    # 4. Convert to Tensor, permute, add batch dim, move to device
    # HWC to CHW -> (3, 1024, 1024)
    img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1)
    # Add batch dimension -> (1, 3, 1024, 1024)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    return img_tensor

def extract_and_save_features(mri_path, model, feature_extractor, output_dir, device):
    """
    Loads an MRI, preprocesses the middle slice, extracts features using the model's
    encoder, and saves the features as a .npy file.

    Args:
        mri_path (str or Path): Path to the input MRI file.
        model (torch.nn.Module): The loaded MedSAM model.
        feature_extractor (EncoderFeatureExtractor): The hook object.
        output_dir (str or Path): Directory to save the .npy feature file.
        device (str or torch.device): The computation device.
    """
    mri_path = Path(mri_path)
    output_dir = Path(output_dir)
    # Ensure the specific output directory for the file exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use mri_path.name to keep original extensions and append .npy
    output_filename = output_dir / (mri_path.name + ".npy")

    # Skip if feature file already exists
    if output_filename.exists():
        # print(f"Skipping {mri_path.name}, feature file already exists: {output_filename}") # Optional: reduce verbosity
        return

    # print(f"Processing: {mri_path.name}") # Optional: print for every file

    # 1. Load middle axial slice
    axial_slice = load_middle_axial_slice(mri_path)
    if axial_slice is None:
        # print(f"Skipping {mri_path.name} due to loading/slicing error.") # Optional: reduce verbosity
        return

    # 2. Preprocess the slice
    preprocessed_tensor = preprocess_slice(axial_slice, device)
    if preprocessed_tensor is None:
        # print(f"Skipping {mri_path.name} due to preprocessing error.") # Optional: reduce verbosity
        return

    # 3. Clear previous features and run forward pass to trigger hook
    feature_extractor.clear()
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        try:
            _ = model.image_encoder(preprocessed_tensor)
        except Exception as e:
            print(f"Error during model forward pass for {mri_path.name}: {e}")
            return

    # 4. Get features from the hook
    encoder_features = feature_extractor.features
    if encoder_features is None:
        # print(f"Failed to extract features for {mri_path.name}.") # Optional: reduce verbosity
        return

    # 5. Save the features as .npy
    # Features are already on CPU due to hook implementation
    try:
        np.save(output_filename, encoder_features.numpy())
        # print(f"Saved features to: {output_filename}") # Optional: print for every file
        # print(f"Feature shape: {encoder_features.shape}") # Optional: print for every file
    except Exception as e:
        print(f"Error saving features for {mri_path.name} to {output_filename}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_DATA_DIR = Path("/data/kuang/Projects/MedSAM/data/BrainAGE")
    OUTPUT_BASE_DIR = Path("/data/kuang/Projects/MedSAM/data/BrainAGE_preprocessed")
    MEDSAM_CKPT_PATH = "medsam_vit_b.pth" # Path to your MedSAM checkpoint
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MRI_EXTENSIONS = ['.nii', '.nii.gz'] # Add other extensions if needed
    SPLITS = ["train", "test", "validation"]
    # --- /Configuration ---

    print(f"Using device: {DEVICE}")
    print(f"Input data directory: {INPUT_DATA_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")

    # Check if input directory exists
    if not INPUT_DATA_DIR.is_dir():
        print(f"Error: Input data directory not found at {INPUT_DATA_DIR}")
        exit()

    # Check if checkpoint exists
    if not os.path.exists(MEDSAM_CKPT_PATH):
        print(f"Error: MedSAM checkpoint not found at {MEDSAM_CKPT_PATH}")
        exit()

    # --- Collect all files and prepare output directories ---
    all_mri_files = []
    output_dirs = {}
    print("Scanning for files...")
    for split in SPLITS:
        split_input_dir = INPUT_DATA_DIR / split
        split_output_dir = OUTPUT_BASE_DIR / split
        output_dirs[split] = split_output_dir # Store output path for the split

        if not split_input_dir.is_dir():
            print(f"Warning: Input directory for split '{split}' not found: {split_input_dir}")
            continue

        # Create the output directory for the split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Find MRI files for the current split
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

    # --- Process all files with a single progress bar ---
    print("\nStarting feature extraction...")
    for mri_file_path in tqdm(all_mri_files, desc="Processing dataset"):
        # Determine the split and corresponding output directory
        try:
            # Assumes path structure like .../BrainAGE/train/file.nii.gz
            split_name = mri_file_path.parent.name
            if split_name not in output_dirs:
                 print(f"Warning: Could not determine split for {mri_file_path}. Skipping.")
                 continue
            correct_output_dir = output_dirs[split_name]
        except IndexError:
             print(f"Warning: Could not determine split for {mri_file_path}. Skipping.")
             continue

        extract_and_save_features(
            mri_path=mri_file_path,
            model=medsam_model,
            feature_extractor=feature_extractor,
            output_dir=correct_output_dir, # Use the determined output dir
            device=DEVICE
        )

    # --- Cleanup ---
    hook_handle.remove()
    print("\nHook removed.")
    print("Batch processing finished.")