import os
import shutil
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path
import sys

# Import MRSegmentator inference module
from mrsegmentator import inference

# MRSegmentator Class Mapping (based on provided class_codes.py)
MRSEG_CLASSES = {
    "heart": 12,
    "aorta": 13,
    "liver": 5,
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "inferior_vena_cava": 14,
    # Add other classes from the documentation if needed
}

class DummyFile:
    def write(self, x): pass
    def flush(self): pass

def segment_4d_with_mrsegmentator(nifti_path, output_dir, rois=["aorta"], fast=True, device='cuda'):
    """
    Segments a 4D NIfTI file using MRSegmentator by processing each time frame.
    
    Args:
        nifti_path (str): Path to the 4D NIfTI file (Magnitude/Anatomy).
        output_dir (str): Folder to save the resulting 4D ROI masks.
        rois (list): List of ROIs to extract (e.g., ['aorta', 'heart']).
        fast (bool): If True, uses only Fold 0 (5x faster). If False, uses ensemble (Folds 0-4).
        device (str): 'cuda' for GPU or 'cpu'.
    
    Returns:
        dict: Paths to the generated 4D mask files for each ROI.
    """
    
    # Fix for PyInstaller --windowed mode where sys.stdout/stderr are None
    # This prevents 'NoneType' object has no attribute 'write' errors in external libs (tqdm, etc.)
    if sys.stdout is None: sys.stdout = DummyFile()
    if sys.stderr is None: sys.stderr = DummyFile()
    
    # 1. Setup paths and load data
    file_name = os.path.basename(nifti_path).replace(".nii.gz", "")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    
    # Check if 4D
    if data.ndim != 4:
        print(f"Input {nifti_path} is not 4D. Processing as 3D volume...")
        data = data[..., np.newaxis] # Add fake time dimension
        
    n_frames = data.shape[3]
    print(f"Processing {n_frames} time frames for {file_name}...")

    # 2. Create a temporary directory to store individual time frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_paths = []
        
        # Split 4D -> 3D files
        print("Splitting 4D volume into temporary 3D frames...")
        for t in range(n_frames):
            frame_data = data[..., t]
            frame_img = nib.Nifti1Image(frame_data, affine)
            frame_path = os.path.join(temp_dir, f"frame_{t:03d}.nii.gz")
            nib.save(frame_img, frame_path)
            frame_paths.append(frame_path)
            
        # 3. Run MRSegmentator Inference
        # We pass the list of all frames to infer() once to keep the model in memory
        print(f"Running MRSegmentator (Fast Mode: {fast})...")
        
        folds = [0] if fast else [0, 1, 2, 3, 4]
        
        # Run inference
        inference.infer(
            images=frame_paths,
            outdir=temp_dir,
            folds=folds,
            postfix="seg",
            batchsize=8,
            cpu_only=(device == 'cpu'),
            verbose=False,
            nproc=8
        )

        # 4. Reconstruct 4D Masks for requested ROIs
        os.makedirs(output_dir, exist_ok=True)
        generated_files = {}

        for roi in rois:
            if roi not in MRSEG_CLASSES:
                print(f"Warning: ROI '{roi}' not found in MRSegmentator classes. Skipping.")
                continue
            
            target_class_id = MRSEG_CLASSES[roi]
            print(f"Reconstructing 4D mask for {roi} (Class ID: {target_class_id})...")
            
            roi_mask_4d = np.zeros(data.shape, dtype=np.uint8)
            
            # Read back the segmented frames
            for t in range(n_frames):
                seg_path = os.path.join(temp_dir, f"frame_{t:03d}_seg.nii.gz")
                if os.path.exists(seg_path):
                    seg_img = nib.load(seg_path)
                    seg_data = seg_img.get_fdata()
                    
                    # Create binary mask for the specific ROI
                    roi_mask_4d[..., t] = (seg_data == target_class_id).astype(np.uint8)
                else:
                    print(f"Error: Segmentation for frame {t} not found.")

            # Save the final 4D mask
            output_filename = f"{file_name}_{roi}_mask_4d.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            
            final_img = nib.Nifti1Image(roi_mask_4d, affine)
            nib.save(final_img, output_path)
            generated_files[roi] = output_path
            print(f"Saved: {output_path}")

    return generated_files