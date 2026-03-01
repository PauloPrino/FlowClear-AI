import os
import re
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- IMPORTS ---
try:
    import ModelArchitectures.UNET_2D as UNET_2D
    import ModelArchitectures.UNET_3D as UNET_3D
    import ModelArchitectures.UNET_2D_Berhane as UNET_2D_Berhane
    import ModelArchitectures.UNET_2D_T_group_norm as UNET_2D_T_group_norm
    import ModelArchitectures.UNET_3D_Berhane as UNET_3D_Berhane
except ImportError:
    print("WARNING: ModelArchitectures not found.")

try:
    import ROISegmentation
except ImportError as e:
    print(f"WARNING: ROISegmentation could not be imported. Detail: {e}")

# --- CONSTANTS ---
GLOBAL_MAX_PHASE = 4096.0

# --- MODEL REGISTRY ---
MODEL_REGISTRY = {
    "UNET_2D": {
        "class": UNET_2D.UNET_2D if 'UNET_2D' in locals() else None,
        "weights": "Weights/UNET_2D.pth",
        "params": {"in_channels": 1, "out_channels": 1, "init_features": 16, "dropout_rate": 0.1}
    },
    "UNET_2D_Berhane": {
        "class": UNET_2D_Berhane.UNET_2D_Berhane if 'UNET_2D_Berhane' in locals() else None,
        "weights": "Weights/UNET_2D_Berhane.pth",
        "params": {"in_channels": 1, "out_channels": 1, "growth_rate": 12, "dropout_rate": 0.1}
    },
    "UNET_3D": {
        "class": UNET_3D.UNET_3D if 'UNET_3D' in locals() else None,
        "weights": "Weights/UNET_3D.pth",
        "params": {"in_channels": 1, "out_channels": 1}
    },
    "UNET_3D_Berhane": {
        "class": UNET_3D_Berhane.UNET_3D_Berhane if 'UNET_3D_Berhane' in locals() else None,
        "weights": "Weights/UNET_3D_Berhane.pth",
        "params": {"in_channels": 1, "out_channels": 1}
    },
    "UNET_2D_T_SAG": {
        "class": UNET_2D_T_group_norm.UNET_2D_T_group_norm if 'UNET_2D_T_group_norm' in locals() else None,
        "weights": "Weights/UNET_2D_T_SAG.pth",
        "params": {"in_channels": 1, "out_channels": 1, "init_features": 40, "dropout_rate": 0.1}
    },
    "UNET_2D_T_COR": {
        "class": UNET_2D_T_group_norm.UNET_2D_T_group_norm if 'UNET_2D_T_group_norm' in locals() else None,
        "weights": "Weights/UNET_2D_T_COR.pth",
        "params": {"in_channels": 1, "out_channels": 1, "init_features": 40, "dropout_rate": 0.1}
    },
    "UNET_2D_T_AX": {
        "class": UNET_2D_T_group_norm.UNET_2D_T_group_norm if 'UNET_2D_T_group_norm' in locals() else None,
        "weights": "Weights/UNET_2D_T_AX.pth",
        "params": {"in_channels": 1, "out_channels": 1, "init_features": 40, "dropout_rate": 0.1}
    }
}

def extract_venc_ratio_from_filename(filename):
    match = re.search(r"venc_ratio([\d\.]+)", filename)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def get_venc_from_json(nifti_path, log_callback=None):
    path_obj = Path(nifti_path)
    json_path = str(path_obj).replace(".nii.gz", "_dicom_header.json")
    if not os.path.exists(json_path):
        base_name = path_obj.stem.split("_e")[0]
        if "_" in base_name and base_name[-1].isdigit(): 
             base_name = "_".join(base_name.split("_")[:-1])
        parent_json = path_obj.parent / (base_name + "_dicom_header.json")
        if parent_json.exists():
            json_path = str(parent_json)

    if not os.path.exists(json_path):
        if log_callback: log_callback("Header JSON not found.")
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        venc_raw = None
        for key, value in data.items():
            if key == "[Velocity encoding]":
                venc_raw = float(value["value"])
                break
        
        if venc_raw is None:
            if log_callback: log_callback("VENC key missing in JSON.")
            return None
            
        venc_cm_s = venc_raw / 10.0
        if log_callback: log_callback(f"Found VENC in JSON: {venc_cm_s}")
        return venc_cm_s

    except Exception as e:
        if log_callback: log_callback(f"Error reading JSON: {e}.")
        return None

def resolve_files(path, log_callback=None):
    files_to_process = []
    path_obj = Path(path)
    if path_obj.is_file(): return [str(path_obj)]
    elif path_obj.is_dir():
        if log_callback: log_callback(f"Scanning folder: {path_obj.name}...")
        basename = path_obj.name
        candidates = {
            "LR": path_obj / f"{basename}_e2.nii.gz",
            "AP": path_obj / f"{basename}_e3.nii.gz",
            "SI": path_obj / f"{basename}_e4.nii.gz"
        }
        if not candidates["LR"].exists():
            all_nii = list(path_obj.glob("*.nii.gz"))
            for f in all_nii:
                parts = f.name.replace(".nii.gz", "").split("_")
                if not parts: continue
                suffix = parts[-1]
                if suffix.endswith("1"): candidates["LR"] = f
                elif suffix.endswith("2"): candidates["AP"] = f
                elif suffix.endswith("3"): candidates["SI"] = f
        for flow_type, fpath in candidates.items():
            if fpath.exists():
                files_to_process.append(str(fpath))
                if log_callback: log_callback(f"Found {flow_type}: {fpath.name}")
        if not files_to_process: raise FileNotFoundError("No valid flow files found.")
        return files_to_process
    return []

def load_nifti_data(file_path):
    img = nib.load(file_path)
    data = np.asarray(img.dataobj, dtype=np.int16)
    return data, img.affine

def find_magnitude_file(flow_file_path):
    path = Path(flow_file_path)
    directory = path.parent
    all_niftis = list(directory.glob("*.nii.gz"))
    candidates = []
    for f in all_niftis:
        if f.resolve() == path.resolve(): continue
        fname = f.name.lower()
        if "mask" in fname or "seg" in fname: continue
        candidates.append(f)
    for c in candidates:
        if c.name.endswith("0.nii.gz"): return str(c)
    for c in candidates:
        if "_e1.nii.gz" in c.name: return str(c)
    for c in candidates:
        name = c.name
        if "_e2" not in name and "_e3" not in name and "_e4" not in name: return str(c)
    return None

def ensure_segmentation(magnitude_path, log_callback=None):
    if 'ROISegmentation' not in globals():
        if log_callback: log_callback("Error: ROISegmentation module failed to load.")
        return None
    if log_callback: log_callback(f"Using Magnitude File: {Path(magnitude_path).name}")
    base_dir = Path(magnitude_path).parent
    patient_name = Path(magnitude_path).name.replace(".nii.gz", "").replace("_e1", "")
    if patient_name.endswith("_"): patient_name = patient_name[:-1]
    seg_dir = base_dir.parent / "MRSegmentations"
    seg_dir.mkdir(exist_ok=True)
    aorta_mask_path = seg_dir / f"{patient_name}_aorta_mask_4d.nii.gz"
    heart_mask_path = seg_dir / f"{patient_name}_heart_mask_4d.nii.gz"
    combined_mask_path = seg_dir / f"{patient_name}_combined_ah_mask_4d.nii.gz"
    if not (aorta_mask_path.exists() and heart_mask_path.exists()):
        if log_callback: log_callback("Running Segmentation (Aorta + Heart)...")
        try:
            ROISegmentation.segment_4d_with_mrsegmentator(
                nifti_path=magnitude_path, output_dir=str(seg_dir), rois=["aorta", "heart"], fast=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        except Exception as e:
            if log_callback: log_callback(f"Segmentation failed: {e}")
            return None
    try:
        img_aorta = nib.load(aorta_mask_path)
        img_heart = nib.load(heart_mask_path)
        combined_data = np.logical_or(img_aorta.get_fdata() > 0, img_heart.get_fdata() > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(combined_data, img_aorta.affine), combined_mask_path)
        return str(combined_mask_path)
    except Exception as e:
        if log_callback: log_callback(f"Error combining masks: {e}")
        return None

def load_model(model_key, device, log_callback=None):
    if model_key not in MODEL_REGISTRY: raise ValueError(f"Model {model_key} not found.")
    entry = MODEL_REGISTRY[model_key]

    script_dir = Path(__file__).parent
    weights_path = script_dir / entry["weights"]
    
    # Instantiate the model architecture
    try: model = entry["class"](**entry.get("params", {}))
    except: model = entry["class"]()
    
    if os.path.exists(weights_path):
        if log_callback: log_callback(f"Loading weights: {weights_path.name}")
        
        # Load the raw state dict
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        
        # --- FIX START: Clean state_dict keys ---
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            # Remove '_orig_mod.' prefix (from torch.compile)
            if name.startswith("_orig_mod."):
                name = name[10:] 
            # Remove 'module.' prefix (from DataParallel/DistributedDataParallel)
            elif name.startswith("module."):
                name = name[7:]
            new_state_dict[name] = v
        # --- FIX END ---

        # Load the cleaned dictionary
        model.load_state_dict(new_state_dict)
    else:
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    model.to(device)
    model.eval()
    return model

def predict_volume_batches(model, volume_data, device, batch_size=32, return_probs=False, threshold=0.5):
    volume_norm = volume_data.astype(np.float32) / GLOBAL_MAX_PHASE
    
    # Check model type to determine handling of dimensions
    model_name = model.__class__.__name__
    
    # --- TEMPORAL / 3D MODELS (2D+T or 3D) ---
    # These models need the Depth/Time dimension intact. 
    # Input shape required: (Batch=1, Channel=1, Depth/Time, Height, Width)
    if "2D_T" in model_name or "3D" in model_name:
        # Original: (H, W, T) -> Permute to (T, H, W)
        volume_tensor = torch.from_numpy(volume_norm).permute(2, 0, 1)
        
        # Add Batch and Channel dimensions: (1, 1, T, H, W)
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        
        # Padding (H, W only)
        _, _, T, H, W = volume_tensor.shape
        pad_h = max(0, 256 - H)
        pad_w = max(0, 256 - W)
        if pad_h > 0 or pad_w > 0:
            volume_tensor = torch.nn.functional.pad(volume_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
        with torch.no_grad():
            inputs = volume_tensor.to(device)
            out = model(inputs)
            probs = torch.sigmoid(out).cpu()
            
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            probs = probs[:, :, :, :H, :W]
            
        # Reshape back to (H, W, T) -> (1, 1, T, H, W) -> (T, H, W) -> (H, W, T)
        full_probs = probs.squeeze(0).squeeze(0).permute(1, 2, 0).numpy()
        
        if return_probs:
            return full_probs
        else:
            return (full_probs > threshold).astype(np.uint8)

    # --- STANDARD 2D MODELS ---
    # These treat T as independent batch samples.
    # Input shape required: (Batch=N, Channel=1, Height, Width)
    else:
        # (H, W, T) -> (T, H, W) -> (T, 1, H, W)
        volume_tensor = torch.from_numpy(volume_norm).permute(2, 0, 1).unsqueeze(1) 
        
        _, _, H, W = volume_tensor.shape
        pad_h = max(0, 256 - H)
        pad_w = max(0, 256 - W)
        if pad_h > 0 or pad_w > 0:
            volume_tensor = torch.nn.functional.pad(volume_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        dataset = TensorDataset(volume_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                out = model(inputs)
                probs = torch.sigmoid(out).cpu()
                preds.append(probs)
        
        full_probs = torch.cat(preds, dim=0)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            full_probs = full_probs[:, :, :H, :W]
            
        # (T, H, W) -> (H, W, T)
        full_probs = full_probs.squeeze(1).permute(1, 2, 0).numpy()
        
        if return_probs:
            return full_probs
        else:
            return (full_probs > threshold).astype(np.uint8)

def predict_ensemble_slice(models, volume_data, device, strategy="Average", threshold=0.5):
    """
    Runs prediction using multiple models and combines them.
    volume_data: (H, W, T) numpy array
    models: list of loaded PyTorch models
    strategy: "Average", "Majority", "Unanimous"
    """
    all_probs = []
    
    # 1. Collect probabilities from all models
    for model in models:
        probs = predict_volume_batches(model, volume_data, device, return_probs=True)
        all_probs.append(probs)
        
    if not all_probs:
        return np.zeros_like(volume_data, dtype=np.uint8)
        
    # Stack: (N_models, H, W, T)
    stack = np.stack(all_probs, axis=0)
    
    # 2. Apply Strategy
    if strategy == "Average":
        avg_prob = np.mean(stack, axis=0)
        final_mask = (avg_prob > threshold).astype(np.uint8)
        
    elif strategy == "Majority":
        # Vote count (threshold each model at threshold)
        votes = np.sum(stack > threshold, axis=0)
        # Majority means >= N/2 (or > N/2). Let's use >= ceil(N/2)
        vote_threshold = len(models) / 2.0
        final_mask = (votes > vote_threshold).astype(np.uint8)
        
    elif strategy == "Unanimous":
        votes = np.sum(stack > threshold, axis=0)
        final_mask = (votes == len(models)).astype(np.uint8)
        
    else:
        # Fallback to Average
        avg_prob = np.mean(stack, axis=0)
        final_mask = (avg_prob > threshold).astype(np.uint8)
        
    return final_mask

def unwrap_slice_inplace(original_data, mask, output_array, venc, venc_ratio):
    correction_value = 2.0 * venc * venc_ratio
    correction = mask * correction_value
    output_array[:] = np.where(
        (mask > 0) & (original_data < 0), original_data + correction,
        np.where(
            (mask > 0) & (original_data > 0), original_data - correction,
            original_data
        )
    )

def save_nifti_file(data, affine, path, is_mask=False):
    if is_mask:
        data_to_save = (data * 255).astype(np.uint8)
    else:
        data_to_save = data.astype(np.float32)
    nib.save(nib.Nifti1Image(data_to_save, affine), path)
    return str(path)

def generate_filenames(original_path):
    script_dir = Path(__file__).parent
    out_dir = script_dir / "Outputs"
    out_dir.mkdir(exist_ok=True)
    base = Path(original_path).stem.replace(".nii", "")
    mask_path = out_dir / f"{base}_MASK.nii.gz"
    unwrap_path = out_dir / f"{base}_UNWRAPPED.nii.gz"
    return str(mask_path), str(unwrap_path)

def transform_rotate_slice(slice_data, section_name):
    """
    Rotates a 2D slice for proper visualization based on anatomical plane.
    Matches logic in StudyCaseNIfTI.py
    """
    if section_name == "axial":
        return np.rot90(np.rot90(slice_data))
    if section_name == "coronal":
        # Coronal needs transpose + 3 rotations (or similar) to match standard view
        return np.rot90(np.rot90(np.rot90(slice_data.T)))
    if section_name == "sagittal":
        return slice_data.T
    return slice_data
