import sys
import time
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from concurrent.futures import ThreadPoolExecutor

# GUI Imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame,
                               QComboBox, QMessageBox, QProgressBar, QCheckBox, 
                               QLineEdit, QFileDialog, QStackedWidget, QSpinBox, 
                               QProgressDialog, QScrollBar, QScrollArea, QGroupBox)
from PySide6.QtCore import Qt, QThread, Signal, QLocale, QTimer, QSize
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDoubleValidator, QCursor, QAction, QPixmap, QColor, QPainter, QFont

# Matplotlib Integration
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import Prediction as P
from DicomConverter import DicomConverter

# --- IMPORT CLASSICAL UNWRAPPING TOOLBOX ---
try:
    import unwrapFlow
except ImportError:
    print("WARNING: unwrapFlow.py not found. Classical methods will be unavailable.")
    unwrapFlow = None

# --- GLOBAL STYLES ---
STYLE_ZONE = """
    QFrame {
        border: 2px dashed #aaaaaa;
        border-radius: 15px;
        background-color: #f9f9f9;
    }
    QFrame:hover {
        background-color: #f0f0f0;
        border-color: #444;
    }
"""

# =============================================================================
# WORKER THREAD (PREDICTION)
# =============================================================================
class InferenceWorker(QThread):
    finished = Signal(str, str)
    error = Signal(str)
    progress_log = Signal(str)
    progress_bar = Signal(int)
    eta_update = Signal(str)
    task_update = Signal(str)
    device_status = Signal(str) 

    def __init__(self, input_path, model_config, do_segmentation, venc, venc_ratio, threshold=0.5):
        super().__init__()
        self.input_path = input_path
        self.model_config = model_config # {'mode': 'single'/'ensemble'/'classic', 'models': [], 'strategy': str}
        self.do_segmentation = do_segmentation
        self.venc = venc
        self.venc_ratio = venc_ratio
        self.threshold = threshold
        self.start_time = 0
        self._is_running = True
        self.executor = None

    def stop(self):
        """Signals the worker to stop processing."""
        self._is_running = False
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def run(self):
        try:
            self.start_time = time.time()
            
            # --- 1. DETERMINE MODE (AI vs CLASSIC vs ENSEMBLE) ---
            is_classic = self.model_config['mode'] == 'classic'
            is_ensemble = self.model_config['mode'] == 'ensemble'
            
            if is_classic:
                if unwrapFlow is None:
                    raise ImportError("unwrapFlow.py is missing. Cannot run classical methods.")
                dev_str = "Running on: CPU (Classical Algo) ⚙️"
                device = torch.device("cpu") # Classical algos run on CPU usually
            else:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    dev_str = "Running on: GPU (Fast) 🚀"
                else:
                    device = torch.device("cpu")
                    dev_str = "Running on: CPU (Slow) 🐢"

            self.device_status.emit(dev_str)
            self.progress_log.emit(f"Device: {device}")
            self.task_update.emit("Initializing...")

            # --- 1.5 AUTOMATIC DICOM CONVERSION ---
            if os.path.isdir(self.input_path):
                # Check if folder contains NIfTI files
                if not self._is_running: return
                has_nifti = any(f.endswith('.nii.gz') or f.endswith('.nii') for f in os.listdir(self.input_path))
                if not has_nifti:
                    self.task_update.emit("Detected DICOM. Converting...")
                    self.progress_log.emit("Input appears to be DICOM. Starting automatic conversion...")
                    
                    converter = DicomConverter()
                    output_nifti_path = self.input_path + "_NIfTI"
                    success = converter.convert_folder(self.input_path, output_nifti_path)
                    
                    if success:
                        self.progress_log.emit(f"Conversion successful. Using: {output_nifti_path}")
                        self.input_path = output_nifti_path
                    else:
                        self.progress_log.emit("Conversion failed or dcm2niix not found. Attempting to proceed with original path...")

            # --- 2. FILE RESOLUTION ---
            file_list = P.resolve_files(self.input_path, self.progress_log.emit)
            total_files = len(file_list)

            # --- 3. MODEL LOADING (Only for AI) ---
            model = None
            ensemble_models = []
            if not self._is_running: return
            
            if not is_classic:
                if is_ensemble:
                    self.task_update.emit(f"Loading {len(self.model_config['models'])} Ensemble Models...")
                    for key in self.model_config['models']:
                        try:
                            m = P.load_model(key, device, self.progress_log.emit)
                            ensemble_models.append(m)
                        except Exception as e:
                            self.progress_log.emit(f"Warning: Could not load {key} for ensemble: {e}")
                    
                    if not ensemble_models:
                        raise ValueError("No models could be loaded for the ensemble.")
                    self.progress_log.emit(f"Ensemble loaded with {len(ensemble_models)} models.")
                else:
                    self.task_update.emit("Loading AI Model...")
                    model = P.load_model(self.model_config['models'][0], device, self.progress_log.emit)

            # --- 4. SEGMENTATION (Optional) ---
            roi_mask_data = None
            if self.do_segmentation:
                if not self._is_running: return
                self.task_update.emit("Running Segmentation (MRSegmentator)...")
                mag_path = P.find_magnitude_file(file_list[0])
                if mag_path:
                    mask_path = P.ensure_segmentation(mag_path, self.progress_log.emit)
                    if mask_path:
                        roi_mask_img = nib.load(mask_path)
                        roi_mask_data = roi_mask_img.get_fdata().astype(np.uint8)
                        self.progress_log.emit("Combined ROI Mask (Aorta + Heart) loaded.")
                else:
                    self.progress_log.emit("Magnitude file not found. Skipping segmentation.")

            last_mask = ""
            last_unwrap = ""
            
            # Executor for AI Slice Unwrapping (Not used for Classic which processes volumes)
            self.executor = ThreadPoolExecutor(max_workers=4) 
            futures = []

            # --- 5. MAIN PROCESSING LOOP ---
            for i, fpath in enumerate(file_list):
                if not self._is_running: break
                fname = Path(fpath).name
                self.progress_log.emit(f"Processing ({i+1}/{total_files}): {fname}")
                self.task_update.emit(f"Processing file {i+1}/{total_files}")
                
                current_venc_ratio = self.venc_ratio 
                data, affine = P.load_nifti_data(fpath)
                
                # Prepare ROI for this file
                roi_for_file = None
                if roi_mask_data is not None:
                    # Handle dimension matching for ROI
                    if roi_mask_data.shape == data.shape:
                        roi_for_file = roi_mask_data
                    elif roi_mask_data.ndim == 3 and data.ndim == 4:
                        roi_for_file = roi_mask_data[..., np.newaxis]
                
                full_mask = np.zeros_like(data, dtype=np.uint8)
                full_unwrapped = np.zeros_like(data, dtype=np.float32)

                # ==========================================
                # BRANCH A: CLASSICAL UNWRAPPING
                # ==========================================
                if is_classic:
                    # Map UI string to unwrapFlow mode string
                    mode_map = {
                        "Classic: Laplacian 3D": "lap3D",
                        "Classic: Laplacian 4D": "lap4D"
                    }
                    algo_mode = mode_map.get(self.model_config['models'][0], "lap3D")
                    
                    # Validation
                    if "4D" in algo_mode and data.ndim != 4:
                        raise ValueError(f"You selected {algo_mode} but the input file is 3D. Please select a 3D mode.")
                    
                    self.task_update.emit(f"Running {algo_mode}...")
                    if not self._is_running: break
                    
                    # 1. Normalize Data to [-pi, pi] for unwrapFlow
                    # Assuming 12-bit signed range or similar. P.predict uses 4096.0
                    GLOBAL_MAX_PHASE = 4096.0 
                    img_radians = (data.astype(np.float32) / GLOBAL_MAX_PHASE) * np.pi

                    # 2. Apply ROI Mask if available (Pass to unwrapFlow)
                    # unwrapFlow.unwrap_data takes a 'mask' argument
                    # We need to ensure mask is binary 0/1 matching shape
                    algo_mask = None
                    if roi_for_file is not None:
                        # Ensure broadcast match
                        if roi_for_file.shape != img_radians.shape:
                             # Simple broadcast if ROI is 3D and Img is 4D
                             if roi_for_file.ndim == 3 and img_radians.ndim == 4:
                                 algo_mask = np.repeat(roi_for_file[...,np.newaxis], img_radians.shape[3], axis=3)
                             else:
                                 algo_mask = roi_for_file
                        else:
                             algo_mask = roi_for_file
                        # Convert to boolean/binary for unwrapFlow logic
                        algo_mask = (algo_mask > 0).astype(np.float32)

                    # 3. Run Algorithm
                    # full=True returns (phi_unwrapped, nr_wraps)
                    self.progress_log.emit("Starting algorithm (this may take time)...")
                    phi_u, nr = unwrapFlow.unwrap_data(
                        img_radians, 
                        mode=algo_mode, 
                        mask=algo_mask, 
                        full=True, 
                        verbose=False # We handle progress via worker signals manually usually, but unwrapFlow has internal tqdm
                    )
                    
                    # 4. Process Outputs
                    # Create Mask: Where number of wraps is non-zero
                    full_mask = (np.abs(nr) > 0).astype(np.uint8)
                    
                    # Create Unwrapped: Convert radians back to original intensity scale
                    full_unwrapped = (phi_u / np.pi) * GLOBAL_MAX_PHASE
                    
                    self._update_progress(i + 1, total_files)

                # ==========================================
                # BRANCH B: AI PREDICTION (Single or Ensemble)
                # ==========================================
                else: 
                    # Initialize Output Arrays
                    full_unwrapped = np.zeros_like(data, dtype=data.dtype)

                    # 4D Processing (Volume + Time)
                    if data.ndim == 4:
                        X, Y, Z, T = data.shape
                        
                        # Determine if we iterate slices (2D+T) or frames (3D)
                        # For Ensemble, we assume 2D+T logic as per current models
                        is_temporal_model = True
                        if not is_ensemble:
                            is_temporal_model = "2D_T" in model.__class__.__name__
                        
                        if is_temporal_model: # Iterate Slices
                            self.progress_log.emit(f"Mode: 2D+T (Iterating over {Z} slices)")
                            for z in range(Z):
                                if not self._is_running: break
                                self.task_update.emit(f"File {i+1}/{total_files} | Processing Slice {z+1}/{Z}")
                                slice_time_data = data[:, :, z, :]
                                
                                if roi_for_file is not None:
                                    roi_slice = roi_for_file[:, :, z, 0] if roi_for_file.ndim == 4 else roi_for_file[:, :, z]
                                    input_t = slice_time_data * roi_slice[..., np.newaxis]
                                else:
                                    input_t = slice_time_data

                                if is_ensemble:
                                    strategy = self.model_config['strategy']
                                    mask_t = P.predict_ensemble_slice(ensemble_models, input_t, device, strategy, threshold=self.threshold)
                                else:
                                    mask_t = P.predict_volume_batches(model, input_t, device, threshold=self.threshold)
                                
                                full_mask[:, :, z, :] = mask_t
                                f = self.executor.submit(P.unwrap_slice_inplace, slice_time_data, mask_t, full_unwrapped[:, :, z, :], self.venc, current_venc_ratio)
                                futures.append(f)
                                self._update_progress(i + (z + 1) / Z, total_files)
                                
                        else: # Iterate Time Frames (Standard 2D or 3D models)
                            self.progress_log.emit(f"Mode: Standard (Iterating over {T} time frames)")
                            for t in range(T):
                                if not self._is_running: break
                                self.task_update.emit(f"File {i+1}/{total_files} | Processing Frame {t+1}/{T}")
                                vol_t = data[..., t]
                                input_t = vol_t * (roi_for_file[..., t] if roi_for_file is not None else 1)
                                
                                if is_ensemble:
                                    # Ensemble usually implies 2D+T in this context, but if used here:
                                    strategy = self.model_config['strategy']
                                    mask_t = P.predict_ensemble_slice(ensemble_models, input_t, device, strategy, threshold=self.threshold)
                                else:
                                    mask_t = P.predict_volume_batches(model, input_t, device, threshold=self.threshold)
                                
                                full_mask[..., t] = mask_t
                                f = self.executor.submit(P.unwrap_slice_inplace, vol_t, mask_t, full_unwrapped[..., t], self.venc, current_venc_ratio)
                                futures.append(f)
                                self._update_progress(i + (t + 1) / T, total_files)

                    # 3D Processing
                    elif data.ndim == 3:
                        self.task_update.emit(f"File {i+1}/{total_files} | Predicting Volume")
                        input_t = data * roi_for_file if roi_for_file is not None else data
                        
                        if is_ensemble:
                            strategy = self.model_config['strategy']
                            mask_t = P.predict_ensemble_slice(ensemble_models, input_t, device, strategy, threshold=self.threshold)
                        else:
                            mask_t = P.predict_volume_batches(model, input_t, device, threshold=self.threshold)
                        
                        full_mask[:] = mask_t
                        
                        self.task_update.emit(f"File {i+1}/{total_files} | Unwrapping...")
                        P.unwrap_slice_inplace(data, mask_t, full_unwrapped, self.venc, current_venc_ratio)
                        self._update_progress(i + 1, total_files)

                    for f in futures: 
                        if not self._is_running: break
                        f.result()
                    futures.clear()

                # --- SAVE RESULTS ---
                if not self._is_running: break

                aliased_count = np.sum(full_mask)
                self.progress_log.emit(f"--> DETECTED {aliased_count} ALIASED PIXELS")

                # Dimension Fix for saving (ITK-SNAP compatibility)
                final_mask = full_mask
                final_unwrapped = full_unwrapped
                
                # If using 2D+T model on 3D data, dimensions might need adjustment, 
                # but standard save follows data shape.
                if data.ndim == 3 and not is_classic and not is_ensemble and "2D_T" in model.__class__.__name__:
                    # Edge case handling from original code
                    final_mask = full_mask[:, :, np.newaxis, :]
                    final_unwrapped = full_unwrapped[:, :, np.newaxis, :]

                mask_path, unwrap_path = P.generate_filenames(fpath)
                P.save_nifti_file(final_mask, affine, mask_path, is_mask=True) 
                last_unwrap = P.save_nifti_file(final_unwrapped, affine, unwrap_path, is_mask=False)
                last_mask = mask_path
                self.progress_log.emit(f"Saved to: Outputs/{Path(mask_path).name}")

            if self.executor:
                self.executor.shutdown(wait=False)
            
            if not self._is_running: return

            self.progress_bar.emit(100)
            self.task_update.emit("Finished")
            self.eta_update.emit("Processing Complete")
            
            if len(file_list) > 1:
                outputs_dir = str(Path(last_mask).parent)
                self.finished.emit("BATCH_COMPLETE", outputs_dir)
            else:
                self.finished.emit(last_mask, last_unwrap)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _update_progress(self, current_progress, total_files):
        global_progress = current_progress / total_files
        percent = min(100, int(global_progress * 100))
        self.progress_bar.emit(percent)
        elapsed = time.time() - self.start_time
        if global_progress > 0.01:
            remaining_seconds = (elapsed / global_progress) - elapsed
            m, s = divmod(int(remaining_seconds), 60)
            h, m = divmod(m, 60)
            if h > 0: time_str = f"{h}h {m}m {s}s"
            elif m > 0: time_str = f"{m}m {s}s"
            else: time_str = f"{s}s"
            self.eta_update.emit(f"Estimated time remaining: {time_str}")

# =============================================================================
# WORKER THREAD (IMAGE LOADER)
# =============================================================================
class ImageLoaderWorker(QThread):
    finished = Signal(object, object, str, str) # data, affine, path, type
    error = Signal(str)

    def __init__(self, path, type_):
        super().__init__()
        self.path = path
        self.type_ = type_

    def run(self):
        try:
            nii = nib.load(self.path)
            data = nii.get_fdata()
            affine = nii.affine
            self.finished.emit(data, affine, self.path, self.type_)
        except Exception as e:
            self.error.emit(str(e))

# =============================================================================
# WORKER THREAD (DICOM CONVERTER)
# =============================================================================
class ConverterWorker(QThread):
    finished = Signal(str) # output_path
    error = Signal(str)
    progress_log = Signal(str)
    
    def __init__(self, input_folder, output_folder):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        
    def run(self):
        try:
            self.progress_log.emit(f"Starting conversion for: {self.input_folder}")
            converter = DicomConverter()
            success = converter.convert_folder(self.input_folder, self.output_folder)
            
            if success:
                self.progress_log.emit("Conversion successful.")
                self.finished.emit(self.output_folder)
            else:
                self.error.emit("Conversion failed. Check logs.")
        except Exception as e:
            self.error.emit(str(e))

# =============================================================================
# CUSTOM WIDGETS
# =============================================================================
class DropZone(QFrame):
    file_dropped = Signal(str)
    clicked = Signal()

    def __init__(self, title):
        super().__init__()
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE_ZONE)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        layout = QVBoxLayout()
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #444; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls(): self.file_dropped.emit(url.toLocalFile())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.clicked.emit()

class OutputBox(QFrame):
    def __init__(self, title, color_theme):
        super().__init__()
        self.file_path = None
        self.setStyleSheet(f"QFrame {{ background-color: white; border: 2px solid #e0e0e0; border-radius: 12px; }}")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(f"color: {color_theme}; font-weight: bold; font-size: 14px; border: none;")
        self.status_lbl = QLabel("Waiting for prediction...")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #aaa; font-style: italic; border: none;")
        self.layout.addWidget(self.title_lbl)
        self.layout.addStretch()
        self.layout.addWidget(self.status_lbl)
        self.layout.addStretch()
        self.setLayout(self.layout)
    def set_success(self, path, is_batch=False):
        self.file_path = path
        if is_batch: self.status_lbl.setText(f"Batch Processing Complete\n\nSaved in:\n{Path(path).name}\n\n(Click to Open Folder)")
        else: self.status_lbl.setText(f"File Saved:\n{Path(path).name}\n\n(Click to Open)")
        self.status_lbl.setStyleSheet("color: #333; font-weight: bold; border: none;")
        self.setStyleSheet("QFrame { background-color: #f0f8ff; border: 2px solid #2196F3; border-radius: 12px; } QFrame:hover { background-color: #e3f2fd; }")
        self.setCursor(Qt.PointingHandCursor)
    def mousePressEvent(self, event):
        if self.file_path:
            import subprocess
            target = self.file_path
            if os.path.isdir(target):
                 if sys.platform == 'win32': subprocess.Popen(['explorer', str(target)])
                 else: subprocess.Popen(['xdg-open', str(target)])
            else:
                if sys.platform == 'win32': subprocess.Popen(['explorer', '/select,', str(target)])
                else: subprocess.Popen(['xdg-open', str(Path(target).parent)])

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

# =============================================================================
# WIDGETS FOR STACKED VIEW
# =============================================================================

class PredictionWidget(QWidget):
    request_manual_labeling = Signal(str, str, str)

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.layout = QVBoxLayout(self)
        
        h_layout = QHBoxLayout()
        
        # Left: Drop Zone
        self.input_zone = DropZone("Drag & Drop File/Folder\nOR Click to Select File")
        self.input_zone.file_dropped.connect(self.load_input)
        self.input_zone.clicked.connect(self.open_file_dialog)
        
        # Center: Controls (Centered)
        ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setAlignment(Qt.AlignCenter) 
        
        # --- MODE SELECTION ---
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single AI Model", "Ensemble AI", "Classical Algorithm"])
        self.mode_combo.currentIndexChanged.connect(self.update_ui_mode)
        self.mode_combo.setStyleSheet("QComboBox { background-color: #0078d7; color: white; border-radius: 5px; padding: 5px; font-weight: bold; }")

        # --- STACKED WIDGET FOR MODES ---
        self.stack_options = QStackedWidget()

        # 1. Single Model UI
        self.page_single = QWidget()
        single_layout = QVBoxLayout(self.page_single)
        single_layout.setContentsMargins(0,0,0,0)
        self.single_model_combo = QComboBox()
        ai_models = list(P.MODEL_REGISTRY.keys())
        for m in ai_models:
            self.single_model_combo.addItem(m)
        single_layout.addWidget(QLabel("Select Model:"))
        single_layout.addWidget(self.single_model_combo)

        # 2. Ensemble UI
        self.page_ensemble = QWidget()
        ens_layout = QVBoxLayout(self.page_ensemble)
        ens_layout.setContentsMargins(0,0,0,0)
        
        self.ensemble_strategy = QComboBox()
        self.ensemble_strategy.addItems(["Average", "Majority", "Unanimous"])
        
        ens_layout.addWidget(QLabel("Strategy:"))
        ens_layout.addWidget(self.ensemble_strategy)
        ens_layout.addWidget(QLabel("Select Models:"))
        
        # Scrollable Checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(120)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.model_checkboxes = []
        
        for m in ai_models:
            cb = QCheckBox(m)
            self.model_checkboxes.append(cb)
            self.scroll_layout.addWidget(cb)
            
        scroll.setWidget(scroll_content)
        ens_layout.addWidget(scroll)

        # 3. Classical UI
        self.page_classic = QWidget()
        classic_layout = QVBoxLayout(self.page_classic)
        classic_layout.setContentsMargins(0,0,0,0)
        self.classic_combo = QComboBox()
        classic_modes = [
            "Classic: Laplacian 3D",
            "Classic: Laplacian 4D"
        ]
        for c in classic_modes:
            self.classic_combo.addItem(c)
        classic_layout.addWidget(QLabel("Algorithm:"))
        classic_layout.addWidget(self.classic_combo)

        # Add pages to stack
        self.stack_options.addWidget(self.page_single)
        self.stack_options.addWidget(self.page_ensemble)
        self.stack_options.addWidget(self.page_classic)

        # Venc Inputs
        venc_box = QWidget()
        venc_ly = QHBoxLayout(venc_box)
        venc_ly.setContentsMargins(0,0,0,0)
        venc_ly.setAlignment(Qt.AlignCenter)
        self.venc_input = QLineEdit()
        self.venc_input.setPlaceholderText("VENC")
        venc_validator = QDoubleValidator(0.0, 10000.0, 2)
        venc_validator.setLocale(QLocale(QLocale.C))
        self.venc_input.setValidator(venc_validator)
        self.venc_input.setFixedWidth(80)
        venc_ly.addWidget(QLabel("VENC:"))
        venc_ly.addWidget(self.venc_input)
        venc_ly.addWidget(QLabel("cm/s"))
        venc_ly.addWidget(self.create_help_icon("Velocity Encoding (VENC) in cm/s.\nIf empty, it is read from the file header."))

        # Ratio Inputs
        ratio_box = QWidget()
        ratio_ly = QHBoxLayout(ratio_box)
        ratio_ly.setContentsMargins(0,0,0,0)
        ratio_ly.setAlignment(Qt.AlignCenter)
        self.ratio_input = QLineEdit()
        self.ratio_input.setPlaceholderText("1.0")
        ratio_validator = QDoubleValidator(0.0, 1.0, 6)
        ratio_validator.setLocale(QLocale(QLocale.C))
        self.ratio_input.setValidator(ratio_validator)
        self.ratio_input.setFixedWidth(80)
        ratio_ly.addWidget(QLabel("Ratio:"))
        ratio_ly.addWidget(self.ratio_input)
        ratio_ly.addWidget(QLabel("[0, 1]"))
        ratio_ly.addWidget(self.create_help_icon("VENC Ratio [0, 1].\nAdjusts unwrapping sensitivity relative to VENC.\nDefault is 1.0."))

        # Threshold Inputs
        thresh_box = QWidget()
        thresh_ly = QHBoxLayout(thresh_box)
        thresh_ly.setContentsMargins(0,0,0,0)
        thresh_ly.setAlignment(Qt.AlignCenter)
        self.thresh_input = QLineEdit()
        self.thresh_input.setPlaceholderText("0.5")
        thresh_validator = QDoubleValidator(0.0, 1.0, 2)
        thresh_validator.setLocale(QLocale(QLocale.C))
        self.thresh_input.setValidator(thresh_validator)
        self.thresh_input.setFixedWidth(80)
        self.thresh_input.setText("0.5")
        thresh_ly.addWidget(QLabel("Threshold:"))
        thresh_ly.addWidget(self.thresh_input)
        thresh_ly.addWidget(QLabel("[0, 1]"))
        thresh_ly.addWidget(self.create_help_icon("Probability Threshold [0, 1].\nValues above this are considered aliased.\nLower values increase sensitivity."))

        self.seg_checkbox = QCheckBox("Segment Patient First")
        self.seg_checkbox.setStyleSheet("color: #e0e0e0;")
        
        seg_layout = QHBoxLayout()
        seg_layout.setAlignment(Qt.AlignCenter)
        seg_layout.addWidget(self.seg_checkbox)
        seg_layout.addWidget(self.create_help_icon("If checked, runs aorta/heart segmentation first\nto restrict processing to the ROI."))

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(220, 20)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #bbb; border-radius: 10px; background-color: #e0e0e0; text-align: center; } 
            QProgressBar::chunk { background-color: #28a745; border-radius: 8px; margin: 2px; }
        """)

        self.eta_label = QLabel("Estimated time: --")
        self.eta_label.setAlignment(Qt.AlignCenter)
        self.task_label = QLabel("Ready")
        self.task_label.setAlignment(Qt.AlignCenter)
        self.task_label.setStyleSheet("color: #0078d7; font-weight: bold;")
        self.device_label = QLabel("Device: --")
        self.device_label.setAlignment(Qt.AlignCenter)
        self.device_label.setStyleSheet("color: #ccc; font-style: italic;")
        
        self.run_btn = QPushButton("▶ Run Prediction")
        self.run_btn.setFixedSize(220, 50)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; border-radius: 25px; font-weight: bold; font-size: 14px; } QPushButton:disabled { background-color: #ddd; color: #888; }")
        self.run_btn.clicked.connect(self.run_inference)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setFixedSize(220, 50)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #d9534f; color: white; border-radius: 25px; font-weight: bold; font-size: 14px; } QPushButton:disabled { background-color: #ddd; color: #888; }")
        self.cancel_btn.clicked.connect(self.cancel_inference)

        method_layout = QHBoxLayout()
        method_layout.setAlignment(Qt.AlignCenter)
        method_layout.addWidget(QLabel("Select Method:"))
        method_layout.addWidget(self.create_help_icon("Select the AI model or classical algorithm to use for dealiasing."))
        ctrl_layout.addLayout(method_layout)
        ctrl_layout.addWidget(self.mode_combo)
        ctrl_layout.addWidget(self.stack_options)
        ctrl_layout.addWidget(venc_box)
        ctrl_layout.addWidget(ratio_box)
        ctrl_layout.addWidget(thresh_box)
        ctrl_layout.addLayout(seg_layout)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addSpacing(5)
        ctrl_layout.addWidget(self.cancel_btn)
        ctrl_layout.addSpacing(5)
        ctrl_layout.addWidget(self.progress_bar)
        ctrl_layout.addWidget(self.eta_label)
        ctrl_layout.addWidget(self.task_label)
        ctrl_layout.addWidget(self.device_label)

        # Right: Outputs
        out_panel = QWidget()
        out_ly = QVBoxLayout(out_panel)
        out_ly.setContentsMargins(0,0,0,0)
        self.box_mask = OutputBox("1. Prediction Mask", "#E91E63")
        self.box_unwrap = OutputBox("2. Unwrapped File", "#2196F3")
        
        self.btn_manual_label = QPushButton("✏️ Label Manually")
        self.btn_manual_label.setEnabled(False)
        self.btn_manual_label.clicked.connect(self.emit_manual_labeling)
        self.btn_manual_label.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold; border-radius: 10px; padding: 8px;")

        out_ly.addWidget(self.box_mask)
        out_ly.addSpacing(10)
        out_ly.addWidget(self.box_unwrap)
        out_ly.addSpacing(10)
        out_ly.addWidget(self.btn_manual_label)

        h_layout.addWidget(self.input_zone, 3)
        h_layout.addWidget(ctrl_panel, 2)
        h_layout.addWidget(out_panel, 3)
        
        self.layout.addLayout(h_layout, 4)
        
        # Bottom: Logs
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        self.log_area.setStyleSheet("background-color: #222; color: #ffffff; font-family: Consolas;")
        self.layout.addWidget(self.log_area, 1)

    def update_ui_mode(self, index):
        self.stack_options.setCurrentIndex(index)

    def create_help_icon(self, text):
        btn = QPushButton("❓")
        btn.setToolTip(text)
        btn.setStyleSheet("QPushButton { background-color: transparent; border: none; color: #0078d7; font-weight: bold; margin-left: 5px; font-size: 14px; }")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: QMessageBox.information(self, "Information", text))
        return btn

    def open_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilters(["NIfTI Files (*.nii.gz)", "All Files (*)"])
        if file_dialog.exec():
            self.load_input(file_dialog.selectedFiles()[0])

    def load_input(self, path):
        self.current_input = path
        self.run_btn.setEnabled(True)
        self.eta_label.setText("Estimated time: --")
        self.task_label.setText("Ready to process")
        
        if os.path.isdir(path):
            files = P.resolve_files(path)
            ref_file = files[0] if files else None
        else:
            ref_file = path
            
        if ref_file:
            val = P.get_venc_from_json(ref_file)
            if val: self.venc_input.setText(f"{val:.1f}")
            else: self.venc_input.clear()
            
            ratio = P.extract_venc_ratio_from_filename(Path(ref_file).name)
            if ratio: self.ratio_input.setText(f"{ratio:.2f}")
            else: self.ratio_input.setText("1.0")

        if os.path.isdir(path):
            self.input_zone.label.setText(f"Folder Loaded:\n{Path(path).name}\n(Batch Mode)")
            self.log_area.append(f"Loaded Patient Folder: {path}")
        else:
            self.input_zone.label.setText(f"File Ready:\n{Path(path).name}")
            self.log_area.append(f"Loaded File: {path}")

    def run_inference(self):
        # Validation
        try:
            # Allow empty inputs if using classic mode (which might calculate from headers, but safer to validate)
            # Default to 0.0/1.0 if empty, let worker handle logic
            venc_text = self.venc_input.text().replace(',', '.')
            venc = float(venc_text) if venc_text else 0.0
            
            ratio_text = self.ratio_input.text().replace(',', '.')
            ratio = float(ratio_text) if ratio_text else 1.0
            
            thresh_text = self.thresh_input.text().replace(',', '.')
            threshold = float(thresh_text) if thresh_text else 0.5
            
            if not (0.0 <= ratio <= 1.0): raise ValueError
            if not (0.0 <= threshold <= 1.0): raise ValueError
        except:
            QMessageBox.critical(self, "Error", "Invalid VENC, Ratio [0-1], or Threshold [0-1].")
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.cancel_btn.setEnabled(True)
        self.task_label.setText("Starting...")
        
        # Construct Model Config
        mode_idx = self.mode_combo.currentIndex()
        config = {}
        
        if mode_idx == 0: # Single
            config['mode'] = 'single'
            config['models'] = [self.single_model_combo.currentText()]
            config['strategy'] = None
        elif mode_idx == 1: # Ensemble
            config['mode'] = 'ensemble'
            selected_models = [cb.text() for cb in self.model_checkboxes if cb.isChecked()]
            if not selected_models:
                QMessageBox.warning(self, "Warning", "Please select at least one model for the ensemble.")
                self.run_btn.setEnabled(True); self.run_btn.setText("▶ Run Prediction")
                return
            config['models'] = selected_models
            config['strategy'] = self.ensemble_strategy.currentText()
        else: # Classic
            config['mode'] = 'classic'
            config['models'] = [self.classic_combo.currentText()]
            config['strategy'] = None

        do_seg = self.seg_checkbox.isChecked()
        
        self.worker = InferenceWorker(self.current_input, config, do_seg, venc, ratio, threshold)
        self.worker.progress_log.connect(self.log_area.append)
        self.worker.progress_bar.connect(self.progress_bar.setValue)
        self.worker.eta_update.connect(self.eta_label.setText)
        self.worker.task_update.connect(self.task_label.setText)
        self.worker.device_status.connect(self.device_label.setText)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, res1, res2):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶ Run Prediction")
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.task_label.setText("Done")
        if res1 == "BATCH_COMPLETE":
            self.box_mask.set_success(res2, is_batch=True)
            self.box_unwrap.set_success(res2, is_batch=True)
            self.btn_manual_label.setEnabled(False)
        else:
            self.box_mask.set_success(res1)
            self.box_unwrap.set_success(res2)
            self.last_mask_path = res1
            self.btn_manual_label.setEnabled(True)

    def cancel_inference(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.log_area.append("Cancellation requested by user.")
            self.task_label.setText("Cancelled")
            self.eta_label.setText("Estimated time: --")
            self.progress_bar.setValue(0)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("▶ Run Prediction")
            self.cancel_btn.setEnabled(False)

    def emit_manual_labeling(self):
        unwrap_path = ""
        if self.box_unwrap.file_path:
            unwrap_path = self.box_unwrap.file_path
        if hasattr(self, 'last_mask_path') and self.current_input:
            self.request_manual_labeling.emit(self.current_input, self.last_mask_path, unwrap_path)

    def stop_worker(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

class VisualisationWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Main vertical layout
        layout = QVBoxLayout(self)
        
        # --- Top Section (Inputs + Controls + Plots) ---
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left: Inputs (Clickable DropZones)
        left_box = QWidget()
        left_ly = QVBoxLayout(left_box)
        self.drop_orig = DropZone("Original Data\n(.nii.gz)\n(Click to Load)")
        self.drop_mask = DropZone("Prediction Mask\n(.nii.gz)\n(Click to Load)")
        self.drop_unwrap = DropZone("Unwrapped Data\n(.nii.gz)\n(Click to Load)")
        
        # Click connections
        self.drop_orig.clicked.connect(lambda: self.open_file_dialog('orig'))
        self.drop_mask.clicked.connect(lambda: self.open_file_dialog('mask'))
        self.drop_unwrap.clicked.connect(lambda: self.open_file_dialog('unwrap'))
        
        # Drag-drop connections
        self.drop_orig.file_dropped.connect(lambda p: self.load_nifti_async(p, 'orig'))
        self.drop_mask.file_dropped.connect(lambda p: self.load_nifti_async(p, 'mask'))
        self.drop_unwrap.file_dropped.connect(lambda p: self.load_nifti_async(p, 'unwrap'))
        
        left_ly.addWidget(self.drop_orig)
        left_ly.addWidget(self.drop_mask)
        left_ly.addWidget(self.drop_unwrap)
        
        # Center: Controls
        center_box = QWidget()
        center_ly = QVBoxLayout(center_box)
        center_ly.setAlignment(Qt.AlignCenter)
        
        self.spin_slice = QSpinBox()
        self.spin_slice.setRange(0, 0)
        self.spin_slice.setPrefix("Sagittal Slice: ")
        
        self.scroll_slice = QScrollBar(Qt.Horizontal)
        self.scroll_slice.setRange(0, 0)
        self.spin_slice.valueChanged.connect(self.scroll_slice.setValue)
        self.scroll_slice.valueChanged.connect(self.spin_slice.setValue)
        self.spin_slice.valueChanged.connect(self.update_plot)
        
        self.spin_time = QSpinBox()
        self.spin_time.setRange(0, 0)
        self.spin_time.setPrefix("Time Frame: ")
        
        self.scroll_time = QScrollBar(Qt.Horizontal)
        self.scroll_time.setRange(0, 0)
        self.spin_time.valueChanged.connect(self.scroll_time.setValue)
        self.scroll_time.valueChanged.connect(self.spin_time.setValue)
        self.spin_time.valueChanged.connect(self.update_plot)
        
        btn_refresh = QPushButton("Refresh View")
        btn_refresh.clicked.connect(self.update_plot)
        btn_refresh.setStyleSheet("padding: 10px; background-color: #0078d7; color: white; font-weight: bold; border-radius: 5px;")

        center_ly.addWidget(QLabel("Visualization Controls"))
        center_ly.addSpacing(20)
        center_ly.addWidget(self.spin_slice)
        center_ly.addWidget(self.scroll_slice)
        center_ly.addSpacing(10)
        center_ly.addWidget(self.spin_time)
        center_ly.addWidget(self.scroll_time)
        center_ly.addSpacing(20)
        center_ly.addWidget(btn_refresh)
        center_ly.addStretch()

        # Right: Plots
        right_box = QWidget()
        right_ly = QVBoxLayout(right_box)
        
        # 1. Create Original Canvas & Toolbar
        self.canvas_orig = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar_orig = NavigationToolbar(self.canvas_orig, self)
        
        # 2. Create Mask Canvas & Toolbar
        self.canvas_mask = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar_mask = NavigationToolbar(self.canvas_mask, self)

        # 3. Create Unwrap Canvas & Toolbar
        self.canvas_unwrap = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar_unwrap = NavigationToolbar(self.canvas_unwrap, self)

        right_ly.addWidget(QLabel("Original Input"))
        right_ly.addWidget(self.toolbar_orig)
        right_ly.addWidget(self.canvas_orig)
        
        right_ly.addWidget(QLabel("Prediction Overlay"))
        right_ly.addWidget(self.toolbar_mask)
        right_ly.addWidget(self.canvas_mask)

        right_ly.addWidget(QLabel("Unwrapped View"))
        right_ly.addWidget(self.toolbar_unwrap)
        right_ly.addWidget(self.canvas_unwrap)

        # Flag for zoom sync
        self._syncing = False

        top_layout.addWidget(left_box, 1)
        top_layout.addWidget(center_box, 1)
        top_layout.addWidget(right_box, 3)
        
        # Add Top Section to Main Layout
        layout.addWidget(top_container, 4)

        # --- Bottom Section (Logs) ---
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        self.log_area.setStyleSheet("background-color: #222; color: #ffffff; font-family: Consolas;")
        self.log_area.setPlaceholderText("Debug Logs will appear here...")
        layout.addWidget(self.log_area, 1)

        self.data_orig = None
        self.data_mask = None
        self.data_unwrap = None
        self.loader_thread = None

    def open_file_dialog(self, type_):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilters(["NIfTI Files (*.nii.gz)", "All Files (*)"])
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.load_nifti_async(selected_files[0], type_)

    def load_nifti_async(self, path, type_):
        """Starts the background loader with a spinning dialog."""
        self.log_area.append(f"Loading {type_} from: {path}...")
        
        # 1. Create Progress Dialog (Infinite Spinner)
        self.loading_dialog = QProgressDialog("Loading NIfTI Data...", None, 0, 0, self)
        self.loading_dialog.setWindowTitle("Please Wait")
        self.loading_dialog.setWindowModality(Qt.WindowModal)
        self.loading_dialog.setMinimumDuration(0) # Show immediately
        self.loading_dialog.show()

        # 2. Start Worker Thread
        self.loader_thread = ImageLoaderWorker(path, type_)
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.error.connect(self.on_load_error)
        self.loader_thread.start()

    def on_load_finished(self, data, affine, path, type_):
        self.loading_dialog.close()
        
        if type_ == 'orig':
            self.data_orig = data
            self.drop_orig.label.setText(f"Loaded:\n{Path(path).name}")
            self.log_area.append("Original Data Loaded Successfully.")
        elif type_ == 'mask':
            self.data_mask = data
            self.drop_mask.label.setText(f"Loaded:\n{Path(path).name}")
            self.log_area.append("Mask Data Loaded Successfully.")
        elif type_ == 'unwrap':
            self.data_unwrap = data
            self.drop_unwrap.label.setText(f"Loaded:\n{Path(path).name}")
            self.log_area.append("Unwrapped Data Loaded Successfully.")
        
        # Update max limits
        max_z = 0
        max_t = 0
        datasets = [d for d in [self.data_orig, self.data_mask, self.data_unwrap] if d is not None]
        for d in datasets:
            if d.ndim >= 3: max_z = max(max_z, d.shape[2] - 1)
            if d.ndim == 4: max_t = max(max_t, d.shape[3] - 1)
        
        self.spin_slice.setMaximum(max_z)
        self.scroll_slice.setMaximum(max_z)
        self.spin_slice.setToolTip(f"Range: 0 to {max_z}")
        self.spin_time.setMaximum(max_t)
        self.scroll_time.setMaximum(max_t)
        self.spin_time.setToolTip(f"Range: 0 to {max_t}")
        
        self.update_plot()

    def on_load_error(self, err_msg):
        self.loading_dialog.close()
        msg = f"Failed to load NIfTI:\n{err_msg}"
        self.log_area.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def sync_views(self, event_ax):
        """Synchronizes Zoom/Pan between the two axes."""
        if self._syncing: return
        self._syncing = True
        try:
            axes_list = [self.canvas_orig.axes, self.canvas_mask.axes, self.canvas_unwrap.axes]
            if event_ax in axes_list:
                for ax in axes_list:
                    if ax != event_ax:
                        ax.set_xlim(event_ax.get_xlim())
                        ax.set_ylim(event_ax.get_ylim())
                        ax.figure.canvas.draw_idle()
        except:
            pass
        finally:
            self._syncing = False

    def update_plot(self):
        idx_z = self.spin_slice.value()
        idx_t = self.spin_time.value()
        
        def get_slice(data, z, t):
            if data is None: return None
            if data.ndim == 4:
                if t >= data.shape[3] or z >= data.shape[2]: return None
                sl = data[:, :, z, t]
            elif data.ndim == 3:
                if z >= data.shape[2]: return None
                sl = data[:, :, z]
            else: return None
            return P.transform_rotate_slice(sl, "sagittal")

        slice_orig = get_slice(self.data_orig, idx_z, idx_t)
        slice_mask = get_slice(self.data_mask, idx_z, idx_t)
        slice_unwrap = get_slice(self.data_unwrap, idx_z, idx_t)

        # 1. Clear axes (this disconnects events, so we must reconnect below)
        self.canvas_orig.axes.clear()
        self.canvas_mask.axes.clear()
        self.canvas_unwrap.axes.clear()

        # 2. Plot Original
        if slice_orig is not None:
            self.canvas_orig.axes.imshow(slice_orig, cmap='gray', origin='lower')
            # self.canvas_orig.axes.set_title(f"Sagittal {idx_z} | Time {idx_t}")
            self.canvas_orig.axes.axis('off')

        # 3. Plot Mask Overlay
        if slice_orig is not None:
            self.canvas_mask.axes.imshow(slice_orig, cmap='gray', origin='lower')
        if slice_mask is not None:
            overlay = np.zeros((*slice_mask.shape, 4))
            overlay[slice_mask > 0] = [1, 0, 0, 0.5] 
            self.canvas_mask.axes.imshow(overlay, origin='lower')
            self.canvas_mask.axes.set_title("Prediction Overlay")
            self.canvas_mask.axes.axis('off')

        # 4. RE-CONNECT SYNCHRONIZATION EVENTS (CRITICAL FIX)
        self.canvas_orig.axes.callbacks.connect('xlim_changed', self.sync_views)
        self.canvas_orig.axes.callbacks.connect('ylim_changed', self.sync_views)
        self.canvas_mask.axes.callbacks.connect('xlim_changed', self.sync_views)
        self.canvas_mask.axes.callbacks.connect('ylim_changed', self.sync_views)
        self.canvas_unwrap.axes.callbacks.connect('xlim_changed', self.sync_views)
        self.canvas_unwrap.axes.callbacks.connect('ylim_changed', self.sync_views)

        self.canvas_orig.draw()
        self.canvas_mask.draw()
        self.canvas_unwrap.draw()

class HandLabelingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # --- Top Section ---
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        
        # Left: Drop Zone
        self.drop_zone = DropZone("Load Wrapped Data\n(.nii.gz)\n(Click to Load)")
        self.drop_zone.clicked.connect(self.open_file_dialog)
        self.drop_zone.file_dropped.connect(lambda p: self.load_nifti_async(p))
        
        # Center: Controls
        ctrl_panel = QWidget()
        ctrl_ly = QVBoxLayout(ctrl_panel)
        ctrl_ly.setAlignment(Qt.AlignCenter)
        
        # --- NEW: VENC Controls ---
        venc_group = QGroupBox("Unwrapping Parameters")
        venc_group.setStyleSheet("QGroupBox { font-weight: bold; color: #e0e0e0; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        venc_ly = QVBoxLayout(venc_group)
        
        h_venc = QHBoxLayout()
        self.venc_input = QLineEdit()
        self.venc_input.setPlaceholderText("VENC")
        val_venc = QDoubleValidator(0.0, 10000.0, 2)
        val_venc.setLocale(QLocale(QLocale.C))
        self.venc_input.setValidator(val_venc)
        self.venc_input.setStyleSheet("color: #333; background-color: #fff; border-radius: 4px; padding: 2px;")
        lbl_v = QLabel("VENC:")
        lbl_v.setStyleSheet("color: #e0e0e0;")
        h_venc.addWidget(lbl_v)
        h_venc.addWidget(self.venc_input)
        lbl_unit = QLabel("cm/s")
        lbl_unit.setStyleSheet("color: #e0e0e0;")
        h_venc.addWidget(lbl_unit)
        
        h_ratio = QHBoxLayout()
        self.ratio_input = QLineEdit()
        self.ratio_input.setPlaceholderText("1.0")
        self.ratio_input.setText("1.0")
        val_ratio = QDoubleValidator(0.0, 1.0, 6)
        val_ratio.setLocale(QLocale(QLocale.C))
        self.ratio_input.setValidator(val_ratio)
        self.ratio_input.setStyleSheet("color: #333; background-color: #fff; border-radius: 4px; padding: 2px;")
        lbl_r = QLabel("Ratio:")
        lbl_r.setStyleSheet("color: #e0e0e0;")
        h_ratio.addWidget(lbl_r)
        h_ratio.addWidget(self.ratio_input)
        
        venc_ly.addLayout(h_venc)
        venc_ly.addLayout(h_ratio)
        
        ctrl_ly.addWidget(venc_group)
        ctrl_ly.addSpacing(10)
        
        self.spin_slice = QSpinBox()
        self.spin_slice.setPrefix("Slice: ")
        self.scroll_slice = QScrollBar(Qt.Horizontal)
        self.spin_slice.valueChanged.connect(self.scroll_slice.setValue)
        self.scroll_slice.valueChanged.connect(self.spin_slice.setValue)
        self.spin_slice.valueChanged.connect(self.update_plot)
        
        self.spin_time = QSpinBox()
        self.spin_time.setPrefix("Time: ")
        self.scroll_time = QScrollBar(Qt.Horizontal)
        self.spin_time.valueChanged.connect(self.scroll_time.setValue)
        self.scroll_time.valueChanged.connect(self.spin_time.setValue)
        self.spin_time.valueChanged.connect(self.update_plot)
        
        self.btn_save = QPushButton("💾 Save Unwrapped Volume")
        self.btn_save.clicked.connect(self.save_mask)
        self.btn_save.setStyleSheet("padding: 10px; background-color: #2196F3; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_save.setEnabled(False)
        
        lbl_nav = QLabel("<b>Navigation</b>")
        lbl_nav.setStyleSheet("color: #e0e0e0;")
        ctrl_ly.addWidget(lbl_nav)
        ctrl_ly.addWidget(self.spin_slice)
        ctrl_ly.addWidget(self.scroll_slice)
        ctrl_ly.addSpacing(10)
        ctrl_ly.addWidget(self.spin_time)
        ctrl_ly.addWidget(self.scroll_time)
        ctrl_ly.addStretch()
        lbl_ctrl = QLabel("<b>Controls:</b>")
        lbl_ctrl.setStyleSheet("color: #e0e0e0;")
        ctrl_ly.addWidget(lbl_ctrl)
        lbl_lc = QLabel("Left Click: +2π (Unwrap Up)")
        lbl_lc.setStyleSheet("color: #e0e0e0;")
        ctrl_ly.addWidget(lbl_lc)
        lbl_rc = QLabel("Right Click: -2π (Unwrap Down)")
        lbl_rc.setStyleSheet("color: #e0e0e0;")
        ctrl_ly.addWidget(lbl_rc)
        ctrl_ly.addSpacing(10)
        
        self.btn_undo = QPushButton("↩️ Undo")
        self.btn_undo.clicked.connect(self.undo_last_action)
        self.btn_undo.setStyleSheet("padding: 10px; background-color: #FF5722; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_undo.setEnabled(False)
        ctrl_ly.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↪️ Redo")
        self.btn_redo.clicked.connect(self.redo_last_action)
        self.btn_redo.setStyleSheet("padding: 10px; background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_redo.setEnabled(False)
        ctrl_ly.addWidget(self.btn_redo)
        
        ctrl_ly.addWidget(self.btn_save)
        
        # Right: Canvas
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        self.lbl_velocity = QLabel("Hover to see velocity")
        self.lbl_velocity.setAlignment(Qt.AlignCenter)
        self.lbl_velocity.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 14px; margin: 2px;")
        
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.lbl_velocity)
        plot_layout.addWidget(self.canvas)
        
        top_layout.addWidget(self.drop_zone, 1)
        top_layout.addWidget(ctrl_panel, 1)
        top_layout.addLayout(plot_layout, 4)
        
        self.layout.addWidget(top_container, 4)
        
        # Bottom: Logs
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        self.log_area.setStyleSheet("background-color: #222; color: #ffffff; font-family: Consolas;")
        self.layout.addWidget(self.log_area, 1)
        
        # State
        self.data = None
        self.affine = None
        self.wrap_val = 2 * P.GLOBAL_MAX_PHASE # Assuming data is in original intensity range
        self.history = []
        self.redo_stack = []

    def open_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilters(["NIfTI Files (*.nii.gz)", "All Files (*)"])
        if file_dialog.exec():
            self.load_nifti_async(file_dialog.selectedFiles()[0])

    def load_nifti_async(self, path):
        # Auto-detect VENC/Ratio
        val = P.get_venc_from_json(path)
        if val: self.venc_input.setText(f"{val:.1f}")
        else: self.venc_input.clear()
        
        ratio = P.extract_venc_ratio_from_filename(Path(path).name)
        if ratio: self.ratio_input.setText(f"{ratio:.2f}")
        else: self.ratio_input.setText("1.0")

        self.log_area.append(f"Loading: {path}...")
        self.loading_dialog = QProgressDialog("Loading Data...", None, 0, 0, self)
        self.loading_dialog.setWindowModality(Qt.WindowModal)
        self.loading_dialog.show()
        
        self.loader = ImageLoaderWorker(path, 'data')
        self.loader.finished.connect(self.on_load_finished)
        self.loader.error.connect(lambda e: self.log_area.append(f"Error: {e}"))
        self.loader.start()

    def on_load_finished(self, data, affine, path, type_):
        self.loading_dialog.close()
        self.data = data.astype(np.float32) # Ensure float for modifications
        self.affine = affine
        self.drop_zone.label.setText(f"Loaded:\n{Path(path).name}")
        self.btn_save.setEnabled(True)
        
        self.history = []
        self.redo_stack = []
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)
        
        # Setup sliders
        max_z = self.data.shape[2] - 1 if self.data.ndim >= 3 else 0
        max_t = self.data.shape[3] - 1 if self.data.ndim == 4 else 0
        
        self.spin_slice.setMaximum(max_z)
        self.scroll_slice.setMaximum(max_z)
        self.spin_time.setMaximum(max_t)
        self.scroll_time.setMaximum(max_t)
        
        self.canvas.axes.clear() # Reset zoom for new file
        self.update_plot()
        self.log_area.append("Data loaded. Ready to edit.")

    def update_plot(self, _=None):
        if self.data is None: return
        
        # Capture current view limits to preserve zoom
        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        is_default = (xlim == (0.0, 1.0) and ylim == (0.0, 1.0))
        
        z = self.spin_slice.value()
        t = self.spin_time.value()
        
        if self.data.ndim == 4:
            slice_data = self.data[:, :, z, t]
        elif self.data.ndim == 3:
            slice_data = self.data[:, :, z]
        else:
            slice_data = self.data
            
        # Rotate for display (Sagittal view standard)
        slice_display = P.transform_rotate_slice(slice_data, "sagittal")
        
        self.canvas.axes.clear()
        im = self.canvas.axes.imshow(slice_display, cmap='gray', origin='lower')
        self.canvas.axes.set_title(f"Slice {z} | Time {t}")
        self.canvas.axes.axis('off')
        
        # Restore zoom if it wasn't the default empty state
        if not is_default:
            self.canvas.axes.set_xlim(xlim)
            self.canvas.axes.set_ylim(ylim)
            
        self.canvas.draw()

    def on_hover(self, event):
        if event.inaxes != self.canvas.axes: return
        if self.data is None: return
        if event.xdata is None or event.ydata is None: return
        
        # Map coordinates (Sagittal view is transposed)
        ix, iy = int(event.xdata), int(event.ydata)
        array_x, array_y = ix, iy
        
        z = self.spin_slice.value()
        t = self.spin_time.value()
        
        if array_x < 0 or array_x >= self.data.shape[0] or array_y < 0 or array_y >= self.data.shape[1]:
            self.lbl_velocity.setText("Out of bounds")
            return

        if self.data.ndim == 4: val = self.data[array_x, array_y, z, t]
        elif self.data.ndim == 3: val = self.data[array_x, array_y, z]
        else: val = self.data[array_x, array_y]
            
        try:
            v_text = self.venc_input.text().replace(',', '.')
            venc = float(v_text) if v_text else 0.0
        except: venc = 0.0
            
        if venc > 0: self.lbl_velocity.setText(f"Velocity: {(val / P.GLOBAL_MAX_PHASE) * venc:.2f} cm/s (Phase: {val:.0f})")
        else: self.lbl_velocity.setText(f"Phase: {val:.0f} (Set VENC for velocity)")

    def on_click(self, event):
        if event.inaxes != self.canvas.axes: return
        if self.data is None: return
        if event.xdata is None or event.ydata is None: return
        
        # Map click coordinates back to array indices
        # Note: transform_rotate_slice("sagittal") does a Transpose (.T)
        # So displayed X is array Y, displayed Y is array X.
        ix, iy = int(event.xdata), int(event.ydata)
        
        # Reverse the transform: Sagittal view is .T, so we swap x/y back
        # Display (x, y) -> Array (y, x)
        array_x, array_y = ix, iy
        
        z = self.spin_slice.value()
        t = self.spin_time.value()
        
        # Bounds check
        if array_x < 0 or array_x >= self.data.shape[0] or array_y < 0 or array_y >= self.data.shape[1]: return
        
        # Calculate wrap_val dynamically based on VENC inputs
        try:
            v_text = self.venc_input.text().replace(',', '.')
            venc = float(v_text) if v_text else 0.0
            r_text = self.ratio_input.text().replace(',', '.')
            ratio = float(r_text) if r_text else 1.0
        except:
            venc = 0.0
            ratio = 1.0
            
        if venc > 0: self.wrap_val = 2.0 * venc * ratio
        else: self.wrap_val = 2.0 * P.GLOBAL_MAX_PHASE

        # Modify Data
        if self.data.ndim == 4:
            old_val = self.data[array_x, array_y, z, t]
            if event.button == 1: # Left Click
                self.data[array_x, array_y, z, t] += self.wrap_val
                self.log_area.append(f"Added wrap at ({array_x}, {array_y}, {z}, {t})")
            elif event.button == 3: # Right Click
                self.data[array_x, array_y, z, t] -= self.wrap_val
                self.log_area.append(f"Removed wrap at ({array_x}, {array_y}, {z}, {t})")
            
            self.history.append(((array_x, array_y, z, t), old_val))
            
        elif self.data.ndim == 3:
            old_val = self.data[array_x, array_y, z]
            if event.button == 1:
                self.data[array_x, array_y, z] += self.wrap_val
            elif event.button == 3:
                self.data[array_x, array_y, z] -= self.wrap_val
            
            self.history.append(((array_x, array_y, z), old_val))
        
        self.btn_undo.setEnabled(True)
        self.redo_stack = []
        self.btn_redo.setEnabled(False)
        self.update_plot()

    def undo_last_action(self):
        if not self.history: return
        
        coords, old_val = self.history.pop()
        current_val = self.data[coords]
        self.data[coords] = old_val
        
        self.redo_stack.append((coords, current_val))
        self.btn_redo.setEnabled(True)
        
        self.log_area.append(f"Undid change at {coords}")
        self.update_plot()
        if not self.history: self.btn_undo.setEnabled(False)

    def redo_last_action(self):
        if not self.redo_stack: return
        
        coords, new_val = self.redo_stack.pop()
        current_val = self.data[coords]
        self.data[coords] = new_val
        
        self.history.append((coords, current_val))
        self.btn_undo.setEnabled(True)
        
        self.log_area.append(f"Redid change at {coords}")
        self.update_plot()
        if not self.redo_stack: self.btn_redo.setEnabled(False)

    def save_mask(self):
        if self.data is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Unwrapped", "unwrapped_manual.nii.gz", "NIfTI (*.nii.gz)")
        if path:
            aff = self.affine if self.affine is not None else np.eye(4)
            img = nib.Nifti1Image(self.data, aff)
            nib.save(img, path)
            self.log_area.append(f"Saved file to {path}")
            QMessageBox.information(self, "Saved", f"File saved to:\n{path}")
    
    def load_data(self, orig_path, *args):
        # Ignores extra arguments (mask_path, unwrap_path) passed by MainWindow
        self.load_nifti_async(orig_path)

class DicomConverterWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        h_layout = QHBoxLayout()
        
        # Left: Drop Zone
        self.drop_zone = DropZone("Drag & Drop DICOM Folder\n(Click to Select)")
        self.drop_zone.file_dropped.connect(self.load_input)
        self.drop_zone.clicked.connect(self.open_folder_dialog)
        
        # Center: Controls
        ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_info = QLabel("Conversion Parameters")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 14px; color: #e0e0e0;")
        
        self.btn_convert = QPushButton("▶ Convert to NIfTI")
        self.btn_convert.setFixedSize(220, 50)
        self.btn_convert.setEnabled(False)
        self.btn_convert.setStyleSheet("QPushButton { background-color: #28a745; color: white; border-radius: 25px; font-weight: bold; font-size: 14px; } QPushButton:disabled { background-color: #ddd; color: #888; }")
        self.btn_convert.clicked.connect(self.run_conversion)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(220, 20)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.hide()
        
        ctrl_layout.addWidget(self.lbl_info)
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(self.btn_convert)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(self.progress_bar)
        
        # Right: Output
        out_panel = QWidget()
        out_ly = QVBoxLayout(out_panel)
        out_ly.setContentsMargins(0,0,0,0)
        self.box_output = OutputBox("Output Folder", "#2196F3")
        out_ly.addWidget(self.box_output)
        
        h_layout.addWidget(self.drop_zone, 3)
        h_layout.addWidget(ctrl_panel, 2)
        h_layout.addWidget(out_panel, 3)
        
        self.layout.addLayout(h_layout, 4)
        
        # Bottom: Logs
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        self.log_area.setStyleSheet("background-color: #222; color: #ffffff; font-family: Consolas;")
        self.layout.addWidget(self.log_area, 1)
        
        self.input_path = None

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.load_input(folder)

    def load_input(self, path):
        if os.path.isdir(path):
            self.input_path = path
            self.drop_zone.label.setText(f"Selected:\n{os.path.basename(path)}")
            self.btn_convert.setEnabled(True)
            self.log_area.append(f"Selected input: {path}")
        else:
            self.log_area.append("Error: Please drop a folder, not a file.")

    def run_conversion(self):
        if not self.input_path: return
        
        output_folder = self.input_path + "_NIfTI"
        self.log_area.append(f"Output will be saved to: {output_folder}")
        
        self.btn_convert.setEnabled(False)
        self.progress_bar.show()
        
        self.worker = ConverterWorker(self.input_path, output_folder)
        self.worker.progress_log.connect(self.log_area.append)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        
    def on_finished(self, output_path):
        self.progress_bar.hide()
        self.btn_convert.setEnabled(True)
        self.box_output.set_success(output_path, is_batch=True)
        self.log_area.append("Done.")
        
    def on_error(self, msg):
        self.progress_bar.hide()
        self.btn_convert.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

# =============================================================================
# MAIN WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiology AI Segmenter")
        self.resize(1200, 850)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title Label
        self.title_label = QLabel("Prediction")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0; margin-bottom: 10px;")
        main_layout.addWidget(self.title_label)

        # Stacked Widget for Pages
        self.stacked_widget = QStackedWidget()
        self.page_predict = PredictionWidget(self)
        self.page_visual = VisualisationWidget()
        self.page_labeling = HandLabelingWidget()
        self.page_converter = DicomConverterWidget()
        
        self.stacked_widget.addWidget(self.page_predict)
        self.stacked_widget.addWidget(self.page_visual)
        self.stacked_widget.addWidget(self.page_labeling)
        self.stacked_widget.addWidget(self.page_converter)
        
        main_layout.addWidget(self.stacked_widget)

        # Menu Bar
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("Menu")
        
        act_pred = QAction("Prediction", self)
        act_pred.triggered.connect(lambda: self.switch_view(0, "Prediction"))
        view_menu.addAction(act_pred)
        
        act_vis = QAction("Visualise Predictions", self)
        act_vis.triggered.connect(lambda: self.switch_view(1, "Visualise Predictions"))
        view_menu.addAction(act_vis)
        
        act_label = QAction("Hand Labeling", self)
        act_label.triggered.connect(lambda: self.switch_view(2, "Hand Labeling"))
        view_menu.addAction(act_label)
        
        act_convert = QAction("DICOM Converter", self)
        act_convert.triggered.connect(lambda: self.switch_view(3, "DICOM Converter"))
        view_menu.addAction(act_convert)

        # Connect Prediction Widget signal
        self.page_predict.request_manual_labeling.connect(self.load_and_switch_to_labeling)

    def switch_view(self, index, title):
        self.stacked_widget.setCurrentIndex(index)
        self.title_label.setText(title)

    def load_and_switch_to_labeling(self, orig_path, mask_path, unwrap_path):
        self.switch_view(2, "Hand Labeling")
        self.page_labeling.load_data(orig_path, mask_path, unwrap_path)

    def closeEvent(self, event):
        # Stop prediction worker
        self.page_predict.stop_worker()
        # Stop converter worker if running (terminate is harsh but ensures app closes)
        if hasattr(self.page_converter, 'worker') and self.page_converter.worker.isRunning():
            self.page_converter.worker.terminate()
            self.page_converter.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Removed Splash Screen
    
    # 3. Launch Main Window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
