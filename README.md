# FlowClear AI
A desktop application to detect, segement and correct velocity aliasing in 4D Flow MRI

## Overview
FlowClear-AI is a standalone desktop application designed for processing 4D Flow MRI data. It provides tools for:
*   Detecting and correcting phase aliasing (velocity wraps).
*   Segmenting anatomical regions of interest (Aorta, Heart).
*   Converting DICOM files to NIfTI format.
*   Manually correcting phase wraps.

It combines classical signal processing techniques with modern deep learning approaches to streamline 4D Flow analysis.

## Features
*   **AI-Powered Unwrapping:** Utilizes 2D, 3D, and 2D+T UNET models to automatically detect and correct phase aliasing.
*   **Ensemble Mode:** Combines multiple AI models using averaging, majority voting, or unanimous agreement for robust unwrapping.
*   **Classical Unwrapping:** Includes Laplacian-based unwrapping algorithms.
*   **Automatic Segmentation:** Integrates `mrsegmentator` to automatically segment the aorta and heart from magnitude MRI data.
*   **DICOM Conversion:** Converts folders of DICOM images into NIfTI format for processing.
*   **Manual Correction:** Interactive GUI to manually fix any remaining phase wraps.
*   **Visualization:** Displays original, unwrapped, and segmented data.

## Usage
1.  **Download:** Download the latest release from the [Releases](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/releases) page.
2.  **Extract:** Unzip the downloaded folder.
3.  **Run:** Double-click `FlowClear-AI.exe`.

See the in-app help buttons for information about each parameter.

## License
This project is licensed under a proprietary license. See `LICENSE.md` for details.
