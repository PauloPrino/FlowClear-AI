import subprocess
import json
import pydicom
import os
import shutil
import sys

class DicomConverter:
    def __init__(self):
        pass

    def convert_folder(self, input_folder, output_folder):
        """
        Converts a single patient folder (containing DICOMs) to NIfTI.
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Check for dcm2niix
        # Check for dcm2niix in PATH or current directory (for portable app)
        dcm2niix_cmd = "dcm2niix"
        
        # Priority 1: Check if bundled with PyInstaller
        if hasattr(sys, '_MEIPASS'):
            bundled_exe = os.path.join(sys._MEIPASS, "dcm2niix.exe")
            if os.path.exists(bundled_exe):
                dcm2niix_cmd = bundled_exe
        
        # Priority 2: Check local directory (useful for development/portable)
        elif os.path.exists(os.path.join(os.getcwd(), "dcm2niix.exe")):
            dcm2niix_cmd = os.path.join(os.getcwd(), "dcm2niix.exe")
            
        # Priority 3: Check System PATH
        elif shutil.which("dcm2niix") is None:
            print("Error: dcm2niix not found in PATH, local folder, or bundle.")
            return False

        # Output filename pattern
        patient_name = os.path.basename(os.path.normpath(input_folder))
        
        command = [
            dcm2niix_cmd,
            "-o", output_dir_path(output_folder),
            "-f", f"{patient_name}_%p_%s", # Patient_Protocol_Series
            "-z", "y", # compress
            "-m", "y", # json sidecar
            input_folder
        ]

        try:
            print(f"Executing: {' '.join(command)}")
            subprocess.run(command, check=True)
            
            # After conversion, try to extract headers similar to the original script
            self.save_dicom_headers(input_folder, output_folder, patient_name)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Conversion error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def save_dicom_headers(self, patient_path, output_folder, patient_name):
        """
        Extracts DICOM headers to JSON, matching the logic of the original script.
        """
        print(f"Extracting DICOM headers for {patient_name}...")
        
        # Walk through the input folder to find DICOM files
        for root, dirs, files in os.walk(patient_path):
            if not files: continue
            
            # Try to find a dicom file
            dicom_file = None
            for f in files:
                if f.lower().endswith('.dcm') or "." not in f: # Assume no extension might be DICOM
                    dicom_file = os.path.join(root, f)
                    break
            
            if not dicom_file: continue

            try:
                # Determine protocol name from folder name
                protocol = os.path.basename(root)
                
                dicom_header = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                header_dict = {}
                for elem in dicom_header:
                    if "PixelData" in elem.name: continue
                    if " Date" in elem.name: 
                        value = ""
                    else:
                        value = str(elem.value)
                    
                    header_dict[elem.name] = {
                        "tag": str(elem.tag),
                        "value": value,
                        "VR": elem.VR,
                    }

                # Construct JSON name
                series_num = str(dicom_header.get((0x0020,0x0011)).value if (0x0020,0x0011) in dicom_header else "unknown")
                json_name = f"{patient_name}_header_{protocol}_{series_num}.json"
                json_file = os.path.join(output_folder, json_name)

                with open(json_file, 'w') as f:
                    json.dump(header_dict, f, indent=4, default=str)
                
            except Exception as e:
                # Ignore non-dicom files or read errors
                pass

def output_dir_path(path):
    # Helper to ensure path string format
    return str(path).replace('\\', '/')
