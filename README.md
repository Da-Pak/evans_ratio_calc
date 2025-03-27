# Evans Ratio Calculation from Brain CT Images

This project provides code for measuring ventricle and cerebral sizes in brain CT images to calculate the Evans ratio.

## Project Structure

The project consists of the following files:

- `utils.py`: Contains all functions needed for image processing, analysis, and Evans ratio calculation
- `main.py`: Main script that executes the actual analysis
- `evans_ratio_calc_code.py`: Original integrated code (for reference)

## Requirements

The following libraries are required to run this project:

```
numpy
pandas
opencv-python (cv2)
pydicom
scipy
scikit-image
matplotlib
SimpleITK
HD-BET
```

## Usage Instructions

1. Install the required libraries:
   ```
   pip install numpy pandas opencv-python pydicom scipy scikit-image matplotlib SimpleITK
   pip install git+https://github.com/MIC-DKFZ/HD-BET
   ```

2. Set up the data directory structure:
   ```
   Directory/
   ├── Hydrocephalus/
   │   ├── patient1_folder/
   │   │   ├── DICOM_file1.dcm
   │   │   ├── DICOM_file2.dcm
   │   │   └── ...
   │   └── ...
   └── Normal/
       ├── patient1_folder/
       │   ├── DICOM_file1.dcm
       │   ├── DICOM_file2.dcm
       │   └── ...
       └── ...
   ```

3. Run the `main.py` script:
   ```
   python main.py
   ```

4. Results will be saved in the following files:
   - `evans_ratio_results.xlsx`: Evans ratio and related measurements for all patients
   - `symmetry_scores.txt`: Record of symmetry scores

## How It Works

The code operates in the following steps:

1. Load DICOM files and convert them to a 3D image
2. Align the image and check symmetry
3. Perform skull stripping
4. Segment brain tissue and ventricles
5. Measure the maximum length of ventricles and cerebrum
6. Calculate the Evans ratio

## Analysis Parameters

The main parameters used in the analysis are:

- `ventricle_threshold_range`: Range of thresholds for ventricle segmentation (11-29)
- `median_filter_size_range`: Range of median filter sizes for noise reduction (3-6)
- `bone_threshold`: Threshold for identifying bone tissue (300)
- `air_threshold`: Threshold for identifying air (-300) 