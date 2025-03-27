import os
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from itertools import product
import cv2

# Import custom functions
from utils import (
    get_3d_image, get_largest_region, get_largest_and_central_region,
    skel_get, align_head_ct, run
)

# Set search parameters
ventricle_threshold_range = list(range(11, 30))  # Hydrocephalus detection thresholds
median_filter_size_range = [3, 4, 5, 6]  # Median filter sizes to try
parameter_combinations = list(product(ventricle_threshold_range, median_filter_size_range))

def main():
    result_list = []
    symmetry_score_list = []

    # Process by cohort
    for cohort in ['Hydrocephalus', 'Normal']:
        folder_dir = f'Directory/{cohort}/'
        folder_paths = os.listdir(folder_dir)
        
        for N, folder in enumerate(folder_paths):
            print('cohort :', cohort)
            print('folder :', folder)
            print('order :', N)
            folder_path = os.path.join(folder_dir, folder)

            # Load 3D image
            image = get_3d_image(folder_path)
            image_array = image.copy()
            image_array = median_filter(image_array, size=3)

            file_result_box = [folder]
            if len(image_array) < 20:
                file_result_box.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                continue
                
            bone_threshold = 300
            binary_image = image_array > bone_threshold
            bone_mask = get_largest_region(binary_image)
            
            # Check symmetry and align
            count = 0
            symmetry_score = 0
            while symmetry_score < 0.95:
                image_array, symmetry_score = align_head_ct(image_array)   
                count += 1
                if count > 2:
                    symmetry_score_list.append(str(symmetry_score) + "_break")
                    break
                    
            if count <= 2:
                symmetry_score_list.append(str(symmetry_score) + "_ok")

            # Create brain mask
            shrinked_mask = run.run_hd_bet(image_array, mode='accurate', device=0) # skull stripping with hd-bet
            shrinked_mask = np.where((shrinked_mask == 1) & (bone_mask == 0), 0, shrinked_mask)

            air_threshold = -300
            air_mask = image_array < air_threshold
            shrinked_mask = np.where((shrinked_mask == 1) & (air_mask == 1), 0, shrinked_mask)
            largest_shrinked_mask = get_largest_region(shrinked_mask)

            tissue_image = np.ma.array(image_array, mask=largest_shrinked_mask)
            tissue_image_binary_mask = tissue_image > 0
            tissue_binary_image = np.empty_like(image_array)
            tissue_binary_image[tissue_image_binary_mask == 1] = 1

            # Analyze ventricles and calculate Evans ratio
            for vent_threshold, median_size in parameter_combinations:
                vent_binary_mask = (tissue_image < vent_threshold) & (tissue_image > 0) # Parameter to adjust    

                vent_binary_image = np.empty_like(image_array)
                vent_binary_image[vent_binary_mask == 1] = 1

                vent_binary_image = vent_binary_image.astype('uint8')
                kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
                vent_binary_image = cv2.dilate(vent_binary_image, kernel)
                vent_binary_image = cv2.erode(vent_binary_image, kernel)

                vent_binary_box_median = median_filter(vent_binary_image, size=median_size)  # Parameter to adjust                                              
                largest_vent_binary_image = get_largest_and_central_region(vent_binary_box_median)

                vent_max, vent_min, cere_max, cere_min, vent_length, cere_length, evans_ratio, slice_number, slice_number_ratio, cere_lines, vent_lines = skel_get(largest_vent_binary_image, tissue_binary_image)
                
                if evans_ratio is not None:
                    file_result_box.extend(['ratio_ok', vent_length, cere_length, evans_ratio, vent_threshold, median_size, slice_number, slice_number_ratio])
                    break
            else:
                file_result_box.extend(['ratio_break',np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            result_list.append(file_result_box)
    
    # Convert results to DataFrame and save to Excel file
    result_df = pd.DataFrame(result_list, columns=['Folder', 'Status', 'Ventricle Length', 'Cerebral Length', 
                                               'Evans Ratio', 'Ventricle Threshold', 'Median Size', 
                                               'Slice Number', 'Slice Number Ratio'])
    result_df.to_excel('evans_ratio_results.xlsx', index=False)
    
    # Save symmetry scores
    with open('symmetry_scores.txt', 'w') as f:
        for score in symmetry_score_list:
            f.write(f"{score}\n")

if __name__ == "__main__":
    main() 