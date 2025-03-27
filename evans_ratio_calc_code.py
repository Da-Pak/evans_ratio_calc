import os
import numpy as np
import pandas as pd
import cv2
import pydicom
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, median_filter
from skimage import measure, feature, graph
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import SimpleITK as sitk
from HD_BET import run
from itertools import product

#-[Search Parameters]--------------------------------
ventricle_threshold_range = list(range(11, 30))  # hydrocephalus detection thresholds
median_filter_size_range = [3, 4, 5, 6]  # median filter sizes to try
parameter_combinations = list(product(ventricle_threshold_range, median_filter_size_range))

#-[Functions]--------------------------------

def resample(image, image_thickness, pixel_spacing):
    """
    Resamples the image to isotropic voxels.
    
    Args:
        image: 3D image array
        image_thickness: Slice thickness in mm
        pixel_spacing: Pixel spacing in mm [x, y]
        
    Returns:
        Resampled image with isotropic voxels
    """
    x_pixel = float(pixel_spacing[0])
    y_pixel = float(pixel_spacing[1])
    size = np.array([float(image_thickness), x_pixel/2, y_pixel/2])
    image_shape = np.array([image.shape[0], image.shape[1], image.shape[2]])
    new_shape = image_shape * size
    new_shape = np.round(new_shape)
    resize_factor = new_shape / image_shape
    resampled_image = ndi.interpolation.zoom(image, resize_factor)
    return resampled_image

def get_3d_image(folder_path):
    """
    Loads a 3D image from a DICOM folder and resamples it.
    
    Args:
        folder_path: Path to the folder containing DICOM files
        
    Returns:
        Resampled 3D image array
    """
    first_slice_name = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')][0]
    first_slice = pydicom.dcmread(first_slice_name)
    spacing = list(first_slice.PixelSpacing) + [first_slice.SliceThickness]    
    
    # Create ImageSeriesReader and set metadata update
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    # Get series IDs
    series_IDs = reader.GetGDCMSeriesIDs(folder_path)
    # Get filenames for the first series
    series_file_names = reader.GetGDCMSeriesFileNames(folder_path, series_IDs[0])

    # Set filenames
    reader.SetFileNames(series_file_names)

    # Sort slices by ImagePositionPatient tag
    sorted_file_names = sorted(series_file_names, key=lambda f: float(sitk.ReadImage(f).GetMetaData('0020|0032').split('\\')[2]))
    reader.SetFileNames(sorted_file_names)

    # Read the image
    image = reader.Execute()
    
    # Change data axis and resample
    img = sitk.GetArrayFromImage(image)
    img = resample(img, first_slice.SliceThickness, first_slice.PixelSpacing)
    
    return img

def get_largest_region(binary_image):
    """
    Finds the largest connected region in a binary image.
    
    Args:
        binary_image: Binary image array
        
    Returns:
        Mask with only the largest region
    """
    labeled_image, num_features = ndi.label(binary_image)
    region_props = measure.regionprops(labeled_image)
    if len(region_props) == 0:
        return binary_image
    largest_cavity = sorted(region_props, key=lambda x: x.area)[-1]

    mask = np.zeros_like(labeled_image)
    mask[labeled_image != largest_cavity.label] = 1
    return mask

def get_largest_and_central_region(binary_image):
    """
    Finds the largest and most centrally located region in a binary image.
    
    Args:
        binary_image: Binary image array
        
    Returns:
        Mask with only the largest central region
    """
    labeled_image, num_features = ndi.label(binary_image)
    region_props = measure.regionprops(labeled_image)
    
    if len(region_props) < 3:
        return binary_image
    
    # Select top 3 largest regions
    largest_cavities = sorted(region_props, key=lambda x: x.area, reverse=True)[:3]
    
    # Calculate image center coordinates
    center_z, center_y, center_x = np.array(binary_image.shape) // 2
    
    # Calculate centrality score for each region
    centrality_scores = []
    for cavity in largest_cavities:
        points = np.array(cavity.coords)
        distances = np.sqrt(np.sum((points - [center_z, center_y, center_x])**2, axis=1))
        centrality_score = np.mean(1 / (distances + 1))  # Average of inverse distances
        centrality_scores.append(centrality_score)
    
    # Select the most central region
    most_central_cavity = largest_cavities[np.argmax(centrality_scores)]
    
    mask = np.zeros_like(labeled_image)
    mask[labeled_image != most_central_cavity.label] = 1
    return mask

def get_largest_region2(binary_image):
    """
    Finds the second largest connected region in a binary image.
    
    Args:
        binary_image: Binary image array
        
    Returns:
        Mask with only the second largest region
    """
    labeled_image, num_features = ndi.label(binary_image)
    region_props = measure.regionprops(labeled_image)

    largest_cavity = sorted(region_props, key=lambda x: x.area)[-2]

    mask = np.zeros_like(labeled_image)
    mask[labeled_image != largest_cavity.label] = 1
    return mask

def euclidian_dist_mask(mask, diameter_threshold):
    """
    Segments a mask using Euclidean distance transform.
    
    Args:
        mask: Binary mask
        diameter_threshold: Threshold for diameter
        
    Returns:
        Segmented binary mask
    """
    distance_transform = ndi.distance_transform_edt(mask)
    split_cavities = distance_transform > (diameter_threshold / 2)
    return split_cavities

def get_endpoint(skel):
    """
    Finds endpoints in a skeletonized image.
    
    Args:
        skel: Skeletonized binary image
        
    Returns:
        x, y coordinates of endpoints
    """
    # Find non-zero pixel positions
    (rows, cols) = np.nonzero(skel)

    # Initialize coordinate list
    skel_coords = []

    # For each non-zero pixel...
    for (r, c) in zip(rows, cols):
        # Extract 8-connected neighbors
        (col_neigh, row_neigh) = np.meshgrid(np.array([c-1, c, c+1]), np.array([r-1, r, r+1]))

        # Convert to integers for image indexing
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')

        # Convert to single 1D array and check non-zero positions
        pix_neighbourhood = skel[row_neigh, col_neigh].ravel() != 0

        if np.sum(pix_neighbourhood) == 2:
            skel_coords.append((r, c))
            
    x = list(map(lambda x: x[0], skel_coords))
    y = list(map(lambda x: x[1], skel_coords))
    return x, y

def check_points_positions(x, y):
    """
    Checks and sorts the positions of points.
    
    Args:
        x, y: Coordinates of points
        
    Returns:
        Sorted x, y coordinates in order: right-bottom, right-top, left-bottom, left-top
    """
    # Order: right-bottom, right-top, left-bottom, left-top
    points = list(zip(y, x))
    
    points.sort(key=lambda point: (point[0], point[1]))  # Sort by x, then by y

    left_top, left_bottom, right_top, right_bottom = None, None, None, None

    if points[0][1] < points[1][1]:
        left_top, left_bottom = points[0], points[1]
    else:
        left_top, left_bottom = points[1], points[0]

    if points[2][1] < points[3][1]:
        right_bottom, right_top = points[2], points[3]
    else:
        right_bottom, right_top = points[3], points[2]

    y = [right_bottom[0], right_top[0], left_bottom[0], left_top[0]]
    x = [right_bottom[1], right_top[1], left_bottom[1], left_top[1]]

    return x, y

def get_path_route(skel_image, x, y, point1, point2):
    """
    Finds the path between two points in a skeletonized image.
    
    Args:
        skel_image: Skeletonized binary image
        x, y: Coordinates of all endpoints
        point1, point2: Indices of the two points to connect
        
    Returns:
        x_path, y_path: Coordinates of the path
        cost: Cost of the path
    """
    x, y = check_points_positions(x, y)
    skel_route = np.where(skel_image, 0, 1000)
    path, cost = graph.route_through_array(skel_route, start=(x[point1], y[point1]), end=(x[point2], y[point2]), fully_connected=True)
    
    x_path = list(map(lambda x: x[0], path))
    y_path = list(map(lambda x: x[1], path))
    
    return x_path, y_path, cost

def skel_get(image_whole, image_all):
    """
    Calculates Evans ratio from ventricle and brain tissue images.
    
    Args:
        image_whole: Ventricle image
        image_all: Brain tissue image
        
    Returns:
        Various measurements including ventricle and cerebral dimensions, Evans ratio,
        slice number, and visualization data
    """

    for slice_number in range(image_whole.shape[0] - 1, image_whole.shape[0]//2, -1):
        slice_number_ratio = slice_number/image_whole.shape[0]
        image = image_whole[slice_number]
        image = np.logical_not(image)
        image2 = image.astype('uint8').copy()
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
        image2_close = cv2.morphologyEx(image2, cv2.MORPH_CLOSE, kernel, iterations=2)
        image_cere = image_all[slice_number].astype('uint8').copy()

        image_cere = cv2.morphologyEx(image_cere, cv2.MORPH_OPEN, kernel, iterations=1)
        image_cere_largest = get_largest_region(image_cere)

        skel_image = skeletonize(image2_close)
        x, y = get_endpoint(skel_image)
        
        if len(x) != 4:
            continue
            
        x_path1, y_path1, cost1 = get_path_route(skel_image, x, y, 0, 2)
        x_path2, y_path2, cost2 = get_path_route(skel_image, x, y, 1, 3)

        if cost1 > 0 or cost2 > 0:
            continue
            
        if not check_cross_point(y_path1, x_path1, y_path2, x_path2):
            continue
            
        cere_val, cere_lines, cere_max, cere_min = calc_longest_line(np.logical_not(image_cere_largest).astype(int).T, image.shape[-1])
        vent_val, vent_lines, vent_max, vent_min = calc_longest_line_vent(image2_close.T, image2_close.shape[-1], np.logical_not(image_cere_largest).astype(int).T)
        
        vent_length = vent_max - vent_min
        cere_length = cere_max - cere_min
        evans_ratio = vent_length/cere_length
        
        if evans_ratio < 0.8:
            return vent_max, vent_min, cere_max, cere_min, vent_length, cere_length, evans_ratio, slice_number, slice_number_ratio, cere_lines, vent_lines
        
    return None, None, None, None, None, None, None, None, None, None, None
    
def get_eroded_image(image):
    """
    Processes an image with dilation followed by erosion (morphological closing).
    
    Args:
        image: Input binary image
        
    Returns:
        Processed binary image
    """
    image = image.astype('uint8').copy()
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
    dilate = cv2.dilate(image, kernel, iterations=2)
    eroded_image = cv2.erode(dilate, kernel, iterations=2)
    return eroded_image

def fit_ellipse_and_get_info(edges):
    """
    Fits an ellipse to edges and returns center point, angle, and ellipse info.
    
    Args:
        edges: 2D binary array (edge-detected image)
        
    Returns:
        center: Center point (x, y) of the fitted ellipse
        angle: Angle of the major axis (in degrees)
        ellipse: Fitted ellipse info (center, axis lengths, angle)
    """
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = np.vstack(contours)

    if len(all_points) < 5:
        raise ValueError("Not enough points to fit an ellipse")

    ellipse = cv2.fitEllipse(all_points)
    (x, y), (minor_axis, major_axis), angle = ellipse

    if angle > 90:
        angle = angle - 180

    if (major_axis - minor_axis)/major_axis < 0.05:
        return (x, y), 0, ellipse  # Return 0 if difference between axes is less than 5%
    else:
        return (x, y), angle/2, ellipse

def check_symmetry(image, center_x):
    """
    Checks the left-right symmetry of an image.
    
    Args:
        image: Input binary image
        center_x: x-coordinate of the center line
        
    Returns:
        dice_score: Dice similarity coefficient between left and right sides
    """
    center_x = int(center_x)
    
    # Split image into left and right sides
    left_side = image[:, :center_x]
    right_side = image[:, center_x:]
    right_side_flipped = np.fliplr(right_side)
    
    # Match minimum width of both sides
    min_width = min(left_side.shape[1], right_side_flipped.shape[1])
    left_side_cropped = left_side[:, -min_width:]
    right_side_cropped = right_side_flipped[:, -min_width:]
    
    # Calculate Dice score
    dice_score = 2 * np.sum(left_side_cropped & right_side_cropped) / \
                 (np.sum(left_side_cropped) + np.sum(right_side_cropped))
    
    return dice_score

def align_head_ct(ct_volume):
    """
    Aligns a 3D head CT volume and checks symmetry.
    
    Args:
        ct_volume: 3D numpy array (CT scan)
        
    Returns:
        aligned_volume: 3D numpy array (aligned CT scan)
        symmetry_score: Symmetry score
    """
    bone_threshold = 60
    bone_mask = ct_volume > bone_threshold
    projection_threshold = 0
    bone_mask_list = bone_mask.sum(axis=1).sum(axis=1)
    max_pixel_sum_index = np.argmax(bone_mask_list[:len(bone_mask_list)//3*2])

    projection = np.sum(bone_mask[max_pixel_sum_index:], axis=0)
    projection2 = (projection > projection_threshold).astype('uint8')
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7,7))
    projection2 = cv2.morphologyEx(projection2, cv2.MORPH_CLOSE, kernel, iterations=3)
    projection2 = cv2.morphologyEx(projection2, cv2.MORPH_OPEN, kernel, iterations=3)
     
    projection2 = gaussian_filter(projection2.astype(float), sigma=2)
    
    edges = feature.canny(projection2.astype(np.uint8), sigma=1, low_threshold=0.1, high_threshold=0.3)

    center, angle, ellipse = fit_ellipse_and_get_info(edges)
    
    # Shift center
    shift_x = ct_volume.shape[2] // 2 - int(center[0])
    shift_y = ct_volume.shape[1] // 2 - int(center[1])
    
    # Apply rotation and shift
    aligned_volume = ndi.rotate(ct_volume, angle, axes=(1, 2), reshape=False, order=1, mode='nearest')
    aligned_volume = ndi.shift(aligned_volume, (0, shift_y, shift_x))
    aligned_bone = aligned_volume > bone_threshold
    aligned_projection = np.sum(aligned_bone[max_pixel_sum_index:], axis=0)
    aligned_projection2 = (aligned_projection > projection_threshold).astype('uint8')
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7,7))
    aligned_projection2 = cv2.morphologyEx(aligned_projection2, cv2.MORPH_CLOSE, kernel, iterations=3)
    aligned_projection2 = cv2.morphologyEx(aligned_projection2, cv2.MORPH_OPEN, kernel, iterations=3)
    symmetry_score = check_symmetry(aligned_projection2, aligned_projection.shape[1] // 2)
   
    return aligned_volume, symmetry_score

def check_cross_point(y_path1, x_path1, y_path2, x_path2):
    """
    Checks if two paths cross each other.
    
    Args:
        y_path1, x_path1: Coordinates of first path
        y_path2, x_path2: Coordinates of second path
        
    Returns:
        Boolean indicating whether paths cross properly
    """
    zip_list1 = list(zip(x_path1, y_path1))
    zip_list2 = list(zip(x_path2, y_path2))

    cross_list = []
    for i in zip_list1:
        for j in zip_list2:
            if i == j:
                cross_list.append(i)
                
    if len(cross_list) == 1:
        return True
    if len(cross_list) == 0:
        return False
        
    for i in range(len(cross_list)-1):
        before_num = 1000
        for i in range(len(cross_list) - 1):
            left_difference = cross_list[i+1][0] - cross_list[i][0]
            right_difference = cross_list[i+1][1] - cross_list[i][1]
            if left_difference in [-1, 0, 1] and right_difference in [-1, 0, 1]:
                if before_num == 1000:
                    before_num = i
                else:
                    if abs((before_num - i)) != 1:
                        return False
                    else:
                        before_num = i
    return True

def calc_longest_line(seg, size):
    """
    Calculates the longest line in a segment.
    
    Args:
        seg: Binary segmentation
        size: Size of the image
        
    Returns:
        max_value: Length of the longest line
        max_line_index: Index of the longest line
        line_max_value: Maximum value along the line
        line_min_value: Minimum value along the line
    """
    template = np.array([list(range(size))]*size).T
    calc = np.where(seg, template, np.nan)
    line_length = np.nanmax(calc, axis=0) - np.nanmin(calc, axis=0)
    
    if len(np.unique(line_length)) <= 2:
        return np.nan, np.nan, np.nan, np.nan
        
    max_value = np.nanmax(line_length)
    max_line_index = np.where(line_length == max_value)[0]
    max_line_index = max_line_index[len(max_line_index)//2]
    line_max_value = np.nanmax(calc[:, max_line_index])
    line_min_value = np.nanmin(calc[:, max_line_index])
    
    return max_value, max_line_index, line_max_value, line_min_value

def calc_longest_line_vent(seg, size, cere_image):
    """
    Calculates the longest line in the ventricle segment.
    
    Args:
        seg: Ventricle segmentation
        size: Size of the image
        cere_image: Cerebral tissue image
        
    Returns:
        max_value: Length of the longest line
        max_line_index: Index of the longest line
        line_max_value: Maximum value along the line
        line_min_value: Minimum value along the line
    """
    template = np.array([list(range(size))]*size).T
    calc = np.where(seg, template, np.nan)
    line_length = np.nanmax(calc, axis=0) - np.nanmin(calc, axis=0)
    
    if len(np.unique(line_length)) <= 2:
        return np.nan, np.nan, np.nan, np.nan    
        
    line_length_half = np.where(cere_image > 0)
    line_length_half = np.nanmin(line_length_half) + (np.nanmax(line_length_half) - np.nanmin(line_length_half))//2
    line_length = line_length[:line_length_half]
    max_value = np.nanmax(line_length)
    max_line_index = np.where(line_length == max_value)[0]
    max_line_index = max_line_index[len(max_line_index)//2]
    line_max_value = np.nanmax(calc[:, max_line_index])
    line_min_value = np.nanmin(calc[:, max_line_index])
    
    return max_value, max_line_index, line_max_value, line_min_value

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
