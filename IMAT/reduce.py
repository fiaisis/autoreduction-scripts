# Copyright (C) 2025 ISIS Rutherford Appleton Laboratory UKRI
# SPDX - License - Identifier: GPL-3.0-or-later
from __future__ import annotations

import cv2
from pathlib import Path
from typing import Any

import numpy as np

from mantidimaging import __version__
from mantidimaging.core.data import ImageStack
from mantidimaging.core.data.dataset import Dataset
from mantidimaging.core.io.loader import loader
from mantidimaging.core.io.loader.loader import create_loading_parameters_for_file_path, ImageParameters
from mantidimaging.core.operations.divide import DivideFilter
from mantidimaging.core.operations.loader import load_filter_packages
from mantidimaging.core.reconstruct import get_reconstructor_for
from mantidimaging.core.rotation import CorTiltDataModel
from mantidimaging.core.utility.data_containers import ReconstructionParameters, ScalarCoR, Degrees, FILE_TYPES
from mantidimaging.core.io.saver import image_save
from mantidimaging.core.rotation.polyfit_correlation import find_center

PROCESS_FILE = Path("process.json")
OUT_DIR = Path("/home/ubuntu/output")
FILTERS = {f.__name__: f for f in load_filter_packages()}
RECON_DEFAULT_SETTINGS = {'algorithm': 'FBP_CUDA', 'filter_name': 'ram-lak', 'cor': 1, 'tilt': 0, 'max_projection_angle': 360}
DATASET_PATH = Path("/home/ubuntu/large")

def version() -> str:
    return __version__


def load_dataset(file_path: Path) -> Dataset:
    parameters = create_loading_parameters_for_file_path(file_path)

    # Based on MainWindowModel.do_load_dataset()
    def load(im_param: ImageParameters) -> ImageStack:
        return loader.load_stack_from_image_params(im_param, None, dtype=parameters.dtype)

    sample = load(parameters.image_stacks[FILE_TYPES.SAMPLE])
    sample.set_geometry()
    ds = Dataset(sample=sample)
    sample._is_sinograms = parameters.sinograms
    sample.pixel_size = parameters.pixel_size

    for file_type in [
            FILE_TYPES.FLAT_BEFORE,
            FILE_TYPES.FLAT_AFTER,
            FILE_TYPES.DARK_BEFORE,
            FILE_TYPES.DARK_AFTER,
            FILE_TYPES.PROJ_180,
    ]:
        if im_param := parameters.image_stacks.get(file_type):
            image_stack = load(im_param)
            ds.set_stack(file_type, image_stack)

    return ds


def show_dataset(ds: Dataset):
    print(f"Dataset: {ds}")
    for stack in ds.all:
        print(f"  name: {stack.name} id: {stack.id}: {stack.data.shape}")


def run_operation(dataset: Dataset, op_name: str, params: dict[str, Any]):
    op_class = FILTERS[op_name]
    op_func = op_class.filter_func
    apply_to_dataset = True

    print(f"Running operation: {op_name} with params: {params}")

    match op_name:
        case 'FlatFieldFilter':
            params = setup_flat_field(dataset, params)
            apply_to_dataset = False

    op_func(dataset.sample, **params)
    if apply_to_dataset:
        for stack in [dataset.flat_before, dataset.flat_after, dataset.dark_before, dataset.dark_after, dataset.proj180deg]:
            if stack:
                op_func(stack, **params)


def setup_flat_field(dataset: Dataset, params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    if dataset.flat_before:
        params['flat_before'] = dataset.flat_before
    if dataset.dark_before:
        params['dark_before'] = dataset.dark_before
    if dataset.flat_after:
        params['flat_after'] = dataset.flat_after
    if dataset.dark_after:
        params['dark_after'] = dataset.dark_after
    return params


def run_recon(image_stack, settings=None):
    if settings is None:
        settings = {}
    settings = RECON_DEFAULT_SETTINGS | settings

    print(f"Running reconstruction with settings: {settings}")

    do_clip = settings.pop('clip', False)

    reconstructor = get_reconstructor_for(settings['algorithm'])

    settings['cor'] = ScalarCoR(settings['cor'])
    settings['tilt'] = Degrees(settings['tilt'])

    params = ReconstructionParameters(**settings)

    cor_tilt = CorTiltDataModel()
    cor_tilt.set_precalculated(params.cor, params.tilt)

    cor_list = cor_tilt.get_all_cors_from_regression(image_stack.height)

    recon = reconstructor.full(image_stack, cor_list, params, progress=None)
    # recon = DivideFilter.filter_func(recon, value=params.pixel_size, unit="micron", progress=None)

    if do_clip:
        np.clip(recon.data, a_min=0, a_max=None, out=recon.data)
    return recon


def save_stack(image_stack: ImageStack, out_dir: Path):
    image_save(image_stack, out_dir, overwrite_all=True)


def get_contour_orientation(image) -> bool:
    """Returns True if object appears vertical, False if horizontal"""
    # Convert to uint8 and normalize to 0-255 range
    img_normalized = ((image - image.min()) * (255 / (image.max() - image.min()))).astype(np.uint8)
    
    # Invert the image since we're looking for dark objects on light background
    img_normalized = cv2.bitwise_not(img_normalized)
    
    # Apply threshold to create binary image
    _, binary = cv2.threshold(img_normalized, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try to find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return True  # Default to vertical if no contours found
        
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit an oriented bounding box
    rect = cv2.minAreaRect(largest_contour)
    width = rect[1][0]
    height = rect[1][1]
    
    # For vertical objects, width will be less than height
    is_vertical = width < height
    
    # Draw the oriented bounding box for visualization
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Create RGB visualization
    viz_img = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(viz_img, [box], 0, (0, 255, 0), 2)  # Green box
    
    # Add text showing orientation
    text = "VERTICAL" if is_vertical else "HORIZONTAL"
    cv2.putText(viz_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add dimensions for debugging
    dim_text = f"W: {width:.1f}, H: {height:.1f}"
    cv2.putText(viz_img, dim_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save visualization
    cv2.imwrite("/home/ubuntu/orientation_detection.png", viz_img)
    
    return is_vertical


def find_border_of_object(dataset: Dataset) -> list[int]:
    def get_largest_contour_bbox(image) -> tuple[int, int, int, int] | None:
        # Convert to uint8 and normalize to 0-255 range
        img_normalized = ((image - image.min()) * (255 / (image.max() - image.min()))).astype(np.uint8)
        
        # Invert the image since we're looking for dark objects on light background
        img_normalized = cv2.bitwise_not(img_normalized)
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(img_normalized, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try to find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add margin of 5% of image dimensions
        height, width = image.shape
        margin_x = int(width * 0.05)
        margin_y = int(height * 0.05)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)
        
        return x, y, x + w, y + h

    # Get images at different positions in the stack
    num_images = len(dataset.sample.data)
    indices = [
        0,  # First image
        num_images // 3,  # One-third through
        2 * num_images // 3,  # Two-thirds through
        -1  # Last image
    ]
    
    # Initialize with maximum possible bounds
    height, width = dataset.sample.data[0].shape
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    
    found_any_contours = False
    
    # Process each image
    for i, idx in enumerate(indices):
        bbox = get_largest_contour_bbox(dataset.sample.data[idx])
        if bbox is not None:
            found_any_contours = True
            x1, y1, x2, y2 = bbox
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
    
    final_bbox = None
    if found_any_contours:
        final_bbox = [min_x, min_y, max_x, max_y]
    else:
        final_bbox = [0, 0, width - 1, height - 1]
    
    return final_bbox


print(f"Mantid Imaging {version()}")

dataset = load_dataset(DATASET_PATH)

# ROI norm
roi_settings = {
    "region_of_interest": [0, 0, 64, 64],
    "normalisation_mode": "Stack Average"
}
run_operation(dataset, "RoiNormalisationFilter", roi_settings)

# Flat fielding
flat_field_settings = {
    "selected_flat_fielding": "Only Before",
    "use_dark": False
}
run_operation(dataset, "FlatFieldFilter", flat_field_settings)

# Check if rotation is needed and rotate if object is horizontal
# Check orientation in first image of stack
is_vertical = get_contour_orientation(dataset.sample.data[0])
if not is_vertical:
    print("Object appears horizontal, rotating 90 degrees clockwise")
    rotation_settings = {
        "angle": 270,
    }
    run_operation(dataset, "RotateFilter", rotation_settings)
else:
    print("Object appears vertical, no rotation needed")

# Crop (ML detect)
border = find_border_of_object(dataset)
crop_settings = {
    "region_of_interest": border
}
run_operation(dataset, "CropCoordinatesFilter", crop_settings)

# Outlier Removal
outlier_filter_settings = {
    "diff": 1000.0,
    "radius": 3,
    "mode": "bright"
}
run_operation(dataset, "OutliersFilter", outlier_filter_settings)

# TODO: Setup a debug mode that saves out images at the end of each step with appropriate names

# Reconstruction
if dataset.proj180deg is not None:
    print("Calculating center of rotation and tilt using 180 degree projection")
    cor, tilt = find_center(dataset.sample, None)
    cor = cor.value
    tilt = tilt.value
else:
    print("Using default center of rotation and tilt, as no 180 degree projection present")
    cor = int(dataset.sample.width / 2)
    tilt = 0.0

recon_settings = {
    "algorithm": "FBP_CUDA",
    "filter_name": "ram-lak",
    "num_iter": 1,
    "cor": cor,
    "tilt": tilt,
    "pixel_size": 0.0,
    "alpha": 1.0,
    "gamma": 1,
    "stochastic": False,
    "projections_per_subset": 50,
    "regularisation_percent": 30,
    "regulariser": "TV"
}
recon = run_recon(dataset.sample, recon_settings)
dataset.add_recon(recon)

save_stack(dataset.recons[0], OUT_DIR)
