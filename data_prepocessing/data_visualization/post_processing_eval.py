
import copy
import cc3d
import os
import nibabel as nib
import numpy as np

import json

import pandas as pd
from scipy.ndimage import label
import scipy.ndimage as ndimage

from monai.transforms import (
    LoadImage,    # for loading .nii or .nii.gz
    SaveImage,    # for saving .nii or .nii.gz
    Rotate90,
    SpatialCrop,
    SpatialPad,
)

import torch.nn.functional as F
import torch

def pad_image_to_shape(image):
    """
    Pads a 3D image tensor to the shape (200, 200, 350).

    Parameters:
    - image: A 3D torch tensor of shape (a, a, b).

    Returns:
    - Padded image with shape (200, 200, 350).
    """
    image = torch.from_numpy(image)
    current_depth, current_height, current_width = image.shape

    # Calculate padding for depth (symmetric padding to 200)
    pad_depth_top = max((200 - current_depth) // 2, 0)
    pad_depth_bottom = max(200 - current_depth - pad_depth_top, 0)

    # Calculate padding for height (symmetric padding to 200)
    pad_height_top = max((200 - current_height) // 2, 0)
    pad_height_bottom = max(200 - current_height - pad_height_top, 0)

    # Calculate padding for width (pad from the beginning to 350)
    pad_width_left = max(350 - current_width, 0)
    pad_width_right = 0

    # Apply padding
    if current_depth < 200 or current_height < 200 or current_width < 350:
        image = F.pad(
            image,
            (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom, pad_depth_top, pad_depth_bottom),
            mode='constant',
            value=0,
        )

    return image.numpy()


def adjust_volume_shape(prediction_data, label_data):
    """
    Adjust the shape of prediction_data to match the shape of label_data.

    Parameters:
    prediction_data (numpy.ndarray): The 3D volume to be adjusted, shape (x, x, y).
    label_data (numpy.ndarray): The reference 3D volume, shape (a, a, b).

    Returns:
    numpy.ndarray: Adjusted prediction_data with the same shape as label_data.
    """
    pred_x, _, pred_y = prediction_data.shape
    label_x, _, label_y = label_data.shape

    # Adjust the x dimension symmetrically
    if pred_x > label_x:
        excess_x = pred_x - label_x
        cut_x = excess_x // 2
        prediction_data = prediction_data[cut_x:pred_x - cut_x, cut_x:pred_x - cut_x, :]

    # Adjust the y dimension by cutting from the left
    if pred_y > label_y:
        prediction_data = prediction_data[:, :, :label_y]

    return prediction_data
def invert_prediction_transform(
        pred_nifti_path,
        output_nifti_path,
        final_shape=(192, 192, 352),
        original_shape=(200, 200, 350),
        rotate90_axes=(0, 1),
        rotate90_k=1,
):
    """
    Inverts the shape/orientation transforms that took a volume
    from original_shape -> final_shape (with a 90-degree rotation),
    returning it to original_shape orientation.

    Args:
        pred_nifti_path : str
            Path to the predicted .nii.gz file (already in shape final_shape).
        output_nifti_path : str
            Where to save the inverted .nii.gz file.
        final_shape : tuple
            The shape of the input to this inversion code. (What you ended with.)
        original_shape : tuple
            The shape you want to return to (200,200,350).
        rotate90_axes : tuple of ints
            Axes that were used in the original Rotate90d.
        rotate90_k : int
            The number of 90-degree rotations that were originally applied
            (typically 1 if you used `transforms.Rotate90d(keys=["image"], k=1)`).
    """

    # 1) Load your prediction (no metadata tracking by default)
    loader = LoadImage(image_only=True)
    pred_data = loader(pred_nifti_path)  # shape = (192,192,352) for example

    # 2) Undo the center crop in X/Y if you had cropped from (200,200) -> (192,192).
    #    We can pad back from (192,192,352) -> (200,200,352).
    #    By default, SpatialPad will pad on both sides (method="symmetric").
    #    Make sure this matches how you performed your center crop.
    #    If you know you only cropped from the center in X/Y, you can do symmetrical padding:
    x_original, y_original, z_original = original_shape
    pad_transform = SpatialPad(spatial_size=(x_original, y_original, final_shape[2]),
                               method="symmetric")
    print(f"pred data in invert: {pred_data}")
    pred_data = pad_transform(pred_data)

    # 3) Undo the Rotate90d.
    #    If you originally did `Rotate90d(k=1)`, then to invert, you can do `k=3`
    #    (3 more 90-degree rotations in the same axis = 270 degrees = inverse).
    #    Alternatively, you can specify `k=(-rotate90_k) % 4`.
    invert_k = (-rotate90_k) % 4  # e.g. if rotate90_k=1, invert_k=3
    rotate_transform = Rotate90(spatial_axes=rotate90_axes, k=invert_k)
    pred_data = rotate_transform(pred_data)

    # 4) Undo any extra padding in Z if you went from 350 to 352 slices.
    #    If you originally padded +2 slices at the end, then you can crop from
    #    z=0..349 (which is 350 slices).
    #    This is an assumption that the padding was at the end or is symmetrical.
    #    Adjust roi_start and roi_end to match your actual pad logic.
    if z_original < final_shape[2]:  # i.e. 350 < 352
        crop_transform = SpatialCrop(roi_start=(0, 0, 0),
                                     roi_end=(x_original, y_original, z_original))
        pred_data = crop_transform(pred_data)

    return pred_data

def resize_3d_prediction(prediction, target_shape):
    """
    Resizes a 3D prediction to match the size of a given label using nearest-neighbor interpolation.

    Args:
        prediction (numpy array): 3D binary prediction (e.g., shape like (192, 192, 352)).
        target_shape (tuple): Target shape (e.g., shape of the label, like (200, 200, 320)).

    Returns:
        numpy array: Resized binary prediction with shape `target_shape`.
    """
    # Calculate zoom factors for each dimension
    zoom_factors = (
        target_shape[0] / prediction.shape[0],
        target_shape[1] / prediction.shape[1],
        target_shape[2] / prediction.shape[2]
    )

    # Resize using nearest-neighbor interpolation (order=0)
    resized_prediction = ndimage.zoom(prediction, zoom_factors, order=0)
    return resized_prediction

def pad_and_crop(prediction, label):
    # Get the shapes of the prediction and label arrays
    pred_shape = prediction.shape
    label_shape = label.shape

    # Calculate the padding for the first two dimensions
    pad1 = (label_shape[0] - pred_shape[0]) // 2
    pad2 = (label_shape[1] - pred_shape[1]) // 2

    # Calculate the cropping for the third dimension
    crop_start = 0
    crop_end = label_shape[2]

    # Pad the first two dimensions of the prediction array
    padded_prediction = np.pad(prediction, ((pad1, label_shape[0] - pred_shape[0] - pad1),
                                            (pad2, label_shape[1] - pred_shape[1] - pad2),
                                            (0, 0)), mode='constant', constant_values=0)

    # Crop the third dimension of the prediction array
    cropped_prediction = padded_prediction[:, :, -1*crop_end:]

    return cropped_prediction

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 26  # 18 or 26
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp
def false_pos_pix(gt_array, pred_array, pred_array_baseline=None):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)

    false_pos = 0
    false_pos_num = 0
    for idx in range(1, min(pred_conn_comp.max() + 1, 50)): #pred_conn_comp.max() + 1):
        #print(f"idx in false pos pix: {idx} max: {pred_conn_comp.max()}")
        comp_mask = np.isin(pred_conn_comp, idx)
        #if comp_mask.sum() <= 8:  # ignore small connected components (0.64 ml)
        #    print("less than 8")
        #    continue
        #if comp_mask.sum() <= 3:  # ignore small connected components (0.81 ml)
            #print("less than 3")
        #    continue
        if (comp_mask * gt_array).sum() == 0:
            false_pos = false_pos + comp_mask.sum()
            false_pos_num = false_pos_num + 1

    return false_pos_num


def false_neg_pix(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    #print(gt_conn_comp)
    false_neg = 0
    true_pos = 0
    false_neg_num = 0
    true_pos_num = 0
    for idx in range(1, min(gt_conn_comp.max() + 1, 50)):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask * pred_array).sum() == 0:
            false_neg = false_neg + comp_mask.sum()
            false_neg_num = false_neg_num + 1
        else:
            true_pos = true_pos + comp_mask.sum()
            true_pos_num = true_pos_num + 1

    return true_pos_num, false_neg_num

def TPFPFNHelper(y_pred, y):
        n_pred_ch = y_pred.shape[1]
        #print(y_pred.shape)
        #print(y_pred)
        #print(f"max: {np.max(y_pred)}")
        #print(f"min: {np.min(y_pred)}")
        y_pred = np.where(y_pred < 0.5, 0, 1)

        """
        if n_pred_ch > 1:
            y_pred = np.argmax(y_pred, axis=1, keepdims=True)
        else:
            raise ValueError("y_pred must have more than 1 channel, use softmax instead")

        n_gt_ch = y.shape[1]
        if n_gt_ch > 1:
            y = np.argmax(y, axis=1, keepdims=True)
        """

        # reducing only spatial dimensions (not batch nor channels)
        TP_sum = 0
        FP_sum = 0
        FN_sum = 0
        y_copy = copy.deepcopy(y).squeeze()
        y_pred_copy = copy.deepcopy(y_pred).squeeze()
        if y_copy.ndim == 3:  # if batch dim is reduced
            y_copy = y_copy[np.newaxis, ...]
            y_pred_copy = y_pred_copy[np.newaxis, ...]

        for ii in range(y_copy.shape[0]):
            y_ = y_copy[ii]
            y_pred_ = y_pred_copy[ii]

            FP = false_pos_pix(y_, y_pred_)
            TP, FN = false_neg_pix(y_, y_pred_)

            TP_sum += TP
            FP_sum += FP
            FN_sum += FN

        return TP_sum, FP_sum, FN_sum  # all are volumes


def dice_score_helper(y_pred, y):
    """
    Computes the Dice score for the prediction volumes.

    Parameters:
    - y_pred: Predicted volumes (numpy array).
    - y: Ground truth volumes (numpy array).

    Returns:
    - dice_score: Dice score for the volumes.
    """
    # Binarize the predictions
    y_pred = np.where(y_pred < 0.5, 0, 1)

    y_copy = copy.deepcopy(y).squeeze()
    y_pred_copy = copy.deepcopy(y_pred).squeeze()

    # Add batch dimension if missing
    if y_copy.ndim == 3:  # if batch dim is reduced
        y_copy = y_copy[np.newaxis, ...]
        y_pred_copy = y_pred_copy[np.newaxis, ...]

    dice_scores = []

    # Iterate over the batch
    for ii in range(y_copy.shape[0]):
        y_ = y_copy[ii]
        y_pred_ = y_pred_copy[ii]

        # Compute True Positives, False Positives, and False Negatives
        intersection = np.sum(y_ * y_pred_)  # TP
        sum_y = np.sum(y_)  # FN + TP
        sum_y_pred = np.sum(y_pred_)  # FP + TP

        # Dice Score calculation
        dice = (2 * intersection) / (sum_y + sum_y_pred + 1e-6)  # Add epsilon to avoid division by zero
        dice_scores.append(dice)

    # Return the score across the batch
    return dice_scores


def TPFPFN_with_dice_threshold(y_pred, y, dice_threshold=0.5):
    """
    Computes the TP, FP, and FN counts based on volumes with a Dice threshold.

    A predicted volume is only considered a True Positive (TP) if the Dice
    score between the predicted and ground truth volume is above the threshold.

    Parameters:
    - y_pred: Predicted volumes (numpy array).
    - y: Ground truth volumes (numpy array).
    - dice_threshold: Minimum Dice score for a volume to be considered a TP.

    Returns:
    - TP_count: Total count of True Positive volumes.
    - FP_count: Total count of False Positive volumes.
    - FN_count: Total count of False Negative volumes.
    """
    # Binarize the predictions
    y_pred = np.where(y_pred < 0.5, 0, 1)

    y_copy = copy.deepcopy(y).squeeze()
    y_pred_copy = copy.deepcopy(y_pred).squeeze()

    # Add batch dimension if missing
    if y_copy.ndim == 3:  # if batch dim is reduced
        y_copy = y_copy[np.newaxis, ...]
        y_pred_copy = y_pred_copy[np.newaxis, ...]

    TP_count = 0
    FP_count = 0
    FN_count = 0

    # Iterate over the batch
    for ii in range(y_copy.shape[0]):
        y_ = y_copy[ii]
        y_pred_ = y_pred_copy[ii]

        # Compute Dice score for the current volume
        intersection = np.sum(y_ * y_pred_)  # Overlapping region
        sum_y = np.sum(y_)  # Ground truth volume size
        sum_y_pred = np.sum(y_pred_)  # Predicted volume size

        dice = (2 * intersection) / (sum_y + sum_y_pred + 1e-6)  # Avoid division by zero

        if dice > dice_threshold:
            # Count as True Positive if Dice threshold is met
            TP_count += 1
        else:
            if sum_y_pred > 0:
                # False Positive if there's a prediction but no matching ground truth
                FP_count += 1
            if sum_y > 0:
                # False Negative if there's ground truth but no matching prediction
                FN_count += 1

    return TP_count, FP_count, FN_count

def TPFPFN_with_same_max_v1(y_pred, y, image, intensity_threshold=0.1):
    """
    Computes the TP, FP, and FN counts based on volumes with an intensity threshold.

    A predicted volume is only considered a True Positive (TP) if the maximum
    pixel intensities of the y_pred mask applied to the image and the y mask
    applied to the image are within the given threshold.

    Parameters:
    - y_pred: Predicted masks (numpy array, binarized with 0s and 1s).
    - y: Ground truth masks (numpy array, binarized with 0s and 1s).
    - image: The corresponding image (numpy array, same shape as masks).
    - intensity_threshold: Maximum allowed difference in max intensity for TP.

    Returns:
    - TP_count: Total count of True Positive volumes.
    - FP_count: Total count of False Positive volumes.
    - FN_count: Total count of False Negative volumes.
    """
    # Binarize the predictions
    y_pred = np.where(y_pred < 0.5, 0, 1)

    y_copy = copy.deepcopy(y).squeeze()
    y_pred_copy = copy.deepcopy(y_pred).squeeze()
    image_copy = copy.deepcopy(image).squeeze()

    # Add batch dimension if missing
    if y_copy.ndim == 3:  # if batch dim is reduced
        y_copy = y_copy[np.newaxis, ...]
        y_pred_copy = y_pred_copy[np.newaxis, ...]
        image_copy = image_copy[np.newaxis, ...]

    TP_count = 0
    FP_count = 0
    FN_count = 0

    # Iterate over the batch
    for ii in range(y_copy.shape[0]):
        y_ = y_copy[ii]
        y_pred_ = y_pred_copy[ii]
        image_ = image_copy[ii]

        # Apply masks to the image
        max_intensity_pred = np.max(image_ * y_pred_)
        max_intensity_gt = np.max(image_ * y_)

        # Check intensity threshold
        if abs(max_intensity_pred - max_intensity_gt) <= intensity_threshold:
            # True Positive if intensity condition is met and overlap exists
            if np.sum(y_ * y_pred_) > 0:
                TP_count += 1
        else:
            # False Positive if prediction exists but no sufficient intensity agreement
            if np.sum(y_pred_) > 0:
                FP_count += 1
            # False Negative if ground truth exists but no sufficient agreement
            if np.sum(y_) > 0:
                FN_count += 1


    return TP_count, FP_count, FN_count

def TPFPFN_with_same_max(y_pred, y, image, intensity_threshold=0.1):

    # Binarize the predictions
    y_pred = np.where(y_pred < 0.5, 0, 1)

    y_copy = copy.deepcopy(y).squeeze()
    y_pred_copy = copy.deepcopy(y_pred).squeeze()
    image_copy = copy.deepcopy(image).squeeze()

    # Add batch dimension if missing
    if y_copy.ndim == 3:  # if batch dim is reduced
        y_copy = y_copy[np.newaxis, ...]
        y_pred_copy = y_pred_copy[np.newaxis, ...]
        image_copy = image_copy[np.newaxis, ...]

    TP_count = 0
    FP_count = 0
    FN_count = 0

    # Iterate over the batch
    for ii in range(y_copy.shape[0]):
        y_ = y_copy[ii]
        y_pred_ = y_pred_copy[ii]
        image_ = image_copy[ii]

        # Label connected components in ground truth
        gt_labels, num_gt = label(y_)
        # Label connected components in prediction
        pred_labels, num_pred = label(y_pred_)

        matched_gt = set()
        matched_pred = set()

        # Iterate over ground truth components
        for gt_idx in range(1, num_gt + 1):
            gt_component = (gt_labels == gt_idx)
            max_intensity_gt = np.max(image_ * gt_component)

            # Find matching prediction components
            overlap = gt_component * y_pred_
            if overlap.sum() > 0:
                # Get the labels of overlapping prediction components
                pred_idxs = np.unique(pred_labels * gt_component)
                pred_idxs = pred_idxs[pred_idxs != 0]  # Exclude background

                for pred_idx in pred_idxs:
                    if pred_idx in matched_pred:
                        continue  # Already matched

                    pred_component = (pred_labels == pred_idx)
                    max_intensity_pred = np.max(image_ * pred_component)

                    # Check intensity threshold
                    if abs(max_intensity_pred - max_intensity_gt) <= intensity_threshold:
                        TP_count += 1
                        matched_gt.add(gt_idx)
                        matched_pred.add(pred_idx)
                        break
                else:
                    continue  # No matching prediction component met the criteria
            else:
                # No overlap with prediction
                continue

        # Count false negatives (ground truth components not matched)
        FN_count += (num_gt - len(matched_gt))
        # Count false positives (prediction components not matched)
        FP_count += (num_pred - len(matched_pred))

    return TP_count, FP_count, FN_count

def analyze_volume(volume):
    # Threshold the volume
    thresholded_volume = np.where(volume > 0.5, 1, 0)

    # Compute connected components
    connectivity = 26  # You can choose 6, 18, or 26 for 3D connectivity
    components = cc3d.connected_components(thresholded_volume, connectivity=connectivity)

    # Find maximum value in the original volume for each component
    max_values = {}
    for component_id in np.unique(components):
        if component_id == 0:
            continue  # Skip the background component
        component_mask = components == component_id
        max_value = np.max(volume[component_mask])
        max_values[component_id] = max_value
        print(f"Max value for component {component_id}: {max_value}")


def analyze_and_filter_volume(volume):
    # Threshold the volume
    thresholded_volume = np.where(volume > 0.5, 1, 0)

    # Compute connected components
    connectivity = 26  # You can choose 6, 18, or 26 for 3D connectivity
    components = cc3d.connected_components(thresholded_volume, connectivity=connectivity)

    # Find the component with the maximum value in the original volume
    max_component_id = None
    max_value = -np.inf  # Start with the lowest possible value
    for component_id in np.unique(components):
        if component_id == 0:
            continue  # Skip the background component
        component_mask = components == component_id
        component_max_value = np.max(volume[component_mask])
        if component_max_value > max_value:
            max_value = component_max_value
            max_component_id = component_id

    # Create a new volume where only the component with the highest max is retained
    filtered_volume = np.zeros_like(volume)
    if max_component_id is not None:
        filtered_volume[components == max_component_id] = volume[components == max_component_id]

    return filtered_volume


def calculate_f1_score(true_positives, false_positives, false_negatives):
    # Calculate precision
    if true_positives + false_positives == 0:
        precision = 0  # Prevent division by zero
    else:
        precision = true_positives / (true_positives + false_positives)

    # Calculate recall
    if true_positives + false_negatives == 0:
        recall = 0  # Prevent division by zero
    else:
        recall = true_positives / (true_positives + false_negatives)

    # Calculate F1 score
    if precision + recall == 0:
        f1_score = 0  # Prevent division by zero
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def filter_prediction_by_average(volume):
    # Threshold the volume
    print(f"volume shape: {volume.shape}")
    thresholded_volume = np.where(volume > 0.5, 1, 0)

    # Compute connected components
    connectivity = 26  # You can choose 6, 18, or 26 for 3D connectivity
    components = cc3d.connected_components(thresholded_volume, connectivity=connectivity)

    # Find the component with the highest average value in the original volume
    max_avg_component_id = None
    max_average = -np.inf  # Start with the lowest possible value
    for component_id in np.unique(components):
        if component_id == 0:
            continue  # Skip the background component
        component_mask = components == component_id
        component_values = volume[component_mask]
        component_average = np.mean(component_values)
        if component_average > max_average:
            max_average = component_average
            max_avg_component_id = component_id

    # Create a new volume where only the component with the highest average is retained
    filtered_volume = np.zeros_like(volume)
    if max_avg_component_id is not None:
        filtered_volume[components == max_avg_component_id] = volume[components == max_avg_component_id]

    return filtered_volume
import regex as re
def remove_leading_number(string):
    # Find the position where the first non-digit character appears
    index = 0
    while index < len(string) and string[index].isdigit():
        index += 1
    # Slice the string from the first non-digit character
    modified_string = string[index:]
    return modified_string

def post_processing_eval():
    json_file_path = "/UserData/Zach_Analysis/uw_lymphoma_pet_3d/final_training_testing_v6.json"
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.5_data_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.1_data_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/f1_.76_dice_.55_best_prediction/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_roberta_large_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_roberta_large_predictions_v4/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_embeddings_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/25d_predictions_v2/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/llmseg_full_data_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/25d_predictions_v4/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_precomputed_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_empty_string_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_bert_predictions_v2/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_data_predictions/"
    prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/swedish_predictions_contextual_net/"


    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
    label_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels6/"
    #image_base = "/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/images/"
    #label_base = "/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/labels/"

    image_base = "/mnt/Bradshaw/UW_PET_Data/physican_labels/final_external_dataset_v2/images/"
    label_base = "/mnt/Bradshaw/UW_PET_Data/physican_labels/final_external_dataset_v2/labels/"

    tracer_df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/meta_data_files/combined_tracer_and_scanner.xlsx")

    labeled_subset = pd.read_excel("/UserData/Zach_Analysis/final_testset_evaluation_vg/all_labels_jdw.xlsx")
    #labeled_subset = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Steve_worksheet_matched.xlsx")


    prediction_list = os.listdir(prediction_location)
    all_images = os.listdir(image_base)
    number_correct = 0
    index = 0
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0

    fdg_tp_sum = 0
    fdg_fp_sum = 0
    fdg_fn_sum = 0

    psma_tp_sum = 0
    psma_fp_sum = 0
    psma_fn_sum= 0

    dice_scores = []

    TP_sum_thresh = 0
    FP_sum_thresh = 0
    FN_sum_thresh = 0

    TP_sum_max = 0
    FP_sum_max = 0
    FN_sum_max = 0

    # Initialize lists to store per-sample metrics for bootstrap resampling
    bootstrap_data = {
        "label_name": [], # name of sample
        "pixel_size": [], # number of positive pixels in label
        "tracer": [], # tracer type
        "machine": [], # name of imaging machine
        "dice_scores": [],  # Dice scores per sample
        "TP_FP_FN": [],  # Combined TP, FP, FN per sample
        "TP_FP_FN_thresh": [],  # TP, FP, FN per sample for threshold F1
        "TP_FP_FN_max": [],  # TP, FP, FN per sample for max F1
    }

    skipped_labels = {"PETWB_002624_01_label_1", "PETWB_017530_01_label_2", "PETWB_011869_01_label_1", "PETWB_011768_01_label_4"}
    skipped = 0
    for label in prediction_list:

        index += 1
        #if number_correct > 1:
        #    print(f"index: {index} number that are correct: {number_correct} accuracy: {number_correct / index} TP: {TP_sum} FP: {FP_sum} FN: {FN_sum}")
        #else:
        #    print(f"index: {index} number that are correct: {number_correct}")
        print(f"index: {index} TP: {TP_sum} FP: {FP_sum} FN: {FN_sum} f1 score: {calculate_f1_score(TP_sum, FP_sum, FN_sum)} skipped: {skipped}")
        #print(f"label name: {label}")
        # image_name = label[:-15]
        # print(label)
        #label = remove_leading_number(label)
        # print(label)
        #image_name = label[:15] # for internal datset
        image_name = label[:12] # for swedish dataset

        # print(image_name)
        #print(f"image name: {image_name}")
        #label_name = label.strip(".nii.gz")
        label_name = label.strip(".gz")
        label_name = label_name.strip(".nii")

        #label_name = label.strip("_label_.nii")
        print(f"label name: {label_name}")
        if label_name in skipped_labels:
            continue
        bootstrap_data["label_name"].append(label_name)
        petlymph_name = image_name.strip(".nii.gz")
        # print(petlymph_name)
        # print(f"label name: {label_name}")

        """
        labeled_row = labeled_subset[labeled_subset["Label_Name"] == label_name]

        # Check if labeled_row is empty or it is a bad label
        if labeled_row.empty:
            print("skipped in empty row")
            continue
        """

        #if labeled_row["Label_is_Correct"].iloc[0] == 0:
        #    skipped += 1
        #    continue

        # Get the row where 'ID' matches petlymph_name
        #tracer_row = tracer_df[tracer_df["ID"] == petlymph_name]
        # Get the value from the 'Tracer' column or set tracer to None if not found
        #tracer = tracer_row["Tracer"].values[0] if not tracer_row.empty else None
        # get the value of the machine type
        #machine = tracer_row["Scanner Type"].values[0] if not tracer_row.empty else None

        tracer = "none"
        machine = "none"
        bootstrap_data["tracer"].append(tracer)
        bootstrap_data["machine"].append(machine)

        #print(f"tracer : {tracer}")
        #continue

        # print(label_name)
        # row = df[df["Label_Name"] == label_name].iloc[0]
        # sent = row["sentence"]
        # print(sent)
        #for entry in data["testing"]:
        #    if label_name in entry.get('label'):
        #        sent = entry.get('report')  # Return the report if label name matches
        #print(sent)
        suv_path_final = os.path.join(image_base, image_name + "_suv_cropped.nii.gz")
        #print(suv_path_final)
        ct_path_final = os.path.join(image_base, image_name + "_ct_cropped.nii.gz")
        full_pred_path = os.path.join(prediction_location, label)
        if ".gz" not in label:
            label += ".gz"
        label_full_path = os.path.join(label_base, label) # oringinal
        #label_full_path = os.path.join(label_base, "label_" + label) # changed for 2.5 d

        #print(label)
        # load in the suv data
        nii_suv = nib.load(suv_path_final)
        pet_image = nii_suv.get_fdata()

        # load in the ct data
        #nii_ct = nib.load(ct_path_final)
        #ct_data = nii_ct.get_fdata()
        # load in the prediciton data
        nii_prediction = nib.load(full_pred_path)
        prediction_data = nii_prediction.get_fdata()

        #print(f"initial prediction sum: {np.sum(prediction_data)}")

        prediction_data = np.squeeze(prediction_data, axis=(0, 1))         # add this back in later

        #prediction_data = np.where(prediction_data >= 0.5, 1, 0)
        #print(f"pred data size: {prediction_data.shape}")
        #prediction_data = analyze_and_filter_volume(prediction_data)
        prediction_data = filter_prediction_by_average(prediction_data)

        # load in label data
        nii_label = nib.load(label_full_path)
        label_data = nii_label.get_fdata()
        #pet_image = pad_image_to_shape(pet_image)
        #prediction_data = adjust_volume_shape(prediction_data, label_data)
        #prediction_data = resize_3d_prediction(prediction_data, label_data.shape)
        #label_data = resize_3d_prediction(label_data, prediction_data.shape)
        #pet_image = resize_3d_prediction(pet_image, prediction_data.shape)
        #print(f"label size: {label_data.shape}")
        #print(f"prediction_data shape: {prediction_data.shape}")
        #print(f"prediction sum: {np.sum(prediction_data)}")
        #print(f"label sum: {np.sum(label_data)}")
        #prediction_data = filter_prediction_by_average(prediction_data)
        #print(f"prediction sum: {np.sum(prediction_data)}")

        # Sum up all the 1's in the label data
        sum_of_ones = np.sum(label_data == 1)
        bootstrap_data["pixel_size"].append(sum_of_ones)

        #prediction_data = pad_and_crop(prediction_data, label_data)
        TP, FP, FN = TPFPFNHelper(prediction_data, label_data)
        TP_sum += TP
        FP_sum += FP
        FN_sum += FN

        dice_score = dice_score_helper(prediction_data, label_data)
        dice_scores.extend(dice_score)
        print(f"dice score: {dice_score}")

        TP_thresh, FP_thresh, FN_thresh = TPFPFN_with_dice_threshold(prediction_data, label_data)
        TP_sum_thresh += TP_thresh
        FP_sum_thresh += FP_thresh
        FN_sum_thresh += FN_thresh

        TP_max, FP_max, FN_max = TPFPFN_with_same_max(prediction_data, label_data, pet_image, intensity_threshold=0.1)
        TP_sum_max += TP_max
        FP_sum_max += FP_max
        FN_sum_max += FN_max

        if tracer == "FDG -- fluorodeoxyglucose":
            fdg_tp_sum += TP
            fdg_fp_sum += FP
            fdg_fn_sum += FN
        elif tracer == "Sodium Flouride" or tracer == "GA68 Dotatate":
            print("other tracer")
        else:
            psma_tp_sum += TP
            psma_fp_sum += FP
            psma_fn_sum += FN

        # Add the computed metrics for this sample to the bootstrap data
        bootstrap_data["dice_scores"].extend(dice_score)  # Append dice scores (list or single value)

        # Append TP, FP, FN as a tuple for combined F1 score
        bootstrap_data["TP_FP_FN"].append((TP, FP, FN))

        # Append TP, FP, FN for threshold F1
        bootstrap_data["TP_FP_FN_thresh"].append((TP_thresh, FP_thresh, FN_thresh))

        # Append TP, FP, FN for max F1
        bootstrap_data["TP_FP_FN_max"].append((TP_max, FP_max, FN_max))

    print(f"Combined f1 score: {calculate_f1_score(TP_sum, FP_sum, FN_sum)}")
    print(f"Combined True positive: {TP_sum} False Positive: {FP_sum} False Negative sum: {FN_sum}")

    print(f"fdg f1 score: {calculate_f1_score(fdg_tp_sum, fdg_fp_sum, fdg_fn_sum)}")
    print(f"fdg True positive: {fdg_tp_sum} False Positive: {fdg_fp_sum} False Negative sum: {fdg_fn_sum}")

    print(f"psma f1 score: {calculate_f1_score(psma_tp_sum, psma_fp_sum, psma_fn_sum)}")
    print(f"psma True positive: {psma_tp_sum} False Positive: {psma_fp_sum} False Negative sum: {psma_fn_sum}")

    print(f"final dice over all samples: {np.mean(dice_scores)}")

    print(f"threshold TP: {TP_sum_thresh} FP: {FP_sum_thresh} FN: {FN_sum_thresh}")
    print(f"combined threshold f1 score:{calculate_f1_score(TP_sum_thresh, FP_sum_thresh, FN_sum_thresh)}")

    print(f"max TP: {TP_sum_max} FP: {FP_sum_max} FN: {FN_sum_max}")
    print(f"combined max f1 score:{calculate_f1_score(TP_sum_max, FP_sum_max, FN_sum_max)}")


    # Save bootstrap_data for later resampling
    np.save("/UserData/Zach_Analysis/final_3d_models_used_in_paper/data_predictions/swedish_external_contextual_net.npy", bootstrap_data) # rerun bootstrap_data_contextual_net_full_test_data

    #np.save("/UserData/Zach_Analysis/final_3d_models_used_in_paper/data_predictions/contextual_net_.1_data.npy", bootstrap_data) # rerun bootstrap_data_contextual_net_full_test_data
    print("Bootstrap data saved for resampling.")