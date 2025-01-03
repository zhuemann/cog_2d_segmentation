
import copy
import cc3d
import os
import nibabel as nib
import numpy as np

import json

import pandas as pd
from scipy.ndimage import label



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
        if comp_mask.sum() <= 3:  # ignore small connected components (0.81 ml)
            #print("less than 3")
            continue
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

def physician_post_processing_eval():
    #json_file_path = "/UserData/Zach_Analysis/uw_lymphoma_pet_3d/final_training_testing_v6.json"
    #with open(json_file_path, 'r') as file:
    #    data = json.load(file)

    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.5_data_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/f1_.76_dice_.55_best_prediction/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_roberta_large_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_roberta_large_predictions_v4/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.25_embeddings_predictions/"
    #prediction_location = "/mnt/Bradshaw/UW_PET_Data/physican_labels/steve_labels/"
    prediction_location = "/mnt/Bradshaw/UW_PET_Data/physican_labels/meghan_labels/"

    #image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
    #label_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels6/"
    image_base = "/mnt/Bradshaw/UW_PET_Data//SUV_images/"
    label_base = "/mnt/Bradshaw/UW_PET_Data//raw_nifti_uw_pet/uw_labels_v4_nifti/"

    tracer_df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/meta_data_files/combined_tracer_and_scanner.xlsx")

    labeled_subset = pd.read_excel("/UserData/Zach_Analysis/final_testset_evaluation_vg/all_labels_jdw.xlsx")

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
        image_name = label[:15]
        print(f"image_name: {image_name}")

        # print(image_name)
        #print(f"image name: {image_name}")
        label_name = label.strip(".nii.gz")
        #label_name = label.strip("_label_.nii")

        petlymph_name = image_name.strip(".nii.gz")
        # print(petlymph_name)
        # print(f"label name: {label_name}")
        labeled_row = labeled_subset[labeled_subset["Label_Name"] == label_name]



        # Check if labeled_row is empty or it is a bad label
        if labeled_row.empty:
            print("skipped in empty row")
            continue

        if labeled_row["Label_is_Correct"].iloc[0] == 0:
            skipped += 1
            continue
        bootstrap_data["label_name"].append(label_name)

        # Get the row where 'ID' matches petlymph_name
        tracer_row = tracer_df[tracer_df["ID"] == petlymph_name]

        # Get the value from the 'Tracer' column or set tracer to None if not found
        tracer = tracer_row["Tracer"].values[0] if not tracer_row.empty else None
        #print(f"tracer : {tracer}")
        #continue
        machine = tracer_row["Scanner Type"].values[0] if not tracer_row.empty else None
        bootstrap_data["tracer"].append(tracer)
        bootstrap_data["machine"].append(machine)

        # print(label_name)
        # row = df[df["Label_Name"] == label_name].iloc[0]
        # sent = row["sentence"]
        # print(sent)
        #for entry in data["testing"]:
        #    if label_name in entry.get('label'):
        #        sent = entry.get('report')  # Return the report if label name matches
        #print(sent)
        image_path_base = os.path.join(image_base, image_name) # ,image_name + "_suv_cropped.nii.gz")
        file_names = os.listdir(image_path_base)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        suv_path_final = os.path.join(image_path_base, file_names[index_of_suv[0]])
        #print(suv_path_final)
        #ct_path_final = os.path.join(image_base, image_name + "_ct_cropped.nii.gz")
        if ".nii" not in label:
            label += ".nii"
        if ".gz" not in label:
            label += ".gz"
        full_pred_path = os.path.join(prediction_location, label)

        print(f"image_name: {image_name}")
        label_full_path = os.path.join(label_base, label)
        print(f"label full path: {label_full_path} and just label: {label}")

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
        #print(prediction_data.shape)
        #prediction_data = np.squeeze(prediction_data, axis=(0, 1))
        #print(f"pred data size: {prediction_data.shape}")
        #prediction_data = analyze_and_filter_volume(prediction_data)
        prediction_data = filter_prediction_by_average(prediction_data)
        # load in label data
        nii_label = nib.load(label_full_path)
        label_data = nii_label.get_fdata()

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

    np.save("/UserData/Zach_Analysis/final_3d_models_used_in_paper/data_predictions/meg_eval.npy", bootstrap_data)