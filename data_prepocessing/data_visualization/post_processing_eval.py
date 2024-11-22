
import copy
import cc3d
import os
import nibabel as nib
import numpy as np

import json

import pandas as pd


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

    prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/.1_data_predictions/"
    #prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/f1_.76_dice_.55_best_prediction/"

    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
    label_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels6/"

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

        # Get the row where 'ID' matches petlymph_name
        tracer_row = tracer_df[tracer_df["ID"] == petlymph_name]

        # Get the value from the 'Tracer' column or set tracer to None if not found
        tracer = tracer_row["Tracer"].values[0] if not tracer_row.empty else None
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
        label_full_path = os.path.join(label_base, label)
        #print(label)
        # load in the suv data
        #nii_suv = nib.load(suv_path_final)
        #suv_data = nii_suv.get_fdata()
        # load in the ct data
        #nii_ct = nib.load(ct_path_final)
        #ct_data = nii_ct.get_fdata()
        # load in the prediciton data
        nii_prediction = nib.load(full_pred_path)
        prediction_data = nii_prediction.get_fdata()
        prediction_data = np.squeeze(prediction_data, axis=(0, 1))
        #print(f"pred data size: {prediction_data.shape}")
        #prediction_data = analyze_and_filter_volume(prediction_data)
        prediction_data = filter_prediction_by_average(prediction_data)
        # load in label data
        nii_label = nib.load(label_full_path)
        label_data = nii_label.get_fdata()

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

    print(f"Combined f1 score: {calculate_f1_score(TP_sum, FP_sum, FN_sum)}")
    print(f"Combined True positive: {TP_sum} False Positive: {FP_sum} False Negative sum: {FN_sum}")

    print(f"fdg f1 score: {calculate_f1_score(fdg_tp_sum, fdg_fp_sum, fdg_fn_sum)}")
    print(f"fdg True positive: {fdg_tp_sum} False Positive: {fdg_fp_sum} False Negative sum: {fdg_fn_sum}")

    print(f"psma f1 score: {calculate_f1_score(psma_tp_sum, psma_fp_sum, psma_fn_sum)}")
    print(f"psma True positive: {psma_tp_sum} False Positive: {psma_fp_sum} False Negative sum: {psma_fn_sum}")

    print(f"final dice over all samples: {np.mean(dice_scores)}")

    print(f"threshold TP: {TP_sum_thresh} TP: {FP_sum_thresh} TN: {FN_sum_thresh}")
    print(f"combined threshold f1 score:{calculate_f1_score(TP_sum_thresh, FP_sum_thresh, FN_sum_thresh)}")