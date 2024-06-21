
import copy
import cc3d
import os
import nibabel as nib
import numpy as np

import json

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
    for idx in range(1, pred_conn_comp.max() + 1): # min(pred_conn_comp.max() + 1, 50)):
        comp_mask = np.isin(pred_conn_comp, idx)
        #if comp_mask.sum() <= 8:  # ignore small connected components (0.64 ml)
        #    print("less than 8")
        #    continue
        if comp_mask.sum() <= 3:  # ignore small connected components (0.81 ml)
            print("less than 3")
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
    for idx in range(1, gt_conn_comp.max()): #min(gt_conn_comp.max() + 1, 50)):
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
        print(y_pred.shape)
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

def pos_processing_eval():
    json_file_path = "/UserData/Zach_Analysis/uw_lymphoma_pet_3d/output_resampled.json"
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_forked2/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/prediction_trash_v2testing/"

    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images/"
    label_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels/"

    prediction_list = os.listdir(prediction_location)
    all_images = os.listdir(image_base)
    number_correct = 0
    index = 0
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    for label in prediction_list:
        index += 1
        if number_correct > 1:
            print(f"index: {index} number that are correct: {number_correct} accuracy: {number_correct / index} TP: {TP_sum} FP: {FP_sum} FN: {FN_sum}")
        #else:
        #    print(f"index: {index} number that are correct: {number_correct}")
        print(
            f"index: {index} TP: {TP_sum} FP: {FP_sum} FN: {FN_sum}")
        #print(f"label name: {label}")
        # image_name = label[:-15]
        image_name = label[:15]
        #print(f"image name: {image_name}")
        label_name = label.strip(".nii.gz")
        # print(label_name)
        # row = df[df["Label_Name"] == label_name].iloc[0]
        # sent = row["sentence"]
        # print(sent)
        for entry in data["testing"]:
            if label_name in entry.get('label'):
                sent = entry.get('report')  # Return the report if label name matches
        #print(sent)
        suv_path_final = os.path.join(image_base, image_name + "_suv_cropped.nii.gz")
        #print(suv_path_final)
        ct_path_final = os.path.join(image_base, image_name + "_ct_cropped.nii.gz")
        full_pred_path = os.path.join(prediction_location, label)
        label_full_path = os.path.join(label_base, label)

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

        # load in label data
        nii_label = nib.load(label_full_path)
        label_data = nii_label.get_fdata()

        TP, FP, FN = TPFPFNHelper(prediction_data, label_data)
        TP_sum += TP
        FP_sum += FP
        FN_sum += FN

    print(f"True positive: {TP_sum} False Positive: {FP_sum} False Negative sum: {FN}")