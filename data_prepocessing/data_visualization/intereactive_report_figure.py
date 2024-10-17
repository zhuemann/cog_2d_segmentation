import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import json

import os
import numpy as np
import scipy.ndimage

def max_suv_in_positive_region(suv_volume, label_volume):
    """
    Load SUV and label volumes, compute the maximum SUV value where label is 1.

    Args:
    - suv_path (str): Path to the SUV .nii.gz file.
    - label_path (str): Path to the label .nii.gz file, where 1 indicates a positive label.

    Returns:
    - float: The maximum SUV value in regions marked as positive by the label.
    """

    # Element-wise multiplication to mask out negative regions
    positive_suv = suv_volume * (label_volume == 1)

    # Retrieve the maximum SUV value in the masked volume
    max_suv_value = np.max(positive_suv)

    return max_suv_value

def resample_img_size(volume, target):
    # Remove singleton dimensions
    volume = np.squeeze(volume)  # volume is now shape (192, 192, 352)
    target_shape = target.shape
    # Resample to target shape
    resampled_volume = scipy.ndimage.zoom(volume, (target_shape[0] / volume.shape[0],
                                               target_shape[1] / volume.shape[1],
                                               target_shape[2] / volume.shape[2]), order=1)  # Linear interpolation
    return resampled_volume

def insert_newlines(text, word_limit=15):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        if len(current_line) == word_limit:
            lines.append(' '.join(current_line))
            current_line = []

    # Add the last line if there are any remaining words
    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)

def make_interactive_figure():

    # Load JSON file
    with open('/UserData/Zach_Analysis/uw_lymphoma_pet_3d/data_for_making_interactive_figure.json', 'r') as file:
        figure_data = json.load(file)

    for index in range(1, 10):
        base_file_path = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/interactive_report_figure/"
        # print(sent)

        image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
        #image_base = "/mnt/PURENFS/Bradshaw/UW_PET_Data/SUV_images/PETWB_001516_02/"
        ct_path_final = os.path.join(image_base, "PETWB_001516_02_ct_cropped.nii.gz ")
        # print(suv_path_final)
        suv_path_final = os.path.join(image_base, "PETWB_001516_02_suv_cropped.nii.gz")
        # full_pred_path = os.path.join(prediction_location, label)
        # label_full_path = os.path.join(label_base, label)

        # load in the suv data
        nii_suv = nib.load(suv_path_final)
        suv_data = nii_suv.get_fdata()
        # load in the ct data
        nii_ct = nib.load(ct_path_final)
        # ct_data = nii_ct.get_fdata()
        # load in the prediciton data
        label_final_path = os.path.join(base_file_path, str(index) + "PETWB_001516_02_label_.nii")
        nii_prediction = nib.load(label_final_path)
        prediction_data = nii_prediction.get_fdata()

        prediction_data = np.squeeze(prediction_data, axis=(0, 1))
        # print(f"pred data size: {prediction_data.shape}")
        suv_data = resample_img_size(suv_data, prediction_data)
        # load in label data
        # nii_label = nib.load(label_full_path + ".gz")
        # label_data = nii_label.get_fdata()

        # Compute maximum intensity projection along axis 1
        suv_mip = np.max(suv_data, axis=1)
        prediction_data = np.where(prediction_data < 0.5, 0, 1)
        prediction_mip = np.max(prediction_data, axis=1)


        # label_suv_max = max_suv_in_positive_region(suv_data, label_data)
        prediction_suv_max = max_suv_in_positive_region(suv_data, prediction_data)
        # correct = False
        # if label_suv_max == prediction_suv_max:
        #    correct = True
        #    number_correct += 1

        # prediction_components = np.max(cc3d.connected_components(label_data, connectivity=26))
        # print(f"number of componets in prediction: {prediction_components}")
        # continue
        # print(f"suv mip size: {suv_mip.shape}")
        # print(f"pred mip size: {prediction_mip.shape}")
        # print(f"label mip size: {label_mip.shape}")

        norm = Normalize(vmin=0.01, clip=True)  # vmin set slightly above zero to make zeros transparent

        # Setup the plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Reflect the data horizontally so the heart is on the left

        # Rotate the images 90 degrees counterclockwise
        suv_mip = np.rot90(suv_mip, k=3)
        # label_mip = np.rot90(label_mip, k=3)
        prediction_mip = np.rot90(prediction_mip, k=3)

        suv_mip = np.fliplr(suv_mip)
        # label_mip = np.fliplr(label_mip)
        prediction_mip = np.fliplr(prediction_mip)

        # Plot 1: Label MIP overlayed on SUV MIP
        axes[0].imshow(suv_mip, cmap='gray_r', aspect='auto', origin='lower', vmin=0, vmax=10)
        # axes[0].imshow(label_mip, alpha=norm(label_mip), aspect='auto', origin='lower')
        # axes[0].set_title(f'Label Overlay on SUV MIP suv_max: {label_suv_max:.3f}')
        axes[0].axis('off')  # Turn off axis

        # Plot 2: Prediction MIP overlayed on SUV MIP
        axes[1].imshow(suv_mip, cmap='gray_r', aspect='auto', origin='lower', vmin=0, vmax=10)
        axes[1].imshow(prediction_mip, cmap="cool", alpha=norm(prediction_mip), aspect='auto', origin='lower')
        axes[1].set_title(f'Prediction Overlay on SUV MIP predicted suv_max: {prediction_suv_max:.3f}')
        axes[1].axis('off')

        # Plot 3: Both Prediction and Label MIP overlayed on SUV MIP
        axes[2].imshow(suv_mip, cmap='gray_r', aspect='auto', origin='lower', vmin=0, vmax=10)
        # axes[2].imshow(label_mip, alpha=norm(label_mip), aspect='auto', origin='lower')
        axes[2].imshow(prediction_mip, cmap="cool", alpha=norm(prediction_mip), aspect='auto', origin='lower')
        # axes[2].set_title(f'Prediction and Label Overlay on SUV MIP is correct: {correct}')
        axes[2].axis('off')

        sent = figure_data[index]["report"]
        sent = insert_newlines(sent, word_limit=25)
        fig.suptitle(sent, fontsize=14)

        plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/interactive_report_figure/" + str(index) + ".png")
        plt.close()