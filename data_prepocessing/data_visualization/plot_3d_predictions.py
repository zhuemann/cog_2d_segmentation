import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import json
import cc3d
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



def plot_3d_predictions():

    df = pd.read_excel("/UserData/Zach_Analysis/petlymph_image_data/" + "uw_final_df_9.xlsx")
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
    for label in prediction_list:
        index += 1
        if number_correct > 1:
            print(f"index: {index} number that are correct: {number_correct} accuracy: {number_correct/index}")
        else:
            print(f"index: {index} number that are correct: {number_correct}")

        print(f"label name: {label}")
        #image_name = label[:-15]
        image_name = label[:15]
        print(f"image name: {image_name}")
        label_name = label.strip(".nii.gz")
        #print(label_name)
        #row = df[df["Label_Name"] == label_name].iloc[0]
        #sent = row["sentence"]
        #print(sent)
        for entry in data["testing"]:
            if label_name in entry.get('label'):
                sent = entry.get('report')  # Return the report if label name matches
        print(sent)
        suv_path_final = os.path.join(image_base, image_name + "_suv_cropped.nii.gz")
        print(suv_path_final)
        ct_path_final = os.path.join(image_base, image_name + "_ct_cropped.nii.gz")
        full_pred_path = os.path.join(prediction_location, label)
        label_full_path = os.path.join(label_base, label)

        # load in the suv data
        nii_suv = nib.load(suv_path_final)
        suv_data = nii_suv.get_fdata()
        # load in the ct data
        nii_ct = nib.load(ct_path_final)
        ct_data = nii_ct.get_fdata()
        # load in the prediciton data
        nii_prediction = nib.load(full_pred_path)
        prediction_data = nii_prediction.get_fdata()
        prediction_data = np.squeeze(prediction_data, axis=(0, 1))
        print(f"pred data size: {prediction_data.shape}")

        # load in label data
        nii_label = nib.load(label_full_path)
        label_data = nii_label.get_fdata()

        # Compute maximum intensity projection along axis 1
        suv_mip = np.max(suv_data, axis=1)
        prediction_mip = np.max(prediction_data, axis=1)
        label_mip = np.max(label_data, axis=1)

        label_suv_max = max_suv_in_positive_region(suv_data, label_data)
        prediction_suv_max = max_suv_in_positive_region(suv_data, prediction_data)
        correct = False
        if label_suv_max == prediction_suv_max:
            correct = True
            number_correct += 1

        prediction_components = np.max(cc3d.connected_components(prediction_data, connectivity=26))
        print(f"number of componets in prediction: {prediction_components}")
        continue
        #print(f"suv mip size: {suv_mip.shape}")
        #print(f"pred mip size: {prediction_mip.shape}")
        #print(f"label mip size: {label_mip.shape}")

        norm = Normalize(vmin=0.01, clip=True)  # vmin set slightly above zero to make zeros transparent

        # Setup the plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Label MIP overlayed on SUV MIP
        axes[0].imshow(suv_mip.T, cmap='gray_r', aspect='auto', origin='lower', vmin = 0, vmax = 10)
        axes[0].imshow(label_mip.T, cmap='Greens', alpha=norm(label_mip.T), aspect='auto', origin='lower')
        axes[0].set_title(f'Label Overlay on SUV MIP suv_max: {label_suv_max}')
        axes[0].axis('off')  # Turn off axis

        # Plot 2: Prediction MIP overlayed on SUV MIP
        axes[1].imshow(suv_mip.T, cmap='gray_r', aspect='auto', origin='lower', vmin = 0, vmax = 10)
        axes[1].imshow(prediction_mip.T, cmap='Blues', alpha=norm(prediction_mip.T), aspect='auto', origin='lower')
        axes[1].set_title(f'Prediction Overlay on SUV MIP predicted suv_max: {prediction_suv_max}')
        axes[1].axis('off')

        # Plot 3: Both Prediction and Label MIP overlayed on SUV MIP
        axes[2].imshow(suv_mip.T, cmap='gray_r', aspect='auto', origin='lower', vmin = 0, vmax = 10)
        axes[2].imshow(label_mip.T, cmap='Greens', alpha=norm(label_mip.T), aspect='auto', origin='lower')
        axes[2].imshow(prediction_mip.T, cmap='Blues', alpha=norm(prediction_mip.T), aspect='auto', origin='lower')
        axes[2].set_title(f'Prediction and Label Overlay on SUV MIP is correct: {correct}')
        axes[2].axis('off')

        sent = insert_newlines(sent, word_limit=25)
        fig.suptitle(sent, fontsize=14)

        # Save the figure
        #plt.tight_layout()
        #plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/3d_predictions_v1/" + label_name + ".png")
        plt.close()
