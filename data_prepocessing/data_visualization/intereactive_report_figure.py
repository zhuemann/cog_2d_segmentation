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
import cc3d

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


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
    with open('/UserData/Zach_Analysis/uw_lymphoma_pet_3d/data_for_making_interactive_figure_v3.json', 'r') as file:
        figure_data = json.load(file)
    figure_data = figure_data["testing"]
    good_index = [12,16, 28, 32, 33, 41, 42, 43, 44] # maybe 25
    for index in range(0, 70):
        print(f"index: {index}")
        base_file_path = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/interactive_report_figure_v3/"
        # print(sent)

        image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
        #image_base = "/mnt/PURENFS/Bradshaw/UW_PET_Data/SUV_images/PETWB_001516_02/"
        ct_path_final = os.path.join(image_base, "PETWB_012541_01_ct_cropped.nii.gz")
        # print(suv_path_final)
        suv_path_final = os.path.join(image_base, "PETWB_012541_01_suv_cropped.nii.gz")
        # full_pred_path = os.path.join(prediction_location, label)
        # label_full_path = os.path.join(label_base, label)

        # load in the suv data
        nii_suv = nib.load(suv_path_final)
        suv_data = nii_suv.get_fdata()
        # load in the ct data
        nii_ct = nib.load(ct_path_final)
        # ct_data = nii_ct.get_fdata()
        # load in the prediciton data
        #label_final_path = os.path.join(base_file_path, str(index) + "PETWB_012541_01_label_.nii")
        label_final_path = os.path.join(base_file_path, "sentence_" + str(index) + ".nii")

        nii_prediction = nib.load(label_final_path)
        prediction_data = nii_prediction.get_fdata()

        prediction_data = np.squeeze(prediction_data, axis=(0, 1))
        # print(f"pred data size: {prediction_data.shape}")
        prediction_data = filter_prediction_by_average(prediction_data)
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

        #suv_mip = np.fliplr(suv_mip)
        # label_mip = np.fliplr(label_mip)
        #prediction_mip = np.fliplr(prediction_mip)
        if (prediction_mip.min() == prediction_mip.max()):
            continue
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

        plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/interactive_report_figure_v3/" + str(index) + ".png")
        plt.close()

def compound_interactive_report_v2():
    # Assuming you have the functions filter_prediction_by_average and resample_img_size defined

    # Define the list of good indices
    #good_index = [12, 16, 28, 32, 33, 41, 43, 44]  # maybe 25
    good_index = [19, 22, 25, 26, 27, 29, 30, 31, 33, 60]
    # Base file paths
    base_file_path = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/paper_predictions/interactive_report_figure_v2/"
    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"

    # Load the SUV data (Assuming SUV data is common for all indices)
    suv_path_final = os.path.join(image_base, "PETWB_012541_01_suv_cropped.nii.gz")
    #suv_path_final = os.path.join(image_base, "PETWB_001516_02_ct_cropped.nii.gz")

    nii_suv = nib.load(suv_path_final)
    suv_data = nii_suv.get_fdata()

    # Compute the Maximum Intensity Projection (MIP) for SUV
    suv_mip = np.max(suv_data, axis=1)
    #suv_mip = suv_data[:,90,:]
    suv_mip = np.rot90(suv_mip, k=3)  # Rotate the SUV MIP

    # Setup the plot with 1 main plot for SUV MIP
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(suv_mip, cmap='gray_r', aspect='auto', origin='lower', vmin=0, vmax=6)
    #ax.imshow(suv_mip, cmap='gray', aspect='auto', origin='lower', vmin=-100, vmax=400)

    # Loop over each good index to load prediction and plot contours
    for i, index in enumerate(good_index):
        print(f"Processing index: {index}")

        # Load the prediction data
        #label_final_path = os.path.join(base_file_path, f"{index}PETWB_001516_02_label_.nii")
        label_final_path = os.path.join(base_file_path, "sentence_" + str(index) + ".nii")

        nii_prediction = nib.load(label_final_path)
        prediction_data = nii_prediction.get_fdata()

        # Squeeze the data and process it
        prediction_data = np.squeeze(prediction_data, axis=(0, 1))
        prediction_data = filter_prediction_by_average(prediction_data)
        suv_data = resample_img_size(suv_data, prediction_data)

        # Compute Maximum Intensity Projection (MIP) for the prediction
        prediction_mip = np.max(prediction_data, axis=1)
        prediction_mip = np.rot90(prediction_mip, k=3)
        #print(f"Levels: {prediction_mip.min()}, {prediction_mip.max()}")

        alpha_value = 0.5  # Use a valid alpha value
        ax.contour(prediction_mip, levels=[0.5], colors=[cm.jet(i / len(good_index))], alpha=alpha_value, linewidths=5)
        #ax.contour(prediction_mip, levels=[0.5], colors= '#00FF00', alpha=alpha_value, linewidths=3)
    ax.axis('off')
    # Display the plot
    #plt.title('SUV MIP with Prediction Contours')
    plt.show()
    plt.savefig(
        "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/interactive_report_figure_v2/" + "interactive_figure_v2" + ".png")
    plt.close()
