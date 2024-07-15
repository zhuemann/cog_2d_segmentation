
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_training_and_inference_images():
    df = pd.read_excel("/UserData/Zach_Analysis/petlymph_image_data/" + "uw_final_df_9.xlsx")
    # json_file_path = "/UserData/Zach_Analysis/uw_lymphoma_pet_3d/output_resampled.json"
    json_file_path = "/UserData/Zach_Analysis/uw_lymphoma_pet_3d/output_resampled_13000_no_validation.json"
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    image_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/inference_images/"

    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4/"
    label_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels4/"

    #prediction_list = os.listdir(prediction_location)
    all_images = os.listdir(image_base)
    number_correct = 0
    index = 0

    #for label in prediction_list:
    for entry in data["testing"]:
        label = entry["label_name"]
        index += 1
        if number_correct > 1:
            print(f"index: {index} number that are correct: {number_correct} accuracy: {number_correct / index}")
        else:
            print(f"index: {index} number that are correct: {number_correct}")

        label_name = label.strip(".nii.gz")
        if "label_1" not in label_name:
            continue
        # these are swapped on purpose I saved them off with the wrong names
        ct_path = os.path.join(image_location, label_name + "_pet.nii")
        pet_path = os.path.join(image_location, label_name + "_ct.nii")


        # load in the suv data
        nii_suv = nib.load(pet_path)
        suv_data = nii_suv.get_fdata()
        # load in the ct data
        nii_ct = nib.load(ct_path)
        ct_data = nii_ct.get_fdata()

        suv_data = np.squeeze(suv_data)
        ct_data = np.squeeze(ct_data)
        print(f"max pet: {np.max(suv_data)} min pet: {np.min(suv_data)}")
        print(f"max ct: {np.max(ct_data)} min ct: {np.min(ct_data)}")

        """
        # Set up the figure and axes
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # Define titles for ease of identification
        titles = ['Sagittal View', 'Coronal View', 'Axial View']
        types = ['CT', 'PET']

        # Display each type of view
        for i, data in enumerate([ct_data, suv_data]):
            # Axial slice
            axial_slice = data[data.shape[0] // 2, :, :]
            #vmin, vmax = (-1000, 2000) if i == 0 else (0, 10)  # Set intensity range
            vmin, vmax = (0, 1) if i == 0 else (0, 1)  # Set intensity range
            axes[i, 0].imshow(axial_slice, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 0].axhline(y=axial_slice.shape[0] // 2, color='r', linestyle='--')  # Horizontal line
            axes[i, 0].axvline(x=axial_slice.shape[1] // 2, color='r', linestyle='--')  # Vertical line
            axes[i, 0].set_title(f'{types[i]} - {titles[0]}')
            axes[i, 0].axis('off')

            # Coronal slice
            coronal_slice = data[:, data.shape[1] // 2, :]
            axes[i, 1].imshow(coronal_slice, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 1].axhline(y=coronal_slice.shape[0] // 2, color='r', linestyle='--')  # Horizontal line
            axes[i, 1].axvline(x=coronal_slice.shape[1] // 2, color='r', linestyle='--')  # Vertical line
            axes[i, 1].set_title(f'{types[i]} - {titles[1]}')
            axes[i, 1].axis('off')

            # Sagittal slice
            sagittal_slice = data[:, :, data.shape[2] // 2]
            axes[i, 2].imshow(sagittal_slice, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 2].axhline(y=sagittal_slice.shape[0] // 2, color='r', linestyle='--')  # Horizontal line
            axes[i, 2].axvline(x=sagittal_slice.shape[1] // 2, color='r', linestyle='--')  # Vertical line
            axes[i, 2].set_title(f'{types[i]} - {titles[2]}')
            axes[i, 2].axis('off')
        """
        # Set up the figure and axes
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Updated for additional subplot

        # Define titles
        titles = ['Axial View', 'Coronal View', 'Sagittal View', 'Combined Axial View']
        types = ['CT', 'PET']

        # Processing images for each view
        for i, data in enumerate([ct_data, suv_data]):
            # Axial, Coronal, Sagittal
            for j in range(3):
                slice = None
                if j == 0:  # Axial
                    slice = data[data.shape[0] // 2, :, :]
                elif j == 1:  # Coronal
                    slice = data[:, data.shape[1] // 2, :]
                elif j == 2:  # Sagittal
                    slice = data[:, :, data.shape[2] // 2]

                vmin, vmax = (0, 1) if i == 0 else (0, 1)
                axes[i, j].imshow(slice, cmap='gray', vmin=vmin, vmax=vmax)
                axes[i, j].axhline(y=slice.shape[0] // 2, color='r', linestyle='--')
                axes[i, j].axvline(x=slice.shape[1] // 2, color='r', linestyle='--')
                axes[i, j].set_title(f'{types[i]} - {titles[j]}')
                axes[i, j].axis('off')

        # Combined Axial View
        axial_ct = ct_data[ct_data.shape[0] // 2, :, :]
        axial_suv = suv_data[suv_data.shape[0] // 2, :, :]

        # Normalize images for display
        axial_ct_norm = (axial_ct - axial_ct.min()) / (axial_ct.max() - axial_ct.min())
        axial_suv_norm = (axial_suv - axial_suv.min()) / (axial_suv.max() - axial_suv.min())

        # Create an RGB image with CT in blue and PET in red
        combined_image = np.zeros((axial_ct.shape[0], axial_ct.shape[1], 3))
        combined_image[..., 0] = axial_suv_norm  # Red channel
        combined_image[..., 2] = axial_ct_norm  # Blue channel

        axes[0, 3].imshow(combined_image)
        axes[0, 3].set_title('Combined Axial View - PET (Red) & CT (Blue)')
        axes[0, 3].axis('off')

        axes[1, 3].axis('off')  # Turn off the unused subplot

        # Show the plots
        plt.tight_layout()
        plt.show()

        plt.savefig(
            "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/data_used_to_inference_v4/" + label_name + ".png")
        plt.close()