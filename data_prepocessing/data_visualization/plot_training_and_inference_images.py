
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

        image_name = label[:15]
        label_name = label.strip(".nii.gz")

        #for entry in data["testing"]:
        #    if label_name in entry.get('label'):
        #        sent = entry.get('report')  # Return the report if label name matches

        pet_path = os.path.join(image_location, label_name + "_pet.nii")
        ct_path = os.path.join(image_location, label_name + "_ct.nii")

        suv_path_final = os.path.join(image_base, image_name + "_suv_cropped.nii.gz")
        ct_path_final = os.path.join(image_base, image_name + "_ct_cropped.nii.gz")
        #full_pred_path = os.path.join(prediction_location, label)
        #label_full_path = os.path.join(label_base, label)

        # load in the suv data
        nii_suv = nib.load(pet_path)
        suv_data = nii_suv.get_fdata()
        # load in the ct data
        nii_ct = nib.load(ct_path)
        ct_data = nii_ct.get_fdata()
        # load in the prediciton data
        #nii_prediction = nib.load(full_pred_path)
        #prediction_data = nii_prediction.get_fdata()
        suv_data = np.squeeze(suv_data)
        ct_data = np.squeeze(ct_data)
        #prediction_data = np.squeeze(prediction_data, axis=(0, 1))

        # load in label data
        #nii_label = nib.load(label_full_path + ".gz")
        #label_data = nii_label.get_fdata()

        # Compute maximum intensity projection along axis 1
        suv_mip = np.max(suv_data, axis=1)
        ct_mip = np.max(ct_data, axis=1)
        #prediction_data = np.where(prediction_data < 0.5, 0, 1)
        #prediction_mip = np.max(prediction_data, axis=1)
        #label_mip = np.max(label_data, axis=1)

        #label_suv_max = max_suv_in_positive_region(suv_data, label_data)
        #prediction_suv_max = max_suv_in_positive_region(suv_data, prediction_data)
        #correct = label_suv_max == prediction_suv_max

        # Reflect the data horizontally so the heart is on the left
        suv_mip = np.fliplr(suv_mip.T)
        ct_mip = np.fliplr(ct_mip.T)

        # Set up the figure and axes
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Define titles for ease of identification
        titles = ['Axial View', 'Coronal View', 'Sagittal View']
        types = ['CT', 'PET']

        # Display each type of view
        for i, data in enumerate([ct_data, suv_data]):
            # Axial slice
            axial_slice = data[data.shape[0] // 2, :, :]
            axes[i, 0].imshow(np.fliplr(axial_slice.T), cmap='gray')
            axes[i, 0].set_title(f'{types[i]} - {titles[0]}')
            axes[i, 0].axis('off')

            # Coronal slice
            coronal_slice = data[:, data.shape[1] // 2, :]
            axes[i, 1].imshow(np.fliplr(coronal_slice.T), cmap='gray')
            axes[i, 1].set_title(f'{types[i]} - {titles[1]}')
            axes[i, 1].axis('off')

            # Sagittal slice
            sagittal_slice = data[:, :, data.shape[2] // 2]
            axes[i, 2].imshow(np.fliplr(sagittal_slice.T), cmap='gray')
            axes[i, 2].set_title(f'{types[i]} - {titles[2]}')
            axes[i, 2].axis('off')

        # Show the plots
        plt.tight_layout()
        plt.show()

        plt.savefig(
            "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/data_used_to_inference_v4/" + label_name + ".png")
        plt.close()