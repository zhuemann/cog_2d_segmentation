import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def plot_3d_predictions():

    prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_forked2/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/prediction_trash_v2testing/"

    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images/"
    label_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels/"

    prediction_list = os.listdir(prediction_location)
    all_images = os.listdir(image_base)

    for label in prediction_list:

        image_name = label[:-15]
        print(f"image name: {image_name}")
        label_name = label[:-7]
        print(label_name)


        suv_path_final = os.path.join(image_base, image_name + "_suv_cropped.nii.gz")
        ct_path_final = os.path.join(image_base, image_name + "_ct_cropped.nii.gz")
        full_label_path = os.path.join(prediction_location, label)
        label_full_path = os.path.join(label_base, label)

        # load in the suv data
        nii_suv = nib.load(suv_path_final)
        suv_data = nii_suv.get_fdata()
        # load in the ct data
        nii_ct = nib.load(ct_path_final)
        ct_data = nii_ct.get_fdata()
        # load in the prediciton data
        nii_prediction = nib.load(full_label_path)
        prediction_data = nii_prediction.get_fdata()
        # load in label data
        nii_label = nib.load(label_full_path)
        label_data = nii_label.get_fdata()

        # Compute maximum intensity projection along axis 1
        suv_mip = np.max(suv_data, axis=1)
        prediction_mip = np.max(prediction_data, axis=1)
        label_mip = np.max(label_data, axis=1)

        print(f"suv mip size: {suv_mip.shape}")
        print(f"pred mip size: {prediction_mip.shape}")
        print(f"label mip size: {label_mip.shape}")

        # Setup the plot with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Label MIP overlayed on SUV MIP
        axes[0].imshow(suv_mip.T, cmap='gray', aspect='auto', origin='lower')
        axes[0].imshow(label_mip.T, cmap='autumn', alpha=0.5, aspect='auto',
                       origin='lower')  # Adjust alpha for overlay transparency
        axes[0].set_title('Label Overlay on SUV MIP')
        axes[0].axis('off')  # Turn off axis

        # Plot 2: Prediction MIP overlayed on SUV MIP
        axes[1].imshow(suv_mip.T, cmap='gray', aspect='auto', origin='lower')
        axes[1].imshow(prediction_mip.T, cmap='autumn', alpha=0.5, aspect='auto', origin='lower')
        axes[1].set_title('Prediction Overlay on SUV MIP')
        axes[1].axis('off')

        # Plot 3: Both Prediction and Label MIP overlayed on SUV MIP
        axes[2].imshow(suv_mip.T, cmap='gray', aspect='auto', origin='lower')
        axes[2].imshow(label_mip.T, cmap='autumn', alpha=0.5, aspect='auto', origin='lower')
        axes[2].imshow(prediction_mip.T, cmap='winter', alpha=0.5, aspect='auto', origin='lower')
        axes[2].set_title('Prediction and Label Overlay on SUV MIP')
        axes[2].axis('off')

        # Save the figure
        plt.tight_layout()
        plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/3d_predictions_v1/" + label_name + ".png")
        plt.close()
