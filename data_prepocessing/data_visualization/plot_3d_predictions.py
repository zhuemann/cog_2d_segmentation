import os
import nibabel as nib
import numpy as np

def plot_3d_predictions():

    prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_forked2/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/prediction_trash_v2testing/"

    image_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    labels_list = os.listdir(prediction_location)

    for label in labels_list:

        image_name = label[:--7]
        print(image_name)
        current_path = os.path.join(image_base, folder)

        ct_path_final = None
        suv_path_final = None

        for file_name in os.listdir(current_path):
            if "CT" in file_name:
                ct_path_final = os.path.join(current_path, file_name)
            if "SUV" in file_name:
                suv_path_final = os.path.join(current_path, file_name)


        full_label_path = os.path.join(prediction_location, label)
        # Load the image file
        nii_image = nib.load(full_label_path)

        # Get the data as a NumPy array
        image_data = nii_image.get_fdata()

