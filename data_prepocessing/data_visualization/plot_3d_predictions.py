import os
import nibabel as nib
import numpy as np

def plot_3d_predictions():

    prediction_location = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_forked2/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/prediction_trash_v2testing/"

    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/"
    labels_list = os.listdir(prediction_location)
    all_images = os.listdir(image_base)

    for label in labels_list:

        image_name = label[:-15]
        print(f"image name: {image_name}")
        label_name = label[:-7]
        print(label_name)


        suv_path_final = os.path.join(image_base, image_name + "_suv_cropped.nii.gz")
        ct_path_final = suv_path_file = os.path.join(image_base, image_name + "_ct_cropped.nii.gz")

        print(suv_path_final)
        print(ct_path_final)
        full_label_path = os.path.join(prediction_location, label)
        # Load the image file
        nii_image = nib.load(full_label_path)

        # Get the data as a NumPy array
        image_data = nii_image.get_fdata()

