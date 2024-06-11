
import numpy as np
import os
import nibabel as nib

def max_pixel_where_label_is_one(image, label):
    # Ensure that the label and image have the same shape
    if image.shape != label.shape:
        raise ValueError("Image and label must have the same dimensions")

    # Mask the image with the label
    masked_image = np.where(label == 1, image, np.nan)

    # Find the maximum value in the masked image
    max_value = np.nanmax(masked_image)

    return max_value

def check_max_pixel_and_slice(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v4_nifti/"

    for index, row in df.iterrows():

        print(f"index: {index}")

        label_path = os.path.join(label_path_base, row["Label_Name"] + ".nii.gz")

        file_path = "/mnt/Bradshaw/UW_PET_Data/SUV_images/" + row["Petlymph"]+ "/"
        files = os.listdir(file_path)
        index_of_suv = [index for index, s in enumerate(files) if "suv" in s.lower()]
        file_name = files[index_of_suv[0]]
        suv_image_path = file_path + file_name
        # print(suv_image_path)
        nii_image = nib.load(suv_image_path)
        image = nii_image.get_fdata()

        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        text_max = row["SUV"]
        max_pixel = max_pixel_where_label_is_one(image, label)

        print(f"max pixel: {max_pixel} text max: {text_max}")