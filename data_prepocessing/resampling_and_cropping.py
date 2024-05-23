
import os
import nibabel as nib
from nilearn.image import resample_img, crop_img
import numpy as np

def resampling_and_cropping(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v2_nifti/"
    print("hi")

    number_of_missing_ct = 0
    label_cropped_out = 0
    for index, row in df.iterrows():

        petlymph = row["Petlymph"]
        image_path = os.path.join(image_path_base, petlymph)

        label_path_base = os.path.join(label_path_base, row["Label_Name"])
        label_path = str(label_path_base) + ".nii.gz"

        file_names = os.listdir(image_path)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        suv_path = os.path.join(image_path, file_names[index_of_suv[0]])


        index_of_ct = [index for index, element in enumerate(file_names) if "ct" in element.lower()]
        # Check if any file was found that contains "CT"
        if index_of_ct:
            # Update image_path to include the file name of the CT image
            ct_image_path = os.path.join(image_path_base, petlymph, file_names[index_of_ct[0]])
        else:
            # Handle the case where no CT file is found
            ct_image_path = None
            print("No CT file found in the directory.")
            number_of_missing_ct += 1
            continue

        ct_image = nib.load(ct_image_path)
        suv_image = nib.load(suv_path)
        label_image = nib.load(label_path)

        # Resample images to 3mm x 3mm x 3mm
        target_affine = np.diag([3, 3, 3])
        ct_resampled = resample_img(ct_image, target_affine=target_affine)
        suv_resampled = resample_img(suv_image, target_affine=target_affine)
        label_resampled = resample_img(label_image, target_affine=target_affine)

        # Crop the images to 60cm x 60cm in x and y, and last 350 slices in z
        def crop_center(img):
            # Calculate the cropping sizes
            x, y, z = img.shape
            start_x = (x - 200) // 2  # 60cm/3mm = 200 voxels
            start_y = (y - 200) // 2
            cropped_img = img.slicer[start_x:start_x + 200, start_y:start_y + 200, -350:]
            return cropped_img

        ct_cropped = crop_center(ct_resampled)
        suv_cropped = crop_center(suv_resampled)
        label_cropped = crop_center(label_resampled)

        # Check if any label is still in the image
        if np.any(label_cropped.get_fdata() != 0):
            # Save the cropped images
            nib.save(ct_cropped, os.path.join("/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images/", f'{petlymph}_ct_cropped.nii.gz'))
            nib.save(suv_cropped, os.path.join("/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images/", f'{petlymph}_suv_cropped.nii.gz'))
            nib.save(label_cropped, os.path.join("/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels/", f'{row["Label_Name"]}.nii.gz'))
        else:
            label_cropped_out += 1

    print(f"Total missing CT scans: {number_of_missing_ct}")
    print(f"Total labels cropped out: {label_cropped_out}")


