
import os
import nibabel as nib
from nilearn.image import resample_img, crop_img
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Casting data from int16 to float32')


def generate_processed_images_dict(save_path):
    processed_images = {}

    # List all files in the given directory
    for filename in os.listdir(save_path):
        if filename.endswith("_cropped.nii.gz"):
            # Extract the petlymph identifier from the filename
            parts = filename.split('_')
            petlymph = '_'.join(parts[:-2])  # Assuming the PET/CT identifier is everything except the last two parts

            # Mark this PET/CT as processed
            if petlymph not in processed_images:
                processed_images[petlymph] = {'ct': False, 'suv': False, 'labels': []}

            # Check which type of image has been processed
            if 'ct' in filename:
                processed_images[petlymph]['ct'] = True
            elif 'suv' in filename:
                processed_images[petlymph]['suv'] = True
            elif 'label' in filename:
                label_name = parts[-2]  # The label name should be the second to last part of the filename
                processed_images[petlymph]['labels'].append(label_name)

    return processed_images


# Crop the images to 60cm x 60cm in x and y, and last 350 slices in z
def crop_center(img):
    x, y, z = img.shape
    crop_x = min(200, x)
    crop_y = min(200, y)
    crop_z = min(350, z)

    start_x = max((x - crop_x) // 2, 0)
    start_y = max((y - crop_y) // 2, 0)
    start_z = max(z - crop_z, 0)

    cropped_img = img.slicer[start_x:start_x + crop_x, start_y:start_y + crop_y, start_z:start_z + crop_z]
    return cropped_img

def resampling_and_cropping(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v2_nifti/"
    print("hi")

    save_path = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/"
    processed_images = generate_processed_images_dict(save_path + str("images"))


    number_of_missing_ct = 0
    label_cropped_out = 0
    resampling_saved = 0
    i = 0
    already_processed = 0
    for index, row in df.iterrows():

        print(f"i: {i} labels cropped out: {label_cropped_out} missing ct: {number_of_missing_ct} resampling saved: {resampling_saved} already processed: { already_processed}")
        i += 1

        petlymph = row["Petlymph"]
        image_path = os.path.join(image_path_base, petlymph)


        label_path = os.path.join(label_path_base, row["Label_Name"])
        label_path = str(label_path) + ".nii.gz"

        # Check if this PET/CT pair has already been processed
        if petlymph in processed_images:
            print(f"{petlymph} PET/CT images already processed. Checking label...")
            if os.path.exists(os.path.join("/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels/", f'{row["Label_Name"]}.nii.gz')):
                already_processed += 1
                continue
            else:
                label_image = nib.load(label_path)
                label_resampled = resample_img(label_image, target_affine=np.diag([3, 3, 3]), interpolation='nearest')
                label_cropped = crop_center(label_resampled)
                nib.save(label_cropped, os.path.join("/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels/", f'{row["Label_Name"]}.nii.gz'))
                resampling_saved += 1
            continue

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
        label_resampled = resample_img(label_image, target_affine=target_affine, interpolation='nearest')

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


