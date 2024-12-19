
import os
import nibabel as nib
import pandas as pd
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


def pad_ct_scan_symetrically(ct_image, target_size=200):
    # Get the data as a numpy array
    data = ct_image.get_fdata()

    # Calculate the padding needed for the first two dimensions
    padding = []
    for dim in data.shape[:2]:
        if dim < target_size:
            # Calculate total padding required to reach the target size
            total_pad = target_size - dim
            # Distribute padding on both sides
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding.append((pad_before, pad_after))
        else:
            padding.append((0, 0))

    # No padding for the third dimension
    padding.append((0, 0))

    # Apply the padding with constant value of -1000 for air in CT
    padded_data = np.pad(data, padding, mode='constant', constant_values=-1000)

    # Create a new NIfTI image from the padded data with the same affine and header as the original
    padded_ct_image = nib.Nifti1Image(padded_data, ct_image.affine, ct_image.header)

    return padded_ct_image


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

def crop_center_with_offset(img, z_offset=0):
    x, y, z = img.shape
    crop_x = min(200, x)
    crop_y = min(200, y)
    crop_z = min(350, z)

    start_x = max((x - crop_x) // 2, 0)
    start_y = max((y - crop_y) // 2, 0)
    start_z = max(z - crop_z - z_offset, 0)  # Subtract the z_offset here

    cropped_img = img.slicer[start_x:start_x + crop_x, start_y:start_y + crop_y, start_z:start_z + crop_z]
    return cropped_img

def crop_z_axis(img, crop_offset):
    if crop_offset == 0:
        return img
    else:
        return img.slicer[:,:,:-crop_offset]

def resampling_and_cropping(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    #label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v4_nifti/"
    #label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v4_nifti/"
    label_path_base = "/UserData/Zach_Analysis/physican_labeling_UWPET/josh_nifti/"
    print("hi")

    #images_folder = "images6"
    #label_folder = "labels6"
    images_folder = "images"
    label_folder = "labels"

    #save_path = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/"
    save_path = "/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/"

    processed_images = generate_processed_images_dict(save_path + images_folder)

    crop_lookup = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/crop_offset_lookup.xlsx")
    #print(processed_images)
    number_of_missing_ct = 0
    label_cropped_out = 0
    resampling_saved = 0
    i = 0
    already_processed = 0

    pet_used = []
    ct_used = []
    label_used = []
    id = []

    for index, row in df.iterrows():

        if row["Zach_drop"] == "1":
            continue
        petlymph = row["id"]

        print(f"i: {i} Petlymph: {petlymph} labels cropped out: {label_cropped_out} missing ct: {number_of_missing_ct} resampling saved: {resampling_saved} already processed: { already_processed}")
        i += 1

        image_path = os.path.join(image_path_base, petlymph)

        #label_path = os.path.join(label_path_base, row["Label_Name"])
        label_path = os.path.join(label_path_base, row["File_Name"])

        label_path = str(label_path) + ".nii.gz"

        # gets the crop offset for all images matching this petlymph
        matched_row = crop_lookup[crop_lookup['id'] == petlymph]

        # Check if the filtered DataFrame is not empty
        if not matched_row.empty:
            # Extract the 'crop_offset' value from the matched row
            crop_offset = int(matched_row['crop_offset'].iloc[0]) - 1
        else:
            crop_offset = 0
            print("no offset, it was missing")

        # Check if this PET/CT pair has already been processed
        if petlymph in processed_images:
            print(f"{petlymph} PET/CT images already processed. Checking label...")
            if os.path.exists(os.path.join("/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/", str(label_folder), f'{row["Label_Name"]}.nii.gz')):
                already_processed += 1
                continue
            else:
                try:
                    label_image = nib.load(label_path)
                except FileNotFoundError:
                    print("One of the files does not exist: label image.")
                    continue
                label_cropped = crop_z_axis(label_image, crop_offset)
                label_resampled = resample_img(label_cropped, target_affine=np.diag([3, 3, 3]), interpolation='nearest')
                label_cropped = crop_center_with_offset(label_resampled, z_offset=0)
                if np.any(label_cropped.get_fdata() != 0):
                    nib.save(label_cropped,
                             os.path.join("/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/", str(label_folder),
                                          f'{row["Label_Name"]}.nii.gz'))
                    resampling_saved += 1
                else:
                    label_cropped_out += 1
                    continue
            continue

        file_names = os.listdir(image_path)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        suv_path = os.path.join(image_path, file_names[index_of_suv[0]])

        index_of_ct = [index for index, element in enumerate(file_names) if "ct" in element.lower() and "irctac" not in element.lower()]
        #index_of_ct = [index for index, element in enumerate(file_names) if "ct" in element.lower()]
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

        try:
            print(f"label path: {label_path}")
            ct_image = nib.load(ct_image_path)
            suv_image = nib.load(suv_path)
            label_image = nib.load(label_path)
        except FileNotFoundError:
            print("One of the files does not exist: CT, SUV, or label image.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue

        ct_used.append(ct_image_path)
        pet_used.append(suv_path)
        label_used.append(label_path)
        id.append(petlymph)

        ct_image = crop_z_axis(ct_image, crop_offset)
        suv_image = crop_z_axis(suv_image, crop_offset)
        label_image = crop_z_axis(label_image, crop_offset)
        # crop from the back end first then do the resmpaling and final crop
        # Resample images to 3mm x 3mm x 3mm
        target_affine = np.diag([3, 3, 3])
        ct_resampled = resample_img(ct_image, target_affine=target_affine, interpolation = "linear")
        suv_resampled = resample_img(suv_image, target_affine=target_affine, interpolation = "linear")
        label_resampled = resample_img(label_image, target_affine=target_affine, interpolation='nearest')
        ct_cropped = crop_center_with_offset(ct_resampled, z_offset=0)
        suv_cropped = crop_center_with_offset(suv_resampled, z_offset=0)
        label_cropped = crop_center_with_offset(label_resampled, z_offset=0)

        if ct_cropped.get_fdata().shape[0] < 200 or ct_cropped.get_fdata().shape[1] < 200:
            ct_cropped = pad_ct_scan_symetrically(ct_cropped, target_size=200)

        # Check if any label is still in the image
        if np.any(label_cropped.get_fdata() != 0):
            # Save the cropped images
            nib.save(ct_cropped, os.path.join("/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/", str(images_folder), f'{petlymph}_ct_cropped.nii.gz'))
            nib.save(suv_cropped, os.path.join("/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/", str(images_folder), f'{petlymph}_suv_cropped.nii.gz'))
            nib.save(label_cropped, os.path.join("/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/", str(label_folder), f'{row["Label_Name"]}.nii.gz'))
            processed_images[petlymph] = 1
        else:
            label_cropped_out += 1

    print(f"Total missing CT scans: {number_of_missing_ct}")
    print(f"Total labels cropped out: {label_cropped_out}")
    df_images_used = pd.DataFrame({
        'ID': id,
        'PET_Used': pet_used,
        'CT_Used': ct_used,
        'Label_Used': label_used,
    })
    return df_images_used

