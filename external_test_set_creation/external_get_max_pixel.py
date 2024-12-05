import pandas as pd


import pandas as pd
import numpy as np
import os
import nibabel as nib
import cc3d
from scipy.ndimage import zoom


def get_threshold(source):
    background = 2
    background = 1
    new_threshold = (.617*(background/source) + .316)*source
    return new_threshold


def slice_overlap_ref(slice_ref, slice_min, slice_max, slice_tolerance):
    min_point = slice_ref - slice_tolerance
    max_point = slice_ref + slice_tolerance

    return min_point <= slice_max and slice_min <= max_point


def get_max_pixel_of_segmented_regions_v2(labeled_regions, img):
    unique_labels = np.unique(labeled_regions[labeled_regions != False])
    max_suv_dict = {}
    for label in unique_labels:
        # Find the indices of all pixels belonging to the current label
        indices = np.argwhere(labeled_regions == label)

        # get the number of indices we have, esentially the volume of segmented area
        #num_indices = len(indices)
        #print(num_indices)
        # this is skip if the volume is too small
        #if num_indices <= 3:
        #    continue
        # Extract the corresponding pixel values from img
        pixel_values = img[indices[:, 0], indices[:, 1], indices[:, 2]]

        # Find the maximum pixel value and its index
        max_pixel_value = pixel_values.max()
        max_pixel_index = indices[pixel_values.argmax()]

        # Compute min and max slice indices for this label
        min_slice, max_slice = indices[:, 2].min(), indices[:, 2].max()

        # Update the dictionary
        max_suv_dict[label] = (max_pixel_value, min_slice, max_slice, tuple(max_pixel_index))

    return max_suv_dict

def get_max_pixel_of_segmented_regions_v3(labeled_regions, img, slice_ref):
    unique_labels = np.unique(labeled_regions[labeled_regions != False])
    max_suv_dict = {}
    for label in unique_labels:
        # Find the indices of all pixels belonging to the current label
        indices = np.argwhere(labeled_regions == label)

        # Check if the current component intersects with the given z-plane
        if not (indices[:, 2].min() < slice_ref < indices[:, 2].max()):
            continue  # Skip components that do not intersect the slice_ref

        # Extract the corresponding pixel values from img
        pixel_values = img[indices[:, 0], indices[:, 1], indices[:, 2]]

        # Find the maximum pixel value and its index
        max_pixel_value = pixel_values.max()
        max_pixel_index = indices[pixel_values.argmax()]

        # Compute min and max slice indices for this label
        min_slice, max_slice = indices[:, 2].min(), indices[:, 2].max()

        # Update the dictionary
        max_suv_dict[label] = (max_pixel_value, min_slice, max_slice, tuple(max_pixel_index))

    return max_suv_dict


def get_ct_name(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Identify the file with the substring 'SUV'
    suv_file = None
    for file in files:
        if 'SUV' in file:
            suv_file = file
            break

    if suv_file is None:
        print("No SUV file found.")
        return None

    # Extract the common part of the filename
    common_part = suv_file.split('_SUV')[0]
    #print(f"all files: {files}")
    #print(f"common part: {common_part}")
    # Find the third file that does not contain the common part and is not SUV
    for file in files:
        if common_part not in file and 'SUV' not in file and file.endswith('.nii.gz'):
            return file

    return None


def resample_image(target_image, target_affine, reference_image):
    # Calculate the zoom factor for each axis
    target_shape = target_image.shape
    reference_shape = reference_image.shape
    zoom_factors = np.array(reference_shape) / np.array(target_shape)

    # Resample the target image
    resampled_image = zoom(target_image, zoom_factors, order=3)  # order=3 for cubic interpolation

    # Create a new NIfTI image
    resampled_nii = nib.Nifti1Image(resampled_image, target_affine)

    return resampled_nii


def get_corresponding_pet_slice_v1(ct_slice_idx, ct_voxel_size, pet_voxel_size):
    """
    Calculates the corresponding PET slice for a given CT slice based on their voxel sizes.

    Args:
        ct_slice_idx (int): The index of the CT slice (0-based index).
        ct_voxel_size (float): The slice thickness of the CT volume (z-axis voxel size).
        pet_voxel_size (float): The slice thickness of the PET volume (z-axis voxel size).

    Returns:
        float: The corresponding PET slice index (can be a non-integer if interpolation is needed).
    """
    # Calculate the position of the CT slice in physical space (z-axis position)
    ct_slice_position = ct_slice_idx * ct_voxel_size

    # Calculate the corresponding PET slice index
    pet_slice_idx = ct_slice_position / pet_voxel_size

    return pet_slice_idx


def get_corresponding_pet_slice(ct_slice_idx, ct_voxel_size, pet_voxel_size):
    """
    Calculates the corresponding PET slice for a given CT slice based on their voxel sizes.

    Args:
        ct_slice_idx (int): The index of the CT slice (0-based index).
        ct_voxel_size (tuple): The voxel size of the CT volume (z, y, x).
        pet_voxel_size (tuple): The voxel size of the PET volume (z, y, x).

    Returns:
        float: The corresponding PET slice index (can be a non-integer if interpolation is needed).
    """
    # Extract the slice thickness (z-axis voxel size)
    ct_slice_thickness = ct_voxel_size[2]
    pet_slice_thickness = pet_voxel_size[2]

    # Calculate the position of the CT slice in physical space (z-axis position)
    ct_slice_position = ct_slice_idx * ct_slice_thickness

    # Calculate the corresponding PET slice index
    pet_slice_idx = ct_slice_position / pet_slice_thickness

    return pet_slice_idx

def get_max_pixel_step3(df):
    # check how many sentences have a pet scan with them
    #uw_100 = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain/concensus_slice_suv_anonymized_2.xlsx"
    #uw_100 = pd.read_excel(uw_100)
    # print(uw_100)
    #uw_100 = uw_100[uw_100["Petlymph"] == 'PETLYMPH_4361']
    # uw_100 = uw_100[uw_100["Petlymph"] == 'PETLYMPH_4513']

    #print(uw_100)
    uw_100 = df

    patient_decoding = "/UserData/Zach_Analysis/patient_decoding.xlsx"
    patient_decoding = pd.read_excel(patient_decoding)
    valid_pet_scans = set(os.listdir("/UserData/Zach_Analysis/suv_nifti/"))
    valid_pet_scans = set(os.listdir("/mnt/Bradshaw/UW_PET_Data/SUV_images/"))
    valid_pet_scans = set(os.listdir("/mnt/Bradshaw/UW_PET_Data/external_testset_v2/"))

    df_orientation = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/orientation_labeled_manually.xlsx")

    count = 0
    two_rows = 0
    found_noted_lesion = 0
    found_pet_scan = 0
    sentences_not_evalued_missing_pet = 0
    no_suv_file_but_does_have_mac = 0
    found_pixels_df = []
    below_suv_threshold = 0
    dups_found = 0
    slice_diff_exceeded = 0
    too_long_not_used = 0
    ct_missing = 0

    for index, row in uw_100.iterrows():
        print(f"index: {index} mathces_found: {found_noted_lesion} duplicates: {dups_found} missing pet: {sentences_not_evalued_missing_pet} missing ct: {ct_missing} too long: {slice_diff_exceeded} ")
        # if index < 3645:
        #    continue
        #if index > 500:
        #if index < 13:
        #    continue
        #    break
        # if index > 10:
        #    break
        #if index < 20000:
        #    continue
        #if index > 20000:
        #    break
        """
        accession_num = row["Accession Number"]
        rows_with_value = patient_decoding[patient_decoding['Accession Number'] == accession_num]
        # print(len(rows_with_value))
        if len(rows_with_value) == 2:
            two_rows += 1
            continue
        # if len(rows_with_value) < 2:
        #    continue
        if patient_decoding['Accession Number'].isin([accession_num]).any():
            pet_id = rows_with_value.iloc[0].iloc[1]
        """
        pet_id = row["Petlymph"]
        #check_id = str(pet_id).lower() + "_" + str(pet_id).lower()
        check_id = pet_id
        if check_id in valid_pet_scans:
            found_pet_scan += 1

            # gets the suv image as a numpy array
            #file_path = "/UserData/Zach_Analysis/suv_nifti/" + check_id + "/"
            file_path = "/mnt/Bradshaw/UW_PET_Data/SUV_images/" + check_id + "/"
            file_path = "/mnt/Bradshaw/UW_PET_Data/external_testset_v2/" + check_id + "/"

            files = os.listdir(file_path)
            index_of_suv = [index for index, s in enumerate(files) if "suv" in s.lower()]
            if len(index_of_suv) == 0:
                no_suv_file_but_does_have_mac += 1
                continue
            file_name = files[index_of_suv[0]]
            suv_image_path = file_path + file_name
            # print(suv_image_path)
            nii_image = nib.load(suv_image_path)
            img = nii_image.get_fdata()
            #print(f"pet shape: {img.shape}")
            # Get the voxel dimensions
            pet_dimensions = nii_image.header.get_zooms()
            #print(f"pet voxel dimensions: {pet_dimensions}")

            file = file_name.split("_")[0]

            ct_name = get_ct_name(file_path)
            if ct_name == None:
                ct_missing += 1
                continue
            #print(ct_name)
            ct_path = file_path + ct_name
            ct_nii = nib.load(ct_path)
            ct_image = ct_nii.get_fdata()
            #print(f"ct shape: {ct_image.shape}")
            ct_dimensions = ct_nii.header.get_zooms()
            #print(f"ct file name: {ct_name} ct voxel dimensions: {ct_dimensions}")

            orientation_row = df_orientation[df_orientation["ID"] == file]

            if orientation_row["Drop"].iloc[0] == 1 or orientation_row["Missing"].iloc[0] == 1:
                print("dropping because missing files or something is wrong with sentence")
                continue


            """
            ct_affine = ct_nii.affine
            pet_affine = nii_image.affine
            resampled_pet_nii = resample_image(img, pet_affine, ct_image)

            img = resampled_pet_nii
            img = img.get_fdata()
            """

            suv_ref = row["SUV"]
            if suv_ref < 2.5:
                below_suv_threshold += 1
                continue
            #print(f"ct dims: {ct_dimensions} pet dims: {pet_dimensions}")
            #slice_ref = np.round(int(row["Slice"]) * (ct_dimensions[2]/pet_dimensions[2]))
            slice_ref = int(row["Slice"])
            #print(f"orginal slice: {row['Slice']} after conversion: {slice_ref}")
            proposed_threshold = get_threshold(suv_ref)
            #print(f"proposed_threshold: {proposed_threshold}")
            threshold_value = suv_ref * .8
            #print(f"current threshold: {threshold_value}")
            # segmented_regions = img > threshold_value
            segmented_regions = img > proposed_threshold
            labels_out = cc3d.connected_components(segmented_regions, connectivity=6)

            max_suv_dic = get_max_pixel_of_segmented_regions_v2(labels_out, img)

            #slice_tolerance = 3
            #slice_tolerance = suv_ref
            slice_tolerance = 1
            suv_tolerance = 1
            #suv_tolerance = suv_ref*0.05

            slice_ref = int(row["Slice"]) # if this is pet slice number
            #slice_ref_pet = img.shape[2] - slice_ref
            #slice_ref_ct = ct_nii.shape[2] - slice_ref
            # if this is ct slice number



            # Get the voxel dimensions
            ct_voxel_size = ct_nii.header.get_zooms()  # (slice thickness, pixel spacing x, pixel spacing y) for CT
            pet_voxel_size = nii_image.header.get_zooms()  # (slice thickness, pixel spacing x, pixel spacing y) for PET
            slice_ref_ct = get_corresponding_pet_slice(slice_ref, ct_voxel_size, pet_voxel_size)

            if orientation_row["Bottom"].iloc[0] == 1:

                if orientation_row["CT"].iloc[0] == 1:

                    slice_ref_ct = ct_nii.shape[2] - slice_ref

                    slice_ref_pet_inverted = get_corresponding_pet_slice(slice_ref, ct_voxel_size, pet_voxel_size)
                    slice_ref = slice_ref_pet_inverted
                else:
                    # pet from the bottom
                    slice_ref_pet = img.shape[2] - slice_ref
                    slice_ref = slice_ref_pet
                    #slice_ref = slice_ref
            else:
                if orientation_row["CT"].iloc[0] == 1: # ct from the top
                    slice_ref_ct = slice_ref
                    #slice_ref_ct = ct_nii.shape[2] - slice_ref
                    slice_ref_pet = get_corresponding_pet_slice(slice_ref_ct, ct_voxel_size, pet_voxel_size)
                    slice_ref = slice_ref_pet
                else: # pet from the top
                    #slice_ref_pet = img.shape[2] - slice_ref
                    #slice_ref = slice_ref_pet
                    slice_ref = slice_ref


            """
            old orientation
            if row["tag"] == "ct":
                print("found ct")
                slice_ref = slice_ref_ct
                print(f"slice_ref if it is PET: {slice_ref} slice_ref if it is CT: {slice_ref_ct} tag: {row['tag']}")

            print(f"orientation: {orientation}")
            if orientation == "HFS": #"FFS": #"HFS": # flipping this to test was "FFS"
                slice_ref = img.shape[2] - slice_ref
            else:
                print(f"no flipping")
                slice_ref = slice_ref
            """

            #ct_from_head = ct_image.shape[2] - row["Slice"]
            #pet_from_head = int(np.round(ct_from_head/ct_dimensions[2]*pet_dimensions[2]))
            #slice_ref = pet_from_head
            found_items = 0
            for key, value in max_suv_dic.items():
                suv_max, slice_min, slice_max, pixel = value
                #print(
                #    f"Real SUVmax: {suv_ref} slice range passed slice_min: {slice_min} slice_max: {slice_max} suv_max: {suv_max}")
                #print(f"length: {slice_max - slice_min}")
                # inverts teh slice indexing to match physican convention
                #slice_min = img.shape[2] - slice_min
                #slice_max = img.shape[2] - slice_max
                #slice_max = img.shape[2] - slice_min
                #slice_min = img.shape[2] - slice_max
                # inverts the refering slice to match the python feet first convention
                #slice_ref = img.shape[2] - slice_ref

                #print(
                #    f"Real SUVmax: {suv_ref} slice range passed slice_min: {slice_min} slice_max: {slice_max} suv_max: {suv_max}")
                # check if our noted slice from the physican is between the max and min slices extracted with tolerance
                if slice_overlap_ref(slice_ref, slice_min, slice_max, slice_tolerance):
                    # if (slice_min - slice_tolerance) <= slice_ref and (slice_max + slice_tolerance) >= slice_ref:
                    # check if our suv_max from segmentation is within the suv tolerance noted
                    #print(f"slice ref: {slice_ref} slice_min: {slice_min} slice_max: {slice_max}")
                    #print(f"slice range: {slice_ref - slice_tolerance} to {slice_ref + slice_tolerance}")
                    #print(f"Real SUVmax: {suv_ref} slice range passed slice_min: {slice_min} slice_max: {slice_max} suv_max: {suv_max}")
                    if abs(suv_max - suv_ref) <= suv_tolerance:
                        # catch long initial contours that are clearly not lesions
                        if slice_max - slice_min > 50 and suv_ref < 5:
                            slice_diff_exceeded += 1
                            continue

                        found_noted_lesion += 1
                        # print(row)
                        pixel_i, pixel_j, pixel_k = pixel
                        row_list = row.tolist()
                        row_list.extend([pixel_i, pixel_j, pixel_k])
                        found_pixels_df.append(row_list)
                        found_items += 1
            if found_items > 1:
                dups_found += found_items

        else:
            print(f"counting as missed pet: {check_id}")
            sentences_not_evalued_missing_pet += 1

    new_columns = list(uw_100.columns) + ['i', 'j', 'k']
    new_df = pd.DataFrame(found_pixels_df, columns=new_columns)
    #new_df.to_excel('/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v2/found_pixels_in_sentence_uw_anonymized_3_v4.xlsx', index=False)
    print(new_df)
    print(len(new_df))
    print(f"below suv 2.5: {below_suv_threshold}")
    print(f"colision of accesion number: {two_rows}")
    print(f"found pet scans: {found_pet_scan}")
    print(f"lesions succesfully located: {found_noted_lesion}")
    print(f"not evaluaged sentences missing pet: {sentences_not_evalued_missing_pet}")
    print(f"no suv but does have pet: {no_suv_file_but_does_have_mac}")
    print(f"number of lesions lost slice dif exceeded 50: {slice_diff_exceeded}")
    return new_df





def external_get_max_pixel():

    df = pd.read_excel('/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/swedish_dataframe_test.xlsx')
    df = get_max_pixel_step3(df)

    df.to_excel('/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/swedish_dataframe_max_pixels_v9_orientation_accounting_flipped.xlsx')
    print(df)
