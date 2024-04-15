
import pandas as pd
import numpy as np
import os
import nibabel as nib
import cc3d

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

    count = 0
    two_rows = 0
    found_noted_lesion = 0
    found_pet_scan = 0
    sentences_not_evalued_missing_pet = 0
    no_suv_file_but_does_have_mac = 0
    found_pixels_df = []
    below_suv_threshold = 0
    dups_found = 0

    for index, row in uw_100.iterrows():
        print(f"index: {index} mathces_found: {found_noted_lesion} duplicates: {dups_found} missing pet: {sentences_not_evalued_missing_pet}")
        # if index < 3645:
        #    continue
        #if index > 500:
        #    break
        # if index > 10:
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
        check_id = str(pet_id).lower() + "_" + str(pet_id).lower()
        if check_id in valid_pet_scans:
            found_pet_scan += 1

            # gets the suv image as a numpy array
            file_path = "/UserData/Zach_Analysis/suv_nifti/" + check_id + "/"
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

            suv_ref = row["SUV"]
            if suv_ref < 2.5:
                below_suv_threshold += 1
                continue
            slice_ref = row["Slice"]
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
            slice_tolerance = 0
            suv_tolerance = 0.1
            #suv_tolerance = suv_ref*0.05
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
                slice_ref = img.shape[2] - slice_ref
                # if it is under 2.3 we don't want it
                if suv_max < 2.3:
                    continue

                #print(
                #    f"Real SUVmax: {suv_ref} slice range passed slice_min: {slice_min} slice_max: {slice_max} suv_max: {suv_max}")
                # check if our noted slice from the physican is between the max and min slices extracted with tolerance
                if slice_overlap_ref(slice_ref, slice_min, slice_max, slice_tolerance):
                    # if (slice_min - slice_tolerance) <= slice_ref and (slice_max + slice_tolerance) >= slice_ref:
                    # check if our suv_max from segmentation is within the suv tolerance noted
                    print(f"slice ref: {slice_ref} slice_min: {slice_min} slice_max: {slice_max}")
                    #print(f"slice range: {slice_ref - slice_tolerance} to {slice_ref + slice_tolerance}")
                    #print(f"Real SUVmax: {suv_ref} slice range passed slice_min: {slice_min} slice_max: {slice_max} suv_max: {suv_max}")
                    if abs(suv_max - suv_ref) <= suv_tolerance:
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
    return new_df