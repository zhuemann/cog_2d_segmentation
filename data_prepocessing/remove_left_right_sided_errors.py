import os
import pandas as pd
import nibabel as nib
import numpy as np


def average_components(tuples_list):
    sum_i, sum_j, sum_k = 0, 0, 0
    count = len(tuples_list)

    # Summing up all i, j, k components
    for i, j, k in tuples_list:
        sum_i += i
        sum_j += j
        sum_k += k

    # Calculating average for each component
    if count > 0:
        avg_i = sum_i / count
        avg_j = sum_j / count
        avg_k = sum_k / count
    else:
        avg_i = avg_j = avg_k = 0  # handle the case where the list is empty

    # Print the averages
    print(f"Average of i: {avg_i}")
    print(f"Average of j: {avg_j}")
    print(f"Average of k: {avg_k}")

def first_nonzero_plane(label):
    # Iterate over each slice along the first dimension (i-index)
    for i in range(label.shape[0]):
        if np.any(label[i,:,:] != 0):
            return i
    return None  # Return None if all elements are zero

def count_left_right_sided(df, label_path_base):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    #label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v2_nifti/"
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1",
                      "PETWB_012579_01_label_2", "PETWB_011401_02_label_3", "PETWB_003190_01_label_2", "PETWB_011401_02_label_2"]
    #labels_to_skip = ["PETWB_011355_01_label_5", "PETWB_012579_01_label_2", "PETWB_003190_01_label_2", "PETWB_002466_01_label_1"]
    left_indices = []
    right_indices = []
    left_and_right_count = 0
    neither_count = 0

    left_index_list = []
    right_index_list = []

    left_cross_midline = 0
    right_cross_midline = 0
    rows_to_return = []
    for index, row in df.iterrows():
        print(f"index: {index} id: {row['Label_Name']}")
        sentence = row["sentence"]
        if row["Label_Name"] in labels_to_skip:
            continue
        #if index < 10000:
        #    continue
        #if index < 5000:
        #    continue
        label_name = row["Label_Name"] + ".nii.gz"
        label_path = os.path.join(label_path_base, label_name)
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        midpoint = label.shape[0]/2
        buffer_dist = 10
        coordinates = (row["i"], row["j"], row["k"])

        number_of_images_correct = 0
        not_checked = 0
        if 'left' in sentence and 'right' not in sentence:
            left_indices.append(coordinates)
            index = first_nonzero_plane(label)
            left_index_list.append(index)
            cutoff = midpoint - buffer_dist
            if index < cutoff:
                print(f"left but on right half index: {index} midline: {cutoff} label name: {row['Label_Name']}")
                left_cross_midline += 1
                rows_to_return.append(row["Label_Name"])
            elif index > midpoint + buffer_dist:
                number_of_images_correct += 1
            else:
                not_checked += 1

        elif 'left' not in sentence and 'right' in sentence:
            right_indices.append(coordinates)
            index = first_nonzero_plane(label)
            right_index_list.append(index)
            cutoff = midpoint + buffer_dist
            if index > cutoff:
                print(f"right but on left half index {index} midline: {cutoff} label name: {row['Label_Name']}")
                right_cross_midline += 1
                rows_to_return.append(row["Label_Name"])
            elif index < midpoint - buffer_dist:
                number_of_images_correct += 1
            else:
                not_checked += 1
        elif 'left'  in sentence and 'right' in sentence:
            left_and_right_count += 1
            not_checked += 1
        else:
            neither_count += 1
            not_checked += 1


    print("left components")
    print(f"len left: {len(left_indices)}")
    average_components(left_indices)
    print("right comonents")
    print(f"len right: {len(right_indices)}")
    average_components(right_indices)
    print(f"left and right count: {left_and_right_count}")
    print(f"neither_count: {neither_count}")

    print(f"left average index: {sum(left_index_list)/len(left_index_list)}")
    print(f"right average index: {sum(right_index_list)/len(right_index_list)}")

    print(f"left cross midline: {left_cross_midline}")
    print(f"right cross midline: {right_cross_midline}")
    print(rows_to_return)

    print(f"images not checked: {not_checked}")
    print(f"total images correct: {number_of_images_correct}")

    filtered_df = df[df['Label_Name'].isin(rows_to_return)]
    return filtered_df