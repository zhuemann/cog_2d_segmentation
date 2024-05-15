import os
import pandas as pd
import nibabel as nib
import numpy as np

def get_suv_file_names(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    image_path_df = []
    for index, row in df.iterrows():
        petlymph = row["Petlymph"]
        folder_name = str(petlymph)
        image_path = os.path.join(image_path_base, folder_name)
        file_names = os.listdir(image_path)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        image_path = os.path.join(image_path, file_names[index_of_suv[0]])
        image_path_df.append(image_path)

    # Convert the list to a DataFrame
    df = pd.DataFrame(image_path_df, columns=["SUV Names"])

    # Save the DataFrame to an Excel file
    df.to_excel("full_suv_names.xlsx", index=False)


def create_prediction_plots():

    print("hi")


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

def count_left_right_sided(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v2_nifti/"
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1",
                      "PETWB_012579_01_label_2", "PETWB_003190_01_label_3",
                      "PETWB_011401_02_label_3"]
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
        print(f"index: {index}")
        sentence = row["sentence"]
        if row["Label_Name"] in labels_to_skip:
            continue
        #if index > 5000:
        #    break
        label_name = row["Label_Name"] + ".nii.gz"
        label_path = os.path.join(label_path_base, label_name)
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        midpoint = label.shape[0]/2
        buffer_dist = 10
        coordinates = (row["i"], row["j"], row["k"])


        if 'left' in sentence and 'right' not in sentence:
            left_indices.append(coordinates)
            index = first_nonzero_plane(label)
            left_index_list.append(index)
            cutoff = midpoint - buffer_dist
            if index < cutoff:
                print(f"left but on right half index: {index} midline: {cutoff} label name: {row['Label_Name']}")
                left_cross_midline += 1
                rows_to_return.append(row["Label_Name"])
        elif 'left' not in sentence and 'right' in sentence:
            right_indices.append(coordinates)
            index = first_nonzero_plane(label)
            right_index_list.append(index)
            cutoff = midpoint + buffer_dist
            if index > cutoff:
                print(f"right but on left half index {index} midline: {cutoff} label name: {row['Label_Name']}")
                right_cross_midline += 1
                rows_to_return.append(row["Label_Name"])

        elif 'left'  in sentence and 'right' in sentence:
            left_and_right_count += 1
        else:
            neither_count += 1


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

    filtered_df = df[df['Label_Name'].isin(rows_to_return)]
    return filtered_df





def finding_missing_images():


    df_path = "/UserData/UW_PET_Data/UWPETCTWB_Research-Image-Requests_20240311.xlsx"
    df = pd.read_excel(df_path)

    dicom_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    key_substrings_pt = ["wb_3d_mac", "WB_MAC", "wb_ac_3d", "PET_AC_3D", "WB_IRCTAC"]  # Add the rest of your PT substrings here
    key_substrings_ct = ["CTAC", "CT_IMAGES", "WB_Standard"]  # Add any more CT substrings if needed
    folders_not_found = 0
    results = {}
    for index, row in df.iterrows():

        print(f"index: {index}")
        folder_name = row["Coded Accession Number"]
        patient_coding = row["Coded Accession Number"]
        patient_path = os.path.join(dicom_path, folder_name)
        #print(patient_path)
        if os.path.exists(patient_path):
            #patient_path = os.path.join(root_dir, patient_coding)
            if os.path.isdir(patient_path):  # Check if it's a directory
                pt_found = False
                ct_found = False
                pt_path = None
                ct_path = None

                # Traverse the directory structure date->modality->series->exam_name
                for date_folder in os.listdir(patient_path):
                    date_path = os.path.join(patient_path, date_folder)
                    if os.path.isdir(date_path):
                        for modality_folder in os.listdir(date_path):
                            modality_path = os.path.join(date_path, modality_folder)
                            #print(f"modality path: {modality_path}")
                            for exam_folder in os.listdir(modality_path):
                                exam_path = os.path.join(modality_path, exam_folder)
                                if os.path.isdir(exam_path):
                                    # Check if the folder belongs to PT or CT modalities and check the names
                                    if 'PT' in modality_folder.upper():
                                        for series_folder in os.listdir(exam_path):
                                            if any(substring.lower() in series_folder.lower() for substring in
                                                   key_substrings_pt):
                                                pt_found = True
                                                pt_path = os.path.join(exam_path,
                                                                       series_folder)  # Store the full path including the matching folder
                                                break  # Stop searching once a match is found
                                    elif 'CT' in modality_folder.upper():
                                        for series_folder in os.listdir(exam_path):
                                            if any(substring.lower() in series_folder.lower() for substring in
                                                   key_substrings_ct):
                                                ct_found = True
                                                ct_path = os.path.join(exam_path,
                                                                       series_folder)  # Store the full path including the matching folder
                                                break  # Stop searching once a match is found


                # Record the results for this patient coding
                results[patient_coding] = (pt_found, ct_found, pt_path, ct_path)

        else:
            results[patient_coding] = (False, False, None, None)
            folders_not_found += 1
    """
    df = pd.DataFrame(list(results.items()), columns=['Patient_Coding', 'Findings'])
    df[['PET_Found', 'CT_Found']] = pd.DataFrame(df['Findings'].tolist(), index=df.index)
    df.drop('Findings', axis=1, inplace=True)
    """
    file_path = "/UserData/UW_PET_Data/full_accounting_of_pet_ct_found.xlsx"
    df = pd.DataFrame.from_dict(results, orient='index', columns=['PET_Found', 'CT_Found', 'PT_Path', 'CT_Path'])
    # Write the DataFrame to an Excel file
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Patient_Coding'}, inplace=True)
    df.to_excel(file_path, index=False, engine='openpyxl')
    # Write the DataFrame to an Excel file
    #df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"folders not found: {folders_not_found}")


from collections import defaultdict

def analyze_ct_series_when_pt_matches(root_dir, pt_substring):
    ct_series_count = defaultdict(int)  # Dictionary to count occurrences of CT series

    # Iterate over each patient coding folder
    for patient_coding in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_coding)
        if os.path.exists(patient_path) and os.path.isdir(patient_path):  # Check if it's a directory
            # Traverse the directory structure date->modality->exam_name->series
            pt_exam_paths = []  # To store PT exam paths with the required substring
            ct_exam_paths = []  # To store corresponding CT exam paths

            for date_folder in os.listdir(patient_path):
                date_path = os.path.join(patient_path, date_folder)
                if os.path.isdir(date_path):
                    for modality_folder in os.listdir(date_path):
                        modality_path = os.path.join(date_path, modality_folder)
                        if os.path.isdir(modality_path):
                            for exam_folder in os.listdir(modality_path):
                                exam_path = os.path.join(modality_path, exam_folder)
                                if os.path.isdir(exam_path):
                                    # Check PT folders for the specific substring
                                    if 'PT' in modality_folder.upper():
                                        for series_folder in os.listdir(exam_path):
                                            if pt_substring.lower() in series_folder.lower():
                                                pt_exam_paths.append(exam_path)  # Store PT exam path that matches the substring
                                                break  # Stop searching once a match is found in this exam path
                                    elif 'CT' in modality_folder.upper():
                                        ct_exam_paths.append(exam_path)  # Store all CT exam paths

            # If matching PT exam paths are found, count CT series in the corresponding CT exam paths
            if pt_exam_paths:
                # Filter CT paths to include only those corresponding to dates of PT matches
                corresponding_ct_paths = [path for path in ct_exam_paths if any(pt_path.split(os.sep)[-3] == path.split(os.sep)[-3] for pt_path in pt_exam_paths)]
                for ct_path in corresponding_ct_paths:
                    for series_folder in os.listdir(ct_path):
                        ct_series_count[series_folder] += 1

    return ct_series_count



