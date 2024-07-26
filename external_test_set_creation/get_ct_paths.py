
import os
import pandas as pd
import dicom2nifti
import nibabel as nib
from datetime import datetime
import numpy as np
import glob
import pydicom

from nilearn.image import resample_img
from pathlib import Path

def ct_check():

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb3/"
    #path_list = ["/mnt/Bradshaw/UW_PET_Data/dsb2b/", "/mnt/Bradshaw/UW_PET_Data/dsb2c/" ,"/mnt/Bradshaw/UW_PET_Data/dsb3/"]
    path_list = ["/mnt/dsb2/BRADSHAWtyler.20240716__201511/RefactoredBags"]

    master_dic = {}
    for dir_path in path_list:
        files_in_directory = os.listdir(dir_path)

        print(f"files in folder: {len(files_in_directory)}")
        no_pt_files_list = []
        index = 0

        num_dates = {}  # key is number of dates in folder value is how many folders have that value
        num_dates[1] = 0
        num_modality = {"PT": 0, "CT": 0, "extra": 0}
        num_study_names = {1: 0, "extra": 0, 0: 0}

        found_cts = 0
        already_found = 0

        for file in files_in_directory:
            # print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
            # if index > 10:
            #    continue
            if index % 500 == 0:
                print(f"index: {index} already found: {already_found} found cts: {found_cts} filename: {file} path: {dir_path}")
            # print(file)
            index += 1

            directory = os.path.join(dir_path, file)
            if not os.path.exists(directory):
                continue
            date = os.listdir(directory)
            if len(date) == 1:
                directory = os.path.join(directory, date[0])
                num_dates[1] += 1
            else:
                print(f"multiple date files in this folder: {directory}")
                if len(date) not in num_dates:
                    num_dates[len(date)] = 1
                else:
                    num_dates[len(date)] += 1

            modality = os.listdir(directory)

            if len(modality) > 2:
                num_modality["extra"] += 1
            if "CT" in modality:
                num_modality["CT"] += 1
            #else:
            #    continue
            if "CT" in modality:
                directory = os.path.join(directory, "CT")
            #else:
                #print(f"file: {file} does not have Pet scan")
                #continue

            study_name = os.listdir(directory)
            if len(study_name) == 0:
                print(f"something funny: {file}")
                no_pt_files_list.append(file)
                num_study_names[0] += 1
            elif study_name == 1:
                num_study_names[1] += 1
            else:
                num_study_names["extra"] += 1

            directory = os.path.join(directory, study_name[0])
            recon_types = os.listdir(directory)
            recon_path_base = os.path.join(dir_path,file,date[0],modality[0],study_name[0])
            for recon in recon_types:
                if file in master_dic:
                    master_dic[file].append(recon_path_base + "/" + recon)
                else:
                    master_dic[file] = [recon_path_base + "/" + recon]

    # Convert dictionary to a list of lists for DataFrame
    data = []
    for file, strings in master_dic.items():
        row = [file] + strings
        data.append(row)

    # Find the maximum length of the lists (to ensure uniform columns)
    max_len = max(len(row) for row in data)

    # Pad lists with empty strings so all rows have the same length
    for row in data:
        while len(row) < max_len:
            row.append('')

    # Create DataFrame
    df = pd.DataFrame(data)

    # Define the column names
    columns = ['File'] + [f'String {i + 1}' for i in range(max_len - 1)]
    df.columns = columns
    #print(df)
    path = '/UserData/Zach_Analysis/visual_ground_file_lists/external_testset_ct_full.xlsx'
    print(path)
    # Save DataFrame to Excel file
    df.to_excel(path, index=False)