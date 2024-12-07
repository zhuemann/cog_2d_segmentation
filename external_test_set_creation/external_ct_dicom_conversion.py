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

from uw_pet_suv_conversion import call_suv_helper

def uw_ct_conversion_external_dataset_v2():

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2c/"

    dir_path = "/mnt/Bradshaw/UW_PET_Data/2024-07-CT/"
    dir_path_suv = "/mnt/Bradshaw/UW_PET_Data/external_testset_v2/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/external_testset/"

    #dir_path = "/mnt/dsb2/BRADSHAWtyler.20240716__201511/RefactoredBags/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/swedish_dicom/RefactoredBags/"
    #top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/external_testset_v2/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/external_testset_try4/"

    #df = pd.read_excel(
    #    "/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/Swedish_sentences_with_uw_ids.xlsx")

    df = pd.read_excel( "/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/additional_sentences.xlsx")

    #files_in_directory = os.listdir(dir_path_suv)
    files_in_directory = os.listdir(dir_path)

    print(f"files in folder: {len(files_in_directory)}")
    no_pt_files_list = []
    index = 0

    missing_inject_info = 0
    potential_suv_images = 0

    num_dates = {}  # key is number of dates in folder value is how many folders have that value
    num_dates[1] = 0
    num_modality = {"PT": 0, "CT": 0, "extra": 0}
    num_study_names = {1: 0, "extra": 0, 0: 0}

    found_cts = 0
    already_found = 0
    matches_dic = {}
    #for file in files_in_directory:
    for index, row in df.iterrows():
        file = row["ID"]
        # print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
        # if index > 10:
        #    continue
        print(f"index: {index} already found: {already_found} found cts: {found_cts} filename: {file}")
        # print(file)
        index += 1
        # if index > 100:
        #    break

        """
        folder_name_exists = os.path.join(top_nifti_folder, file)
        if os.path.exists(folder_name_exists):
            print(folder_name_exists)
            list_of_folders = os.listdir(folder_name_exists)
            if any('CT' in filename for filename in list_of_folders) and not any("IRCTAC" in filename for filename in list_of_folders):
                found_cts += 1
                already_found += 1
                print("already found this image with CT")
                continue
        """


        suv_dims = (0, 0, 0)
        """
        suv_path = os.path.join(dir_path_suv, file)
        if not os.path.exists(suv_path):
            continue
        for filename in os.listdir(suv_path):
            if filename.endswith(".nii.gz") and "suv" in filename.lower():
                filepath = os.path.join(suv_path, filename)
                try:
                    # Load the NIfTI file
                    nii = nib.load(filepath)
                    suv_dims = nii.header.get_data_shape()
                except:

                    print("can't get dimensions from suv")
        """


        # print(f"suv_dims: {suv_dims}")
        directory = os.path.join(dir_path, file)
        if not os.path.exists(directory):
            print("path does not exist")
            continue

        random_id = os.listdir(directory)
        if len(random_id) == 1:
            directory = os.path.join(directory, random_id[0])

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
        #    print("too many modalities")
        #    continue
        if "CT" in modality:
            directory = os.path.join(directory, "CT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue

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
        print(f"recon types: {recon_types}")
        substrings_to_check = ["Cor_Head_In_3.75thk", "Body-Low_Dose", "Body-ldCT_700m", "Cor_Head_In_3.75_thk", "Cor_Head_In_3.75_thk"]
        # Iterate over each substring and check if it's present in any element of recon_types
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            if matched_recon:
                # If a match is found, build the path
                top_dicom_folder = os.path.join(directory, matched_recon, file)
                z = len(os.listdir(top_dicom_folder))
                # checks if slices line up other wise don't convert and keep searching
                print(f"matched recon: {matched_recon}")
                #if z == suv_dims[2]:
                if True:
                    # Perform your additional logic or function calls here
                    try:
                        found_cts = call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts)
                    except:
                        print("tried and errored")
                        continue  # If an error occurs, continue with the next substring