
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


def pet_suv_conversion_external_v3():

    dir_path = "/mnt/dsb2/BRADSHAWtyler.20240716__201511/RefactoredBags/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/external_testset_v2/"


    df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/Swedish_sentences_with_uw_ids.xlsx")

    files_in_directory = os.listdir(dir_path)

    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])

    missing_pet = 0

    # for file in files_in_directory:
    for index, row in df.iterrows():
        file = row["ID"]
        print(f"index: {index} already_converted: {already_converted } found pet images: {found_pet_images} file: {file} missing pet: {missing_pet}")
        index += 1
        #if index < 24200:
        #    continue
        folder_name_exists = os.path.join(top_nifti_folder, file)
        if os.path.exists(folder_name_exists):
            if any('SUV' in filename for filename in os.listdir(folder_name_exists)):
                found_pet_images += 1
                already_converted += 1
                print("already found this image with SUV")
                continue

        if file not in os.listdir(dir_path):
            print("don't have folder")
            missing_pet += 1
            continue

        if file in dicom_error:
            continue
        directory = os.path.join(dir_path, file)

        random_id = os.listdir(directory)
        if len(random_id) == 1:
            directory = os.path.join(directory, random_id[0])

        date = os.listdir(directory)
        if len(date) == 1:
            directory = os.path.join(directory, date[0])
        else:
            print(f"multiple date files in this folder: {directory}")
        modality = os.listdir(directory)
        if "PT" in modality:
            #directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "PT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue
        #print(directory)
        ref_num = os.listdir(directory)
        if len(ref_num) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            continue
        directory = os.path.join(directory, ref_num[0])
        # print(test_directory)
        type_exam = os.listdir(directory)
        # print(modality)
        # print(test)

        recon_types = os.listdir(directory)
        substrings_to_check = ["WB_CTAC"]
        #print(f"recon_types: {recon_types}")
        # Iterate over each substring and check if it's present in any element of recon_types
        print(f"recon types: {recon_types}")
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            #matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            for matched_recon in recon_types:
                print(f"matched recond: {matched_recon}")

                if "wb_ctac" not in matched_recon or "pet_ac_2d" not in matched_recon:
                    continue

                if matched_recon == None or "fused" in matched_recon.lower() or "mip" in matched_recon.lower():
                    continue
                print(f"matched: {matched_recon}")
                if matched_recon:
                    # If a match is found, build the path
                    top_dicom_folder = os.path.join(directory, matched_recon, file)
                    #top_dicom_folder = os.path.join(directory, matched_recon)

                    #top_dicom_folder = directory + "/" + str(matched_recon) + ""
                    #print(f"top dicom folder: {top_dicom_folder}")

                    try:
                        print("called pet conversion")
                        found_pet_images = call_suv_helper(top_dicom_folder, top_nifti_folder, found_pet_images)
                        break
                    except Exception as e:
                        print(f"error: {e}")
                        continue  # If an error occurs, continue with the next substring