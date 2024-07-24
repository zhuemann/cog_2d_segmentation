
import os

import pandas as pd
import pydicom

def get_scanner_type(path):

    dicom_files = os.listdir(path)
    ds = pydicom.dcmread(os.path.join(path, dicom_files[0]))

    # Retrieve manufacturer and model name
    manufacturer = ds.get('Manufacturer', 'Manufacturer not found')
    model_name = ds.get('ManufacturerModelName', 'Model name not found')
    return model_name

def scanner_types():

    all_scans = {}

    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    files_in_directory = os.listdir(dir_path)


    files_used = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/id_pet_used_ct_used.xlsx")
    files_in_dataset = []


    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])
    skipped_files = 0

    #for file in files_in_directory:
    for _, row in files_used.iterrows():
        path = row["PET_Used"]
        path_parts = path.split(os.sep)
        file = path_parts[-2]
        print(f"index: {index} already_converted: {already_converted } found pet images: {found_pet_images} file: {file}")
        index += 1
        #if index < 24200:
        #    continue
        folder_name_exists = os.path.join(top_nifti_folder, file)
        """
        if os.path.exists(folder_name_exists):
            if any('SUV' in filename for filename in os.listdir(folder_name_exists)):
                found_pet_images += 1
                already_converted += 1
                print("already found this image with SUV")
                continue
        """
        files_in_dir = os.listdir(dir_path)
        if file not in files_in_dir:
            skipped_files += 1
            continue

        if file in dicom_error:
            continue
        directory = os.path.join(dir_path, file)
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
        substrings_to_check = ["wb_3d_mac", "WB_MAC", "wb_ac_3d", "PET_AC_3D", "WB_IRCTAC"]
        # Iterate over each substring and check if it's present in any element of recon_types
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            if matched_recon:
                # If a match is found, build the path
                top_dicom_folder = os.path.join(directory, matched_recon)
                try:
                    scanner_type = get_scanner_type(top_dicom_folder)
                    if scanner_type in all_scans:
                        all_scans[scanner_type] += 1
                    else:
                        all_scans[scanner_type] = 1
                except:
                    continue  # If an error occurs, continue with the next substring

    print(all_scans)
    print(f"files skipped because not in dsb2b: {skipped_files}")


def scanner_types_external_test_set():

    all_scans = {}

    #top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/2024-07/"
    files_in_directory = os.listdir(dir_path)


    files_used = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/id_pet_used_ct_used.xlsx")
    files_in_dataset = []


    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])
    skipped_files = 0

    num_dates = {} # key is number of dates in folder value is how many folders have that value
    num_dates[1] = 0
    num_modality = {"PT": 0, "CT": 0, "extra": 0}
    num_study_names = {1:0 , "extra": 0, 0: 0}
    types_of_scans_ct = {}
    types_of_scans_pt = {}
    study_names = {}
    #for file in files_in_directory:
    for file in files_in_directory:
        print(f"index: {index}")
        index += 1
        directory = os.path.join(dir_path, file)
        print(directory)
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
        # print(f"modality: {modality}")
        if len(modality) > 2:
            num_modality["extra"] += 1
        if "CT" in modality:
            num_modality["CT"] += 1
        # else:
        # print(f"file: {file} does not have ct scan modality: {modality}")
        #    continue

        if "PT" in modality:
            # directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "PT")
            num_modality["PT"] += 1
        else:
            # print(f"file: {file} does not have Pet scan modality: {modality}")
            continue

        """
        if "CT" in modality:
            # directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "CT")
            num_modality["CT"] += 1
        else:
            #print(f"file: {file} does not have Pet scan modality: {modality}")
            continue
        """

        # print(directory)
        study_name = os.listdir(directory)
        # print(study_name)
        if len(study_name) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            num_study_names[0] += 1
            # continue
        elif study_name == 1:
            num_study_names[1] += 1
        else:
            num_study_names["extra"] += 1

        directory = os.path.join(directory, study_name[0])
        recon_types = os.listdir(directory)
        print(recon_types)
        for matched_recon in recon_types:
            top_dicom_folder = os.path.join(directory, matched_recon)
            if matched_recon in study_names:
                study_names[matched_recon] += 1
            else:
                study_names[study_names] = 1
            try:
                scanner_type = get_scanner_type(top_dicom_folder)
                if scanner_type in all_scans:
                    all_scans[scanner_type] += 1
                else:
                    all_scans[scanner_type] = 1
            except:
                print("contining")
                continue

    print(all_scans)
    print(study_names)
    print(f"files skipped because not in dsb2b: {skipped_files}")