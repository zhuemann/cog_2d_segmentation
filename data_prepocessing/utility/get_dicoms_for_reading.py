
import os
import pandas as pd
import shutil

def get_dicoms_for_reading():


    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    # path_list = ["/mnt/Bradshaw/UW_PET_Data/dsb2b/", "/mnt/Bradshaw/UW_PET_Data/dsb2c/" ,"/mnt/Bradshaw/UW_PET_Data/dsb3/"]
    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2c/"
    files_in_directory = os.listdir(dir_path)
    destination_dir = "/UserData/Zach_Analysis/physican_labeling_UWPET/dicom_folders/"

    #files_used = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/id_pet_used_ct_used.xlsx")
    files_used = pd.read_excel("/UserData/Zach_Analysis/final_testset_evaluation_vg/all_labels_jdw.xlsx")

    files_in_dataset = []


    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])
    skipped_files = 0

    #for file in files_in_directory:
    for _, row in files_used.iterrows():
        """
        path = row["PET_Used"]

        path_parts = path.split(os.sep)
        file = path_parts[-2]
        print(f"index: {index} already_converted: {already_converted } found pet images: {found_pet_images} file: {file}")
        index += 1
        #if index < 24200:
        #    continue
        folder_name_exists = os.path.join(top_nifti_folder, file)
        """
        """
        if os.path.exists(folder_name_exists):
            if any('SUV' in filename for filename in os.listdir(folder_name_exists)):
                found_pet_images += 1
                already_converted += 1
                print("already found this image with SUV")
                continue
        """
        file = row["Petlymph"]
        files_in_dir = os.listdir(dir_path)
        if file not in files_in_dir:
            skipped_files += 1
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
                print(top_dicom_folder)
                new_destination = os.path.join(destination_dir, file)
                if os.path.exists(new_destination):
                    continue
                shutil.copytree(top_dicom_folder, new_destination)
                """
                try:
                    #scanner_type = get_scanner_type(top_dicom_folder)
                    #if scanner_type in all_scans:
                    #    all_scans[scanner_type] += 1
                    #else:
                    #    all_scans[scanner_type] = 1
                    shutil.copytree(top_dicom_folder, destination_dir)
                    print("inside try")
                except:
                    print("error happened")
                    continue  # If an error occurs, continue with the next substring
                """