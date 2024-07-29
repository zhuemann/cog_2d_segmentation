
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

def get_tracer_type(path):
    dicom_files = os.listdir(path)
    ds = pydicom.dcmread(os.path.join(path, dicom_files[0]))

    # Retrieve Radiopharmaceutical Information Sequence
    radiopharmaceutical_info = ds.get('RadiopharmaceuticalInformationSequence', None)

    if radiopharmaceutical_info and len(radiopharmaceutical_info) > 0:
        radiopharmaceutical_info = radiopharmaceutical_info[0]
        tracer_type = radiopharmaceutical_info.get('Radiopharmaceutical', 'Tracer type not found')
    else:
        tracer_type = 'Tracer type not found'

    return tracer_type

def tracer_type_all_files():

    all_scans = {}

    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2c/"
    files_in_directory = os.listdir(dir_path)


    files_used = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/id_pet_used_ct_used.xlsx")
    files_in_dataset = []


    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])
    skipped_files = 0

    id_list = []
    tracer_list = []
    scanner_list = []

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
                    tracer_type = get_tracer_type(top_dicom_folder)
                    scanner_type = get_scanner_type(top_dicom_folder)
                    print(f"tracer type: {tracer_type}")
                    id_list.append(file)
                    tracer_list.append(tracer_type)
                    scanner_list.append(scanner_type)
                except:
                    continue  # If an error occurs, continue with the next substring

    #print(all_scans)
    print(f"files skipped because not in dsb2b: {skipped_files}")
    # Create a dataframe
    df = pd.DataFrame({
        'ID': id_list,
        'Tracer': tracer_list,
        'Scanner Type': scanner_list
    })

    # Save the dataframe to an Excel file
    df.to_excel('/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/tracer_and_scanner_dsb2b.xlsx', index=False)


