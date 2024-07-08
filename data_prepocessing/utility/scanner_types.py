
import os
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

    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])

    for file in files_in_directory:
        print(f"index: {index} already_converted: {already_converted } found pet images: {found_pet_images} file: {file}")
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