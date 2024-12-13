
import pydicom
import numpy as np
from platipy.dicom.io import rtstruct_to_nifti
import pandas as pd
import os
import regex as re
from datetime import datetime


def get_folder_by_index_v1(folder_list, input_string, index):
    # Extract the identifier from the input string
    identifier = input_string.replace('_', '.')

    # Compile a regex to match folders containing the identifier and a date
    #pattern = re.compile(rf"{re.escape(identifier)}_(\d{{4}}-\d{{2}}-\d{{2}})")
    pattern = re.compile(rf"{re.escape(identifier)}_(\d{{4}}-\d{{2}}-\d{{2}})_.*")

    print(f"identifier: {identifier}")
    print(folder_list)
    # Filter folders matching the identifier and extract their dates
    matching_folders = []
    for folder in folder_list:
        match = pattern.search(folder)
        if identifier in folder:
            print("found")
        if match:
            date_str = match.group(1)  # Extract the date part
            matching_folders.append((folder, datetime.strptime(date_str, "%Y-%m-%d")))

    print(f"matching folders: {matching_folders}")
    # Sort the folders by date (oldest to newest)
    sorted_folders = sorted(matching_folders, key=lambda x: x[1])
    print(sorted_folders)
    # Check if the index is within range
    if index - 1 < len(sorted_folders):
        # Return the folder name at the specified index (1-based index)
        return sorted_folders[index - 1][0]
    else:
        # If the index is out of range, return None or handle as needed
        return None
def get_folder_by_index(folder_list, input_string, index):
    # Extract the identifier from the input string
    identifier = input_string.replace('_', '.')

    print(f"identifier: {identifier}")
    print(folder_list)

    # Filter folders matching the identifier and extract their dates
    matching_folders = []
    for folder in folder_list:
        if identifier in folder:  # Check if the identifier exists in the folder name
            print("found")
            # Extract the date part by locating the pattern YYYY-MM-DD
            match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", folder)
            if match:
                date_str = match.group(0)  # Extract the matched date
                try:
                    matching_folders.append((folder, datetime.strptime(date_str, "%Y-%m-%d")))
                except ValueError:
                    print(f"Invalid date format in folder name: {folder}")
                    continue
            else:
                print(f"No valid date found in folder name: {folder}")

    print(f"matching folders: {matching_folders}")

    # Sort the folders by date (oldest to newest)
    sorted_folders = sorted(matching_folders, key=lambda x: x[1])
    print(sorted_folders)

    # Check if the index is within range
    if index - 1 < len(sorted_folders):
        # Return the folder name at the specified index (1-based index)
        return sorted_folders[index - 1][0]
    else:
        # If the index is out of range, return None or handle as needed
        print(f"Index {index} out of range for identifier: {identifier}")
        return None
def process_rt_strcuts_to_nifty():

    df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Meghan_worksheet_returned.xlsx")

    dicom_location_base = "/UserData/Zach_Analysis/physican_labeling_UWPET/dicom_folders/"

    rt_location = "/UserData/Zach_Analysis/physican_labeling_UWPET/rt_structs_meg_structured/"
    folder_list = os.listdir(rt_location)
    nifti_save_path = "/UserData/Zach_Analysis/physican_labeling_UWPET/meg_nifti/"
    # Compile a regex pattern to match the desired parts of the filename
    pattern = re.compile(r'UWPETCTWB\.(\d+)_UWPETCTWB\.\1_RTst_(\d{4}-\d{2}-\d{2})')

    order_index = 1
    previous_id = "PETWB_000000_00"

    for _, row in df.iterrows():
    #for folder in folder_list:

        #match = pattern.search(folder)
        #if match:
        #    file_id = match.group(1)  # Extract the XXXXXX
        #    date = match.group(2)  # Extract the Date

        patient_id = row["Coded Patient ID"]
        petlymph = row["id"]
        dicom_series_path_pet = os.path.join(dicom_location_base, petlymph, "pet")

        print(f"patient_id: {patient_id}")

        #if petlymph == previous_id:
        #    print("keep order index")
        if petlymph.split("_")[1] != previous_id.split("_")[1]: # if we moved on to a new patient set index to 0
            order_index = 1
        elif int(petlymph.split("_")[2]) > int(previous_id.split("_")[2]): # if the image is higher increment order
            order_index += 1

        print(f"order_index: {order_index}")
        folder_name = get_folder_by_index(folder_list = folder_list, input_string = patient_id, index = order_index)
        print(f"folder name: {folder_name}")
        dicom_path_RT_folder = os.path.join(rt_location, folder_name)

        dicom_file_name = os.listdir(dicom_path_RT_folder)[0]
        dicom_path_RT = os.path.join(dicom_path_RT_folder, dicom_file_name)

        save_rt_struct_path = os.path.join(nifti_save_path, petlymph)

        rtstruct_to_nifti.convert_rtstruct(dicom_series_path_pet, dicom_path_RT, save_rt_struct_path)