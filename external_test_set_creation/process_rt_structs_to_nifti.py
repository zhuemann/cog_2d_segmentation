
import pydicom
import numpy as np
from platipy.dicom.io import rtstruct_to_nifti
import pandas as pd
import os
import re


def get_folder_by_index(folder_list, input_string, index):
    # Extract the identifier from the input string
    identifier = input_string.replace('_', '.')

    # Compile a regex to match folders containing the identifier and a date
    pattern = re.compile(rf"{re.escape(identifier)}_(\d{{4}}-\d{{2}}-\d{{2}})")

    # Filter folders matching the identifier and extract their dates
    matching_folders = []
    for folder in folder_list:
        match = pattern.search(folder)
        if match:
            date_str = match.group(1)  # Extract the date part
            matching_folders.append((folder, datetime.strptime(date_str, "%Y-%m-%d")))

    # Sort the folders by date (oldest to newest)
    sorted_folders = sorted(matching_folders, key=lambda x: x[1])

    # Check if the index is within range
    if index - 1 < len(sorted_folders):
        # Return the folder name at the specified index (1-based index)
        return sorted_folders[index - 1][0]
    else:
        # If the index is out of range, return None or handle as needed
        return None

def process_rt_strcuts_to_nifty():

    df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Meghan_worksheet_returned.xlsx")

    dicom_location_base = "/UserData/Zach_Analysis/physican_labeling_UWPET/dicom_folders/"

    folder_list = "/UserData/Zach_Analysis/physican_labeling_UWPET/rt_structs_meg_structured/"
    nifti_save_path = "/UserData/Zach_Analysis/physican_labeling_UWPET/meg_nifti/"
    # Compile a regex pattern to match the desired parts of the filename
    pattern = re.compile(r'UWPETCTWB\.(\d+)_UWPETCTWB\.\1_RTst_(\d{4}-\d{2}-\d{2})')

    order_index = 0
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



        #if petlymph == previous_id:
        #    print("keep order index")
        if petlymph.split("_")[1] != previous_id.split("_")[1]: # if we moved on to a new patient set index to 0
            order_index = 0
        elif int(petlymph.split("_")[2]) > int(previous_id.split("_")[2]): # if the image is higher increment order
            order_index += 1

        print(f"order_index: {order_index}")
        folder_name = get_folder_by_index(folder_list = folder_list, input_string = patient_id, index = order_index)

        dicom_path_RT_folder = os.path.join(folder_list, folder_name)
        dicom_file_name = os.listdir(dicom_path_RT_folder)
        dicom_path_RT = os.path.join(dicom_path_RT_folder, dicom_file_name)

        save_rt_struct_path = os.path.join(nifti_save_path, petlymph)

        rtstruct_to_nifti.convert_rtstruct(dicom_series_path_pet, dicom_path_RT, save_rt_struct_path)