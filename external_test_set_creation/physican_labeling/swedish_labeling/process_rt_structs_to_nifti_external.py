
import pydicom
import numpy as np
from platipy.dicom.io import rtstruct_to_nifti
import pandas as pd
import os
import regex as re
from datetime import datetime

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
            # Split the folder name to locate the date part
            parts = folder.split("_")  # Split by underscores
            for part in parts:
                if part.startswith("20") and len(part) == 10:  # Look for 'YYYY-MM-DD'
                    try:
                        # Attempt to parse the date
                        date = datetime.strptime(part, "%Y-%m-%d")
                        matching_folders.append((folder, date))
                        break  # Stop once a valid date is found
                    except ValueError:
                        print(f"Invalid date format: {part}")
                        continue

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
def process_rt_strcuts_to_nifty_external():

    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Meghan_worksheet_returned.xlsx")
    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Steve_worksheet_returned.xlsx")
    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Josh_worksheet_returned.xlsx")
    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/mim_manual_labeling.xlsx")

    #dicom_location_base = "/UserData/Zach_Analysis/physican_labeling_UWPET/dicom_folders/"
    dicom_location_base = "/UserData/Zach_Analysis/swedish_dicoms"
    dicom_location_base = "/UserData/Zach_Analysis/upload_to_mim2/"
    #rt_location = "/UserData/Zach_Analysis/physican_labeling_UWPET/rt_structs_meg_structured/"
    #rt_location = "/UserData/Zach_Analysis/physican_labeling_UWPET/rt_structs_steve_structured/"
    rt_location = "/UserData/Zach_Analysis/physican_labeling_UWPET/swedish_labeled_dataset/rt_structs_external_structured/"


    folder_list = os.listdir(rt_location)
    #nifti_save_path = "/UserData/Zach_Analysis/physican_labeling_UWPET/steve_nifti/"
    nifti_save_path = "/UserData/Zach_Analysis/physican_labeling_UWPET/swedish_labeled_dataset/swedish_nifti/"

    for dicom_name in folder_list:

        rt_path = os.path.join(rt_location, dicom_name)
        rt_file = pydicom.dcmread(rt_path)
        rt_file_name = rt_file["00080050"][:]

        dicom_series_path_pet = os.path.join(dicom_location_base, "swedish_" + rt_file_name, "PT")
        save_rt_struct_path = os.path.join(nifti_save_path, rt_file_name)

        rtstruct_to_nifti.convert_rtstruct(dicom_series_path_pet, rt_path, save_rt_struct_path)