import os
import shutil
import pandas as pd

def copy_physican_labels_to_folder():

    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Meghan_worksheet_matched.xlsx")
    df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Steve_worksheet_matched.xlsx")
    # Define the base folder for the new location
    #new_location_folder = "/mnt/Bradshaw/UW_PET_Data/physican_labels/meghan_labels/"
    new_location_folder = "/mnt/Bradshaw/UW_PET_Data/physican_labels/steve_labels/"


    #original_location_base = "/UserData/Zach_Analysis/physican_labeling_UWPET/meg_nifti_v2/"
    original_location_base = "/UserData/Zach_Analysis/physican_labeling_UWPET/steve_nifti/"

    # Iterate through the dataframe and copy files with new names
    for index, row in df.iterrows():
        original_path = row["File_Name"]
        new_name = row["Label_Name"]
        new_name = new_name + ".nii.gz"
        destination_path = os.path.join(new_location_folder, new_name)

        original_path = os.path.join(original_location_base, original_path)
        try:
            # Copy the file (not move)
            shutil.copy2(original_path, destination_path)
            print(f"Copied {original_path} to {destination_path}")
        except FileNotFoundError:
            print(f"File not found: {original_path}")
        except Exception as e:
            print(f"Error copying {original_path} to {destination_path}: {e}")