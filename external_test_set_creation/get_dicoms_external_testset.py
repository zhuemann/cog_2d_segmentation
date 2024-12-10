import os
import shutil
import pandas as pd

def copy_matching_folders(df, id_column, source_folder, destination_folder):
    """
    Copies folders matching IDs in the DataFrame from source to destination, preserving structure.

    Parameters:
        df (pd.DataFrame): DataFrame containing IDs.
        id_column (str): Column name in the DataFrame containing the IDs to match.
        source_folder (str): Path to the source folder containing folders to be copied.
        destination_folder (str): Path to the destination folder where folders will be copied.

    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for _, row in df.iterrows():
        folder_id = str(row[id_column])  # Get the ID from the DataFrame
        source_path = os.path.join(source_folder, folder_id)

        if os.path.exists(source_path) and os.path.isdir(source_path):
            destination_path = os.path.join(destination_folder, folder_id)
            shutil.copytree(source_path, destination_path)
            print(f"Copied {source_path} to {destination_path}")
        else:
            print(f"Folder not found for ID {folder_id}: {source_path}")

def get_dicoms_external_testset():


    df = pd.read_excel("/UserData/path_to_move.xlsx")

    destination = "/UserData/Zach_Analysis/swedish_dicoms/"  # Replace with your destination folder

    copy_matching_folders(df, id_column = "ID", source_folder="/mnt/Bradshaw/UW_PET_Data/swedish_dicom/RefactoredBags/", destination)