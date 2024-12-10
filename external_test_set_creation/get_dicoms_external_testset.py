import os
import shutil
import pandas as pd
from pathlib import Path

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


def copy_and_reorganize_files(dataframe, destination_folder):
    """
    Copies and reorganizes folders based on PET and CT paths in the dataframe.
    Creates a structure: destination_folder -> swedish_<ID> -> PT and CT.
    Only creates ID folder if both PT and CT exist.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing 'ID', 'PT', and 'CT' columns.
        destination_folder (str): Path to the folder where files will be copied.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for _, row in dataframe.iterrows():
        id_ = row['ID']
        pt_path = row['PT']
        ct_path = row['CT']

        # Skip if either PT or CT is missing
        if pd.isna(pt_path) or pd.isna(ct_path):
            print(f"Skipping ID {id_}: Missing PT or CT path.")
            continue
        pt_path = pt_path.replace("dicom_to_copy", "swedish_dicoms")
        ct_path = ct_path.replace("dicom_to_copy", "swedish_dicoms")
        pt_path = pt_path.replace("Z:", "/UserData")
        ct_path = ct_path.replace("Z:", "/UserData")
        pt_path = str(Path(pt_path))
        ct_path = str(Path(ct_path))
        pt_path = "/UserData/Zach_Analysis/swedish_dicoms/"
        # Create swedish_<ID> directory in the destination folder
        id_dest_folder = os.path.join(destination_folder, f"swedish_{id_}")
        if not os.path.exists(id_dest_folder):
            os.makedirs(id_dest_folder)

        # Process PT
        if os.path.exists(pt_path):
            pt_subfolders = os.listdir(pt_path)
            if pt_subfolders:  # Ensure the directory is not empty
                pt_parent_folder = os.path.join(pt_path, pt_subfolders[0])
                if os.path.exists(pt_parent_folder):
                    pt_dest_folder = os.path.join(id_dest_folder, "PT")
                    shutil.copytree(pt_parent_folder, pt_dest_folder)
                    print(f"Copied PT folder for ID {id_} to {pt_dest_folder}")
            else:
                print(f"Skipping PT for ID {id_}: PT path is empty.")
        else:
            print(f"pet path did not exist")
            print(f"pet path: {pt_path}")

        # Process CT
        if os.path.exists(ct_path):
            ct_subfolders = os.listdir(ct_path)
            if ct_subfolders:  # Ensure the directory is not empty
                ct_parent_folder = os.path.join(ct_path, ct_subfolders[0])
                if os.path.exists(ct_parent_folder):
                    ct_dest_folder = os.path.join(id_dest_folder, "CT")
                    shutil.copytree(ct_parent_folder, ct_dest_folder)
                    print(f"Copied CT folder for ID {id_} to {ct_dest_folder}")
            else:
                print(f"Skipping CT for ID {id_}: CT path is empty.")

def get_dicoms_external_testset():


    df = pd.read_excel("/UserData/Zach_Analysis/path_to_move.xlsx")

    destination = "/UserData/Zach_Analysis/swedish_dicoms/"  # Replace with your destination folder

    #copy_matching_folders(df, id_column = "ID", source_folder="/mnt/Bradshaw/UW_PET_Data/swedish_dicom/RefactoredBags/", destination_folder=destination)

    # Define the destination folder for moved files
    destination = "/UserData/Zach_Analysis/upload_to_mim2/"  # Replace with your destination folder

    # Run the function
    copy_and_reorganize_files(df, destination)