
import os
import pandas as pd
import regex as re

def rename_nifti_files():

    df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Meghan_worksheet_returned.xlsx")
    folder_path = "/UserData/Zach_Analysis/physican_labeling_UWPET/meg_nifti/"
    """
        Matches the label identifiers in the DataFrame with the corresponding filenames in the folder
        and adds a new column to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'Label_Name' column.
            folder_path (str): Path to the folder containing .nii.gz files.

        Returns:
            pd.DataFrame: Updated DataFrame with a new column 'File_Name'.
        """
    # Extract all .nii.gz files from the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz")]

    # Prepare a new column for the matching file names
    matched_files = []

    for label in df['Label_Name']:
        # Extract parts from the label name
        match = re.match(r"PETWB_(\d+)_(\d+)_label_(\d+)", label)
        if not match:
            matched_files.append(None)
            continue

        coded_patient_id, image_number, label_identifier = match.groups()
        label_identifier = int(label_identifier)

        # Find matching files for the same coded_patient_id and image_number
        matching_files = [
            f for f in files if re.match(rf"PETWB_{coded_patient_id}_{image_number}.*", f)
        ]

        # Sort the matching files by their index
        matching_files = sorted(
            matching_files,
            key=lambda x: int(re.search(r"(ROI|Finding)_(-?\d+)", x).group(2)) if re.search(r"(ROI|Finding)_(-?\d+)",
                                                                                            x) else float("inf")
        )

        # Match the label_identifier to the ascending index
        if label_identifier <= len(matching_files):
            matched_files.append(matching_files[label_identifier - 1])
        else:
            matched_files.append(None)

    # Add the matched file names as a new column
    df['File_Name'] = matched_files

    df.to_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Meghan_worksheet_matched.xlsx")
