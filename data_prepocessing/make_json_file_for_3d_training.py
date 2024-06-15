import os
import random
import json
import numpy as np
import pandas as pd

def get_suv_images_list(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4/"
    image_path_df = []
    for index, row in df.iterrows():
        petlymph = row["Petlymph"]
        folder_name = str(petlymph)
        image_path = os.path.join(image_path_base, folder_name)
        file_names = os.listdir(image_path)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        image_path = os.path.join(image_path, file_names[index_of_suv[0]])
        image_path_df.append(image_path)

    # Convert the list to a DataFrame
    df = pd.DataFrame(image_path_df, columns=["SUV Names"])

    # Save the DataFrame to an Excel file
    df.to_excel("/UserData/Zach_Analysis/suv_slice_text/full_suv_names_uw_pet_4.xlsx", index=False)
    return df

def filter_dataframe_for_images_in_folder(df):
    # Define your target folder
    target_folder = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4/"

    # Get a list of all files in the target folder
    files = os.listdir(target_folder)

    # Filter the dataframe to keep only rows where the ID in 'Petlymph' is found in any file name in the folder
    df_filtered = df[df['Petlymph'].apply(lambda id: any(id in file for file in files))]

    # Reset the index of the filtered dataframe
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered

def filter_dataframe_based_on_files(df, column):
    """Filter a DataFrame to only include rows where the column matches label names in a specified folder."""
    # Hardcoded directory
    target_folder = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels4/"

    # Retrieve label names from files in the specified directory, stripping the '.nii.gz'.
    labels = [file.replace(".nii.gz", "") for file in os.listdir(target_folder) if file.endswith(".nii.gz")]

    # Filter the DataFrame to only include rows where the column matches one of the labels.
    return df[df[column].isin(labels)]

def make_json_file_for_3d_training(df):
    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels4/"

    print(f"length of dataframe before: {len(df)}")
    df = filter_dataframe_for_images_in_folder(df)
    print(f"after image filtering: {len(df)}")
    df = filter_dataframe_based_on_files(df, column = "Label_Name")
    print(f"after label filtering: {len(df)}")
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1",
                      "PETWB_012579_01_label_2", "PETWB_003190_01_label_3", "PETWB_013006_03_label_2"]
    df = df[~df["Label_Name"].isin(labels_to_skip)]

    print(f"length of dataframe final: {len(df)}")

    # Assuming df_2d is your main DataFrame and image_base, label_path_base, image_list are defined
    np.random.seed(42)  # For reproducibility

    # Unique list of PetlympIDs
    unique_petlymps = df['Petlymph'].unique()
    np.random.shuffle(unique_petlymps)  # Shuffle the array

    # Split the IDs into training, validation, and test sets
    train_split = int(len(unique_petlymps) * 0.7)
    val_split = int(len(unique_petlymps) * 0.9)

    train_ids = unique_petlymps[:train_split]
    val_ids = unique_petlymps[train_split:val_split]
    test_ids = unique_petlymps[val_split:]

    # Initialize the storage dictionary
    data = {'training': [], 'validation': [], 'testing': []}

    # Now iterate through the DataFrame
    for index, row in df.iterrows():
        petlymph = row["Petlymph"]
        folder_name = str(petlymph)  # .lower() #+ "_" + str(petlymph).lower()

        image_path = os.path.join(image_base, petlymph + "_suv_cropped.nii.gz")
        image2_path = os.path.join(image_base, petlymph + "_ct_cropped.nii.gz")


        label_name = row["Label_Name"]
        label_path = os.path.join(label_path_base, label_name + ".nii.gz")

        # Decide the data split based on PetlympID
        entry = {
            "image": image_path,
            "image2": image2_path,
            "label": label_path,
            "report": row["sentence"],
            "slice_num": row["Slice"],
            "suv_num": row["SUV"],
            "label_name": label_name
        }
        if petlymph in train_ids:
            # Optionally randomize folds within training data
            entry['fold'] = random.randint(0, 5)
            data['training'].append(entry)
        elif petlymph in val_ids:
            data['validation'].append(entry)
        else:
            data['testing'].append(entry)

    # Write the JSON data to a file
    with open('/UserData/Zach_Analysis/uw_lymphoma_pet_3d/output_resampled_13000.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)  # Use indent=4 for pretty printing