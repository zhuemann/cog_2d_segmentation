import json
import pandas as pd
import random
import os
import numpy as np

def dataframe_to_json(df):

    print(fail use the other make json file for 3d training)
    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images/"
    label_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels/"
    dropped_missing_files = 0
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1",
                      "PETWB_012579_01_label_2", "PETWB_003190_01_label_3"]
    df_2d = df[~df["Label_Name"].isin(labels_to_skip)]

    # add patient id so we can split patient wise
    df_2d['PatientID'] = df_2d['Petlymph'].apply(lambda x: x.split('_')[1])

    # Unique list of PatientIds
    unique_petlymps = df_2d['PatientID'].unique()
    np.random.shuffle(unique_petlymps)  # Shuffle the array

    # Split the IDs into training, validation, and test sets
    train_split = int(len(unique_petlymps) * 0.8)
    val_split = int(len(unique_petlymps) * 0.9)

    train_ids = unique_petlymps[:train_split]
    val_ids = unique_petlymps[train_split:val_split]
    test_ids = unique_petlymps[val_split:]

    # Initialize the storage dictionary
    data = {'training': [], 'validation': [], 'testing': []}


    # Now iterate through the DataFrame
    for index, row in df_2d.iterrows():
        patient_id = row['PatientID']
        petlymph = row["Petlymph"]
        label_name = row["Label_Name"]

        pet_path = image_base + str(petlymph)  + "_suv_cropped.nii.gz"
        ct_path = image_base + str(petlymph)  + "_ct_cropped.nii.gz"
        label_path = label_base + str(label_name) + ".nii.gz"


        if os.path.exists(pet_path) and os.path.exists(ct_path) and os.path.exists(label_path):
            # Decide the data split based on PetlympID
            entry = {
                "image": pet_path,
                "image2": ct_path,
                "label": label_path,
                "report": row["sentence"],
                "slice_num": row["Slice"],
                "suv_num": row["SUV"],
                "label_name": label_name
            }
            if patient_id in train_ids:
                # Optionally randomize folds within training data
                entry['fold'] = random.randint(1, 5)
                data['training'].append(entry)
            elif patient_id in val_ids:
                # makes the validation set fold 0 which is the validation fold in current training
                entry['fold'] = 0
                data['training'].append(entry)
            else:
                data['testing'].append(entry)
        else:
            dropped_missing_files += 1
            continue

    print(f"number of dropped labels: {dropped_missing_files}")
    # Write the JSON data to a file
    with open('/UserData/Zach_Analysis/uw_lymphoma_pet_3d/final_training_testing_v6.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)  # Use indent=4 for pretty printing