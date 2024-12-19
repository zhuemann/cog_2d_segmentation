import os
import shutil
import pandas as pd

def copy_images_and_labels_to_folder_v1():

    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Meghan_worksheet_matched.xlsx")
    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Steve_worksheet_matched.xlsx")
    df = pd.read_excel("/UserData/Zach_Analysis/final_testset_evaluation_vg/all_labels_jdw")
    # Define the base folder for the new location
    new_location_folder_base = "/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/"

    images_folder = "images"
    label_folder = "labels"
    new_image_folder = os.path.join(new_location_folder_base, images_folder)
    new_label_folder = os.path.join(new_location_folder_base, label_folder)
    original_location_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/"

    # Iterate through the dataframe and copy files with new names
    for index, row in df.iterrows():

        petlymph = row["Petlymph"]
        #original_path = row["File_Name"]
        new_name = row["Label_Name"]
        label_name = new_name + ".nii.gz"
        destination_path = os.path.join(new_location_folder_image, new_name)

        original_path = os.path.join(original_location_base, original_path)
        try:
            # Copy the file (not move)
            shutil.copy2(original_path, destination_path)
            print(f"Copied {original_path} to {destination_path}")
        except FileNotFoundError:
            print(f"File not found: {original_path}")
        except Exception as e:
            print(f"Error copying {original_path} to {destination_path}: {e}")


def copy_images_and_labels_to_folder():
    # Load the DataFrame
    df = pd.read_excel("/UserData/Zach_Analysis/final_testset_evaluation_vg/all_labels_jdw.xlsx")

    # Define the base folder for the new location
    new_location_folder_base = "/mnt/Bradshaw/UW_PET_Data/physican_labels/final_internal_dataset/"
    images_folder = "images"
    label_folder = "labels"
    new_image_folder = os.path.join(new_location_folder_base, images_folder)
    new_label_folder = os.path.join(new_location_folder_base, label_folder)

    # Original folders to search for files
    source_image_folder = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
    source_label_folder = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels6/"

    # Create destination folders if they don't exist
    os.makedirs(new_image_folder, exist_ok=True)
    os.makedirs(new_label_folder, exist_ok=True)

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        petlymph = str(row["Petlymph"])
        label_name = row["Label_Name"] + ".nii.gz"

        # Copy images that contain the Petlymph ID in their filename
        for file in os.listdir(source_image_folder):
            if petlymph in file:
                src_path = os.path.join(source_image_folder, file)
                dest_path = os.path.join(new_image_folder, file)

                # Copy only if the file doesn't already exist
                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(src_path, dest_path)
                        print(f"Copied {src_path} to {dest_path}")
                    except Exception as e:
                        print(f"Error copying {src_path} to {dest_path}: {e}")

        # Copy labels that match the Label_Name in the DataFrame
        # First, look in the main label folder
        label_src_path = os.path.join(source_label_folder, label_name)
        label_dest_path = os.path.join(new_label_folder, label_name)

        if os.path.exists(label_src_path) and not os.path.exists(label_dest_path):
            try:
                shutil.copy2(label_src_path, label_dest_path)
                print(f"Copied {label_src_path} to {label_dest_path}")
            except Exception as e:
                print(f"Error copying {label_src_path} to {label_dest_path}: {e}")