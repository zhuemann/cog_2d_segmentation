import os
import pandas as pd
import nibabel as nib

def get_suv_file_names(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

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
    df.to_excel("full_suv_names.xlsx", index=False)


def create_prediction_plots():

    print("hi")


def average_components(tuples_list):
    sum_i, sum_j, sum_k = 0, 0, 0
    count = len(tuples_list)

    # Summing up all i, j, k components
    for i, j, k in tuples_list:
        sum_i += i
        sum_j += j
        sum_k += k

    # Calculating average for each component
    if count > 0:
        avg_i = sum_i / count
        avg_j = sum_j / count
        avg_k = sum_k / count
    else:
        avg_i = avg_j = avg_k = 0  # handle the case where the list is empty

    # Print the averages
    print(f"Average of i: {avg_i}")
    print(f"Average of j: {avg_j}")
    print(f"Average of k: {avg_k}")
def count_left_right_sided(df):

    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v2_nifti/"

    left_indices = []
    right_indices = []
    for index, row in df.iterrows():

        sentence = row["sentence"]

        label_name = row["Label_Name"]
        label_path = os.path.join(label_path_base, label_name)
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()
        coordinates = (row["i"], row["j"], row["k"])

        if 'left' in sentence and 'right' not in sentence:

            left_indices.append(coordinates)
        elif 'left' not in sentence and 'right' in sentence:
            right_indices.appen(coordinates)

    print("left components")
    average_components(left_indices)
    print("right comonents")
    average_components(right_indices)

