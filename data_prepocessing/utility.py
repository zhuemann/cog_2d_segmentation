import os
import pandas as pd

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
