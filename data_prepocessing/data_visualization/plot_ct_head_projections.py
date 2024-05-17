import os



def plot_ct_head_projections():

    base_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    folder_list = os.listdir(base_folder)

    for folder in folder_list:

        current_path = os.path.join(base_folder, folder)

        for file_name in os.listdir(base_folder):
            if "CT" in file_name and os.path.isfile(os.path.join(base_folder, file_name)):
                current_path = os.path.join(current_path, file_name)

        print(current_path)
