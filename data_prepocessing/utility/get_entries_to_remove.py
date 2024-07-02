
import os

def get_entries_to_remove():

    image_path_base5 = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images5/"
    label_path_base5 = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels5/"

    image_path_base4 = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4/"
    label_path_base4 = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels4/"

    # Get a list of all label file names in labels5
    labels5 = set(os.listdir(label_path_base5))

    # Get a list of all label file names in labels4
    labels4 = set(os.listdir(label_path_base4))

    # Find entries that are in labels4 but not in labels5
    missing_labels = labels4.difference(labels5)

    return list(missing_labels)