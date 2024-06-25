import random
import json
import pandas as pd

# Function to extract the image ID from the path
def extract_image_id(path):
    # Extract the part of the filename before '_suv_cropped.nii.gz'
    return path.split('/')[-1].split('_suv_cropped')[0]

def select_250_images_from_json():

    input_file = "/UserData/Zach_Analysis/uw_lymphoma_pet_3d/output_resampled_13000_no_validation.json"
    # Read the existing data from the JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)
    # Assuming 'data' is your JSON data loaded into a dictionary
    testing_data = data["testing"]

    # Group data by image ID extracted from the 'image' path
    image_groups = {}
    for entry in testing_data:
        image_id = extract_image_id(entry['image'])
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(entry)

    # Randomly select groups until we have at least 250 images
    selected_entries = []
    image_ids = list(image_groups.keys())
    random.shuffle(image_ids)  # Shuffle image ID list for random selection

    for img_id in image_ids:
        selected_entries.extend(image_groups[img_id])
        if len(selected_entries) >= 250:
            break

    df = pd.DataFrame(selected_entries)

    return df