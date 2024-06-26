import random
import json
import pandas as pd

# Function to extract the image ID from the path
def extract_image_id(path):
    # Extract the part of the filename before '_suv_cropped.nii.gz'
    return path.split('/')[-1].split('_suv_cropped')[0]

def select_250_images_from_json():

    test_excel_file = "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/removed_wrong_suv_max_and_slices_13.xlsx"
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

    # Create DataFrame from selected entries
    json_df = pd.DataFrame(selected_entries)

    # Load data from Excel
    excel_df = pd.read_excel(test_excel_file)

    # Filter Excel DataFrame to only include rows where 'label_name' matches any 'label_name' in json_df
    label_names = json_df['label_name'].unique()
    matching_excel_entries = excel_df[excel_df['Label_Name'].isin(label_names)]

    # Combine the JSON DataFrame with the matching Excel DataFrame
    final_df = pd.concat([json_df, matching_excel_entries], ignore_index=True)

    return json_df, final_df