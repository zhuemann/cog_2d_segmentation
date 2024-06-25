import random
import json
import pandas as pd

def select_250_images_from_json():

    input_file = "/UserData/Zach_Analysis/uw_lymphoma_pet_3d/output_resampled_13000_no_validation.json"
    # Read the existing data from the JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)
    # Assuming 'data' is your JSON data loaded into a dictionary
    testing_data = data["testing"]

    # Group data by 'Petlymph'
    patient_groups = {}
    for entry in testing_data:
        patient_id = entry['Petlymph']
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(entry)

    # Randomly select groups until we have at least 250 images
    selected_entries = []
    patient_ids = list(patient_groups.keys())
    random.shuffle(patient_ids)  # Shuffle patient ID list for random selection

    for pid in patient_ids:
        selected_entries.extend(patient_groups[pid])
        if len(selected_entries) >= 250:
            break

    df = pd.DataFrame(selected_entries)

    return df