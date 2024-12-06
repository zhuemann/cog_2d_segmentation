
import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import pydicom


def get_slice_thickness(folder_name):
    #image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    image_path_base = "/mnt/Bradshaw/UW_PET_Data/external_testset_v2/"
    image_path_base = "/mnt/Bradshaw/UW_PET_Data/external_testset_try2/"

    image_path = os.path.join(image_path_base, folder_name)
    file_names = os.listdir(image_path)
    index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
    image_path = os.path.join(image_path, file_names[index_of_suv[0]])
    nii_image = nib.load(image_path)
    header = nii_image.header
    voxel_dims = header.get_zooms()
    return voxel_dims
def extract_image_id(path):
    # Extract the part of the filename before '_suv_cropped.nii.gz'
    return path.split('/')[-1].split('_suv_cropped')[0]

def insert_newlines(text, word_limit=15):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        if len(current_line) == word_limit:
            lines.append(' '.join(current_line))
            current_line = []

    # Add the last line if there are any remaining words
    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)

def normalize_mip(mip):
    # Step 2: Clip values above 11 to 11
    clipped = np.clip(mip, None, 11)

    # Step 3: Normalize the image so that 11 maps to 255
    # First, ensure no division by zero errors by setting the max value to at least 1
    max_value = np.max(clipped.max())
    normalized = (clipped / max_value) * 255

    # Convert to uint8 type for image representation
    normalized_uint8 = normalized.astype(np.uint8)
    return normalized_uint8

def get_corresponding_pet_slice(ct_slice_idx, ct_voxel_size, pet_voxel_size):
    """
    Calculates the corresponding PET slice for a given CT slice based on their voxel sizes.

    Args:
        ct_slice_idx (int): The index of the CT slice (0-based index).
        ct_voxel_size (tuple): The voxel size of the CT volume (z, y, x).
        pet_voxel_size (tuple): The voxel size of the PET volume (z, y, x).

    Returns:
        float: The corresponding PET slice index (can be a non-integer if interpolation is needed).
    """
    # Extract the slice thickness (z-axis voxel size)
    ct_slice_thickness = ct_voxel_size[2]
    pet_slice_thickness = pet_voxel_size[2]

    # Calculate the position of the CT slice in physical space (z-axis position)
    ct_slice_position = ct_slice_idx * ct_slice_thickness

    # Calculate the corresponding PET slice index
    pet_slice_idx = ct_slice_position / pet_slice_thickness

    return pet_slice_idx

def plot_for_orientation_and_modality():
    # Paths and DataFrame loading (adjust as necessary)
    image_path_root = "/mnt/Bradshaw/UW_PET_Data/external_testset_v2/"
    image_path_root = "/mnt/Bradshaw/UW_PET_Data/external_testset_try2/"

    label_path_base = "/mnt/Bradshaw/UW_PET_Data/external_raw_pet/testv4/"
    df = pd.read_excel(
        "/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/swedish_dataframe_pet_ct_labeled.xlsx")

    df = pd.read_excel(
        "/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/additional_sentences_labeled.xlsx")
    # Add the path to your fused PET/CT DICOM file
    #fused_dicom_path = '/path/to/your/fused_PET_CT.dcm'  # Update this path

    for index, row in df.iterrows():
        print(f"Processing index: {index}")
        #print(row)
        # Extract the folder name and get voxel dimensions
        petlymph = row["ID"]
        if float(row["Keep"]) != float(1):
            print(str(row["Keep"]))
            print("skipping cuz noted skip")
            continue
        """
        try:
            pet_voxel_dims = get_slice_thickness(petlymph)
        except:
            print("can't load this")
            continue
        print(f"PET voxel dimensions: {pet_voxel_dims}")
        """

        # Build paths to PET and CT images
        image_path_base = os.path.join(image_path_root, petlymph)
        file_names = os.listdir(image_path_base)
        index_of_suv = [idx for idx, element in enumerate(file_names) if "suv" in element.lower()]
        image_path = os.path.join(image_path_base, file_names[index_of_suv[0]])

        #index_of_ct = [idx for idx, element in enumerate(file_names) if "ct" in element.lower()]
        # Filter file names to find the desired CT image
        index_of_ct = [idx for idx, element in enumerate(file_names) if
                       "ct" in element.lower() and "pt" not in element.lower()]
        print(f"index of ct: {index_of_ct} image name: {petlymph}")

        if len(index_of_ct) == 0:
            print("missing ct")
            continue
        ct_image_path = os.path.join(image_path_base, file_names[index_of_ct[0]])
        print(f"Image name: {petlymph}")

        # Load PET and CT images
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()
        ct_image = nib.load(ct_image_path)
        ct_volume = ct_image.get_fdata()
        ct_voxel_dims = ct_image.header.get_zooms()
        print(f"CT voxel dimensions: {ct_voxel_dims}")

        # Load the fused PET/CT DICOM file to get the slice thickness
        # fused_dicom = pydicom.dcmread(fused_dicom_path)
        # fused_slice_thickness = float(fused_dicom.SliceThickness)
        # print(f"Fused PET/CT slice thickness: {fused_slice_thickness}")

        # Flip data if necessary (adjust axes as per your data)
        img = np.flip(img, axis=0)
        ct_volume = np.flip(ct_volume, axis=0)

        # Compute MIP images for PET and CT
        mip_coronal_pet = np.max(img, axis=1)
        mip_coronal_ct = np.max(ct_volume, axis=1)

        # Normalize MIP images if needed
        #mip_coronal_pet = normalize_mip(mip_coronal_pet)
        #mip_coronal_ct = normalize_mip(mip_coronal_ct)

        # Rotate and flip images for correct orientation
        mip_coronal_pet = np.rot90(mip_coronal_pet)
        mip_coronal_ct = np.rot90(mip_coronal_ct)
        mip_coronal_pet = np.fliplr(mip_coronal_pet)
        mip_coronal_ct = np.fliplr(mip_coronal_ct)

        # Calculate the positions of the lines
        # Assuming 'Slice' column contains the PET slice number
        pet_slice_num = row["Image"]  # Replace with the correct column name if different
        #ct_slice_num = row["Image"]  # Replace with the correct column name for CT slice number
        #ct_slice_num = get_corresponding_pet_slice(pet_slice_num, ct_voxel_dims, pet_voxel_dims)
        ct_slice_num = row["Image"]
        # Calculate positions in the MIP images
        pet_line_position = pet_slice_num #* pet_voxel_dims[2] / pet_voxel_dims[1]  # Adjust for pixel spacing
        ct_line_position = ct_slice_num #ct_slice_num * ct_voxel_dims[2] / ct_voxel_dims[1]
        print(f"pet slice: {pet_slice_num} ct slice: {ct_slice_num}")
        # For fused PET/CT, calculate line position using fused slice thickness
        # fused_slice_num = pet_slice_num  # Adjust if the fused image has a different slice number
        # fused_line_position = fused_slice_num * fused_slice_thickness / pet_voxel_dims[1]  # Adjust for pixel spacing

        # Calculate positions when indexing from the bottom
        pet_total_slices = mip_coronal_pet.shape[0]  # Total number of slices in PET MIP
        ct_total_slices = mip_coronal_ct.shape[0]  # Total number of slices in CT MIP

        # Positions when indexing from the bottom
        pet_line_position_bottom = pet_total_slices - pet_slice_num
        ct_line_position_bottom = ct_total_slices - ct_slice_num

        # convert ct lines to pet for plotting, the lines should line up
        ct_line_position_for_pet = get_corresponding_pet_slice(ct_line_position, ct_voxel_dims, pet_voxel_dims)
        ct_line_position_for_pet_bottom = get_corresponding_pet_slice(ct_line_position_bottom, ct_voxel_dims, pet_voxel_dims)
        #conver pet lines to CT for plotting, the lines should line up
        pet_line_position_for_ct = get_corresponding_pet_slice(pet_line_position, pet_voxel_dims, ct_voxel_dims)
        pet_line_position_for_ct_bottom = get_corresponding_pet_slice(pet_line_position_bottom, pet_voxel_dims, ct_voxel_dims)

        # Plotting
        plt.figure(figsize=(20, 10))

        # Plot PET MIP
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(mip_coronal_pet, cmap='gray', aspect='auto', vmin=0, vmax=10)
        ax1.set_title('PET MIP (Coronal)')

        # Plot lines on PET MIP
        ax1.axhline(y=pet_line_position, color='r', linestyle='-', label='PET Slice')
        ax1.axhline(y=pet_line_position_bottom, color='orange', linestyle='-', label='PET Slice (Bottom Indexing)')
        ax1.axhline(y= ct_line_position_for_pet, color='g', linestyle='--', label='CT Slice')
        ax1.axhline(y=ct_line_position_for_pet_bottom, color='lime', linestyle='--', label='CT Slice (Bottom Indexing)')

        # ax1.axhline(y=fused_line_position, color='b', linestyle=':', label='Fused PET/CT Slice')
        ax1.legend()

        # Plot CT MIP
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(mip_coronal_ct, cmap='gray', aspect='auto', vmax=600, vmin=-300)
        ax2.set_title('CT MIP (Coronal)')

        # Plot lines on CT MIP
        ax2.axhline(y=pet_line_position_for_ct, color='r', linestyle='-', label='PET Slice')
        ax2.axhline(y=pet_line_position_for_ct_bottom, color='orange', linestyle='-', label='PET Slice (Bottom Indexing)')
        ax2.axhline(y=ct_line_position, color='g', linestyle='--', label='CT Slice')
        ax2.axhline(y=ct_line_position_bottom, color='lime', linestyle='--', label='CT Slice (Bottom Indexing)')
        # ax2.axhline(y=fused_line_position, color='b', linestyle=':', label='Fused PET/CT Slice')
        ax2.legend()
        print(row)
        sentence = row["Sentence"]
        # print(sentence)
        # print(type(sentence))
        sentence = insert_newlines(sentence, word_limit=25)
        plt.suptitle(sentence + " Image number used: " + str(row['Image']) + " SUV: " + str(
            row["SUV"]), fontsize=12, color='black')

        # Save the figure
        label_name = row["Label_Name"]
        plt.savefig(f"/UserData/Zach_Analysis/final_testset_evaluation_vg/orientation_line_plots_2/{row['ID']}_" + f"{label_name}_MIP.png",
                    dpi=300)
        plt.close()
        print(f"Saved figure for {label_name}")