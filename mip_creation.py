import os
import glob
import nibabel as nib
import numpy as np
import imageio


def load_nii(filepath):
    # Load a .nii.gz file and return the 3D numpy array
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return data


def save_image(projection, out_path):
    # Normalize and save the projection as PNG for display
    # Convert to uint8 if necessary
    proj_min, proj_max = projection.min(), projection.max()
    if proj_max > proj_min:
        proj_norm = (projection - proj_min) / (proj_max - proj_min)
    else:
        # Handle flat images (all same intensity)
        proj_norm = projection - proj_min

    proj_uint8 = (proj_norm * 255).astype(np.uint8)
    imageio.imwrite(out_path, proj_uint8)

def mip_creation():

    image_dir = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
    label_dir = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels6/"

    output_sagital_images = "/UserData/Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_sagittal_images/"
    output_coronal_images = "/UserData/Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_coronal_images/"
    output_sagital_labels = "/UserData/Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_sagittal_labels/"
    output_coronal_labels = "/UserData/Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_coronal_labels/"

    # Create output directories if they don't exist
    os.makedirs(output_sagital_images, exist_ok=True)
    os.makedirs(output_coronal_images, exist_ok=True)
    os.makedirs(output_sagital_labels, exist_ok=True)
    os.makedirs(output_coronal_labels, exist_ok=True)

    # Get all image files
    image_files = glob.glob(os.path.join(image_dir, "*.nii.gz"))
    i = 0
    for img_path in image_files:
        fname = os.path.basename(img_path)
        fname_lower = fname.lower()

        # Skip images that contain "ct"
        if "ct" in fname_lower:
            continue

        # Only process images that contain "SUV"
        if "suv" not in fname_lower:
            continue
        i += 1
        print(f"image index: {i} out of {len(image_files)/2}")
        # Load the image volume
        img_data = load_nii(img_path)

        # Make sagittal and coronal projections
        sagittal_proj = np.max(img_data, axis=0)  # shape (Y, Z)
        coronal_proj = np.max(img_data, axis=1)  # shape (X, Z)

        # Save the image projections
        sagittal_img_path = os.path.join(output_sagital_images, fname.replace(".nii.gz", "_sag.png"))
        coronal_img_path = os.path.join(output_coronal_images, fname.replace(".nii.gz", "_cor.png"))
        save_image(sagittal_proj, sagittal_img_path)
        save_image(coronal_proj, coronal_img_path)

        # Attempt to find the corresponding label file
        # This depends on your naming scheme for labels.
        # For simplicity, let's assume labels have the same filename in label_dir
        # If they differ, you may need to adjust this logic.
        """
        label_path = os.path.join(label_dir, fname)
        if os.path.exists(label_path):
            label_data = load_nii(label_path)
            sagittal_label_proj = np.max(label_data, axis=0)
            coronal_label_proj = np.max(label_data, axis=1)

            sagittal_label_path = os.path.join(output_sagital_labels, fname.replace(".nii.gz", "_sag_label.png"))
            coronal_label_path = os.path.join(output_coronal_labels, fname.replace(".nii.gz", "_cor_label.png"))
            save_image(sagittal_label_proj, sagittal_label_path)
            save_image(coronal_label_proj, coronal_label_path)
        else:
            print(f"Warning: No corresponding label file found for {fname}.")
        """
    label_files = glob.glob(os.path.join(label_dir, "*.nii.gz"))
    i = 0
    for label_path in label_files:
        i += 1
        print(f"label index: {i} out of {len(label_files)}")
        label_fname = os.path.basename(label_path)

        # Load label volume
        label_data = load_nii(label_path)

        # Make sagittal and coronal projections
        sagittal_label_proj = np.max(label_data, axis=0)
        coronal_label_proj = np.max(label_data, axis=1)

        # Save label projections
        sagittal_label_path = os.path.join(output_sagital_labels, label_fname.replace(".nii.gz", "_sag.png"))
        coronal_label_path = os.path.join(output_coronal_labels, label_fname.replace(".nii.gz", "_cor.png"))
        save_image(sagittal_label_proj, sagittal_label_path)
        save_image(coronal_label_proj, coronal_label_path)

    print("Finished processing labels.")
