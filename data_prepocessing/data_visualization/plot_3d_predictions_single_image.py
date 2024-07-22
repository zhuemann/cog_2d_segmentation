import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from skimage import morphology, measure
from scipy import ndimage as ndi

def extend(label):
    # Squeeze the array to remove singleton dimensions
    label_squeezed = np.squeeze(label)
    # Apply binary dilation
    dilated = morphology.binary_dilation(
                label_squeezed, morphology.ball(radius=0))
    return dilated

def plot_3d_predictions_single_image(PET_file, label_file, prediction_file, save_file, sent):
    # Get PET from ref_folder
    ref_img_PT = nib.load(PET_file).get_fdata().astype(np.float32)
    ref_label = nib.load(label_file).get_fdata().astype(np.int8)
    ref_pred = nib.load(prediction_file).get_fdata().astype(np.int32)
    ref_pred = (ref_pred >= .5).astype(np.uint8)

    # gets the pet image ready
    PET_mip = np.max(ref_img_PT, axis=1)  # 0 or 1
    PET_mip = np.rot90(PET_mip)
    PET_mip[PET_mip < 0.0] = 0.0
    PET_mip[PET_mip > 6] = 6
    PET_mip = -PET_mip

    # labels
    ref_label = extend(ref_label)
    ref_label = np.max(ref_label, axis=1)
    ref_label = np.rot90(ref_label)

    # prediction preprocessing
    ref_pred = np.squeeze(ref_pred)
    ref_pred = extend(ref_pred)
    ref_pred = np.max(ref_pred, axis=1)
    ref_pred = np.rot90(ref_pred)

    # Label connected components in the prediction and label images
    labeled_pred, num_pred = ndi.label(ref_pred)
    labeled_label, num_label = ndi.label(ref_label)

    # Create a figure
    plt.figure(figsize=(10, 10))
    plt.imshow(PET_mip, cmap='gray')

    # Function to plot contours with specific colors
    def plot_contours(mask, color):
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

    # Loop over each predicted contour
    for pred_id in range(1, num_pred + 1):
        pred_mask = labeled_pred == pred_id
        overlap = False
        for label_id in range(1, num_label + 1):
            label_mask = labeled_label == label_id
            if np.any(np.logical_and(pred_mask, label_mask)):
                overlap = True
                break
        if overlap:
            plot_contours(pred_mask, 'green')  # Overlap
        else:
            plot_contours(pred_mask, 'red')  # False positive

    # Loop over each label contour to find false negatives
    for label_id in range(1, num_label + 1):
        label_mask = labeled_label == label_id
        if not np.any(np.logical_and(label_mask, ref_pred)):
            plot_contours(label_mask, 'blue')  # False negative

    plt.title(f'{sent}')
    plt.axis('off')
    plt.savefig(save_file)
    #plt.show()

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
import os
def plot_all_images():

    prediction_folder = "/UserData/Zach_Analysis/git_multimodal/3DVision_Language_Segmentation_inference/COG_dynunet_baseline/COG_dynunet_0_baseline/dynunet_0_0/predictions_v5_f1_.65_v2/"
    all_predictions = os.listdir(prediction_folder)

    labels = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels5/"

    pet_images = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images5/"

    save_location = "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/single_plot_predictions/"

    df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/removed_wrong_suv_max_and_slices_13.xlsx")

    for pred in all_predictions:

        label_name = pred.strip(".nii")

        row = df[df['Label_Name'] == label_name]

        sentence = row["sentence"].iloc[0]
        print(sentence)
        sentence = insert_newlines(sentence, word_limit=10)

        image_name = pred[:15]
        PET_file = pet_images + image_name + "_suv_cropped.nii.gz"
        label_file = labels + pred + ".gz"
        prediction_file = prediction_folder + pred
        save_file = save_location + pred[:-4] + ".png"

        #print(f"pet file: {PET_file}")
        #print(f"label_file: {label_file}")
        #print(f"fpred file: {prediction_file}")
        #print(f"save location: {save_file}")
        plot_3d_predictions_single_image(PET_file, label_file, prediction_file, save_file, sentence)


