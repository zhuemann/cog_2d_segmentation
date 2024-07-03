
import os
import numpy as np
import nibabel as nib


def ct_analysis():


    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4/"

    ct_list = os.listdir(image_base)
    i = 0
    total_below = 0
    total_above = 0
    ct_to_high = []
    for ct_name in ct_list:
        #print(ct_name)
        if "suv" in ct_name:
            continue

        i += 1
        ct_path = os.path.join(image_base, ct_name)

        ct_nii = nib.load(ct_path)
        ct = ct_nii.get_fdata()

        max_ct = np.max(ct)
        print(f"index: {i} max of: {max_ct} total number below: {total_below} total above: {total_above}")
        if max_ct < 500:
            total_below += 1
            print(ct_name)
        if max_ct > 10000:
            total_above += 1
            print(ct_name)
            ct_to_high.append(ct_name)


    print(f"total cts that are below thereshold: {total_below}")
    print(ct_to_high)


def ct_analysis_1():

    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images5/"

    ct_list = os.listdir(image_base)


