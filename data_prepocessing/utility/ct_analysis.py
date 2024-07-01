
import os
import numpy as np
import nibabel as nib


def ct_analysis():


    image_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images5/"

    ct_list = os.listdir(image_base)
    i = 0
    total_below = 0
    for ct in ct_list:
        print(f"index: {i}")
        i += 1
        ct_path = os.path.join(image_base, ct)

        ct_nii = nib.load(ct_path)
        ct = ct_nii.get_fdata()

        max_ct = np.max(ct)
        print(f"index: {i} max of: {max_ct} total number below: {total_below}")
        if max_ct < 200:
            total_below += 1
            print(ct)


    print(f"total cts that are below thereshold: {total_below}")