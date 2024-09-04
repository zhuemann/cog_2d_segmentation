
import cc3d
import numpy as np
from rt_utils import RTStructBuilder
import nibabel as nib
#from scipy import ndimage, morphology
import pydicom
import scipy

# sample different color codes
def sample_color_codes():
    color_codes = []
    for i in range(0, 256, 10):
        for j in range(0, 256, 10):
            for k in range(0, 256, 10):
                color_codes.append([i ,j ,k])
    return color_codes


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18 # 18 or 26
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def generate_RTs(dicom_series_path, pet_nifti_path, nifti_label_path, save_rtstruct_path):


    close_iterations = 1

    color_codes = sample_color_codes()
    # set random generator seed
    np.random.seed(1)
    # shuffle color codes
    np.random.shuffle(color_codes)

    # load nifti label
    nifti_label = nib.load(nifti_label_path).get_fdata()
    # permute the axes to match the dicom series
    nifti_label = np.transpose(nifti_label, (1 ,0 ,2))
    # get connected components
    nifti_label_comp = con_comp(nifti_label)

    # reassgin the study instance UID
    rtstruct_new = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)
    count = 0
    for i in range(1, nifti_label_comp.max( ) +1):
        mask_3d = nifti_label_comp == i
        mask_3d = scipy.ndimage.binary_closing(mask_3d, iterations=close_iterations) # 3D closing

        # ''' dilate the mask by 1 voxel and reapply the thresholding, sometimes, part of the mask include heart
        mask_3d = scipy.morphology.binary_dilation(mask_3d, scipy.morphology.ball(radius=1))
        # morphology.ball(radius=1) (all 26 neighbors)
        # default: cross-shaped structuring element, only the 6 neighboring voxels are considered

        # '''

        if np.sum(mask_3d) < 1:
            continue
        roi_new_name = f'ROI-{count +1}'
        rtstruct_new.add_roi(
            mask=mask_3d,
            color=color_codes[count],
            name=roi_new_name,
            use_pin_hole=True
        )
        count += 1

    series_description = "test"
    rtstruct_new.save(save_rtstruct_path)
    ds = pydicom.dcmread(save_rtstruct_path)
    ds.SeriesDescription = 'AI_contour' + '_' + series_description

    with open(save_rtstruct_path, 'wb') as outfile:
        ds.save_as(outfile)


def make_all_rts():


    dicom_series_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/PETWB_008427_01/20190821/PT/PET_CT_SKULL_BASE_TO_THIGH"
    pet_nifti_path = "/mnt/Bradshaw/UW_PET_Data/SUV_images/PETWB_008427_01/PETWB_008427_01_20190821_PT_WB_MAC_SUV.nii.gz"

    nifti_label_path = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v4_nifti/PETWB_008427_01_labelel_1.nii.gz"
    save_rtstruct_path = "/UserData/Zach_Analysis/test_folder/test_rt_struct"
    generate_RTs(dicom_series_path, pet_nifti_path, nifti_label_path, save_rtstruct_path)

