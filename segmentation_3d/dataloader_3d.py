import os

import numpy as np
#import pydicom as pdcm
import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk import word_tokenize, sent_tokenize
import nltk
import random
import pandas as pd

#import matplotlib as plt
import nibabel as nib
import numpy as np

from utility import rle_decode_modified, rle_decode


def convert_to_two_channel(volume_3d):
    """
    Convert a single-channel 3D volume with labels 0 and 1 into a two-channel volume.
    The first channel represents the background (label 0), and the second channel represents the object (label 1).

    Parameters:
    volume_3d (np.ndarray): A 3D NumPy array of shape (D, H, W) with labels 0 and 1.

    Returns:
    np.ndarray: A 4D NumPy array of shape (2, D, H, W) where:
                - channel 0 is the background mask (label == 0)
                - channel 1 is the object mask (label == 1)
    """
    if not isinstance(volume_3d, np.ndarray):
        raise TypeError("Input volume must be a NumPy array.")
    if volume_3d.ndim != 3:
        raise ValueError("Input volume must be a 3D array.")
    if not np.array_equal(np.unique(volume_3d), [0, 1]) and not np.array_equal(np.unique(volume_3d),
                                                                               [0]) and not np.array_equal(
            np.unique(volume_3d), [1]):
        raise ValueError("Input volume must contain only 0 and 1 as labels.")

    # Create empty array for two channels
    channels = np.zeros((2, *volume_3d.shape), dtype=volume_3d.dtype)

    # Channel 0: Background (label == 0)
    channels[0] = (volume_3d == 0).astype(volume_3d.dtype)

    # Channel 1: Object (label == 1)
    channels[1] = (volume_3d == 1).astype(volume_3d.dtype)

    return channels


class TextImageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, truncation=True,
                 dir_base='/home/zmh001/r-fcb-isilon/research/Bradshaw/', mode=None, transforms=None, resize=None,
                 img_size=256,
                 wordDict = None,
                 ngram_synonom = [], norm = None):  # data_path = os.path.join(dir_base,'Lymphoma_UW_Retrospective/Data/mips/')
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.report
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = max_len
        self.img_size = img_size
        self.wordDict = wordDict

        #self.df_data = dataframe.values
        self.transforms = transforms
        self.mode = mode
        self.data_path = os.path.join(dir_base, "Zach_Analysis/cog_data_splits/mips/")
        self.dir_base = dir_base
        self.resize = resize
        self.norm = norm

    def __len__(self):
        return len(self.text)

    # Load the .nii.gz file
    def load_nii_to_numpy(self, file_path):
        # Load the NIfTI file using nibabel
        img = nib.load(file_path)

        # Get the image data as a NumPy array
        img_data = img.get_fdata()

        # Optionally, you can cast the data to a specific type if needed (e.g., np.float32)
        img_array = np.array(img_data, dtype=np.float32)

        return img_array

    def __getitem__(self, index):
        # text extraction
        #global img, image

        text = str(self.text[index])
        text = " ".join(text.split())
        #print(text)

        text = text.replace("\n", "")

        #if self.wordDict != None:
        #    text = TextImageDataset.synonymsReplacement(self, text)
        #    text = TextImageDataset.shuffledTextAugmentation(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            #pad_to_max_length=True,
            padding= 'max_length',   #True,  # #TOD self.max_len,
            # padding='longest',
            truncation='longest_first',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # images data extraction
        img_name = self.row_ids[index]
        img_name = str(img_name)  # + "_mip.png"
        img_path = self.data.image[index]
        #print(img_name)

        #pet_img = self.load_nii_to_numpy(img_path).astype(np.float32)
        #ct_img = self.load_nii_to_numpy(self.data.image2[index]).astype(np.float32)
        #label = self.load_nii_to_numpy(self.data.label[index]).astype(np.float32)

        pet_img = self.load_nii_to_numpy(img_path)
        ct_img = self.load_nii_to_numpy(self.data.image2[index])
        label = self.load_nii_to_numpy(self.data.label[index])
        label = convert_to_two_channel(label)
        #keys = ['pet', 'ct', 'label']
        print(f"label after conversion dimensions: {label.shape}")
        #pet_img = torch.from_numpy(pet_img)  # Convert to tensor
        #ct_img = torch.from_numpy(ct_img)  # Convert to tensor
        #label = torch.from_numpy(label)

        # Chain expand_dims with from_numpy to avoid intermediate variables
        pet_img = torch.from_numpy(np.expand_dims(pet_img, axis=0))
        ct_img = torch.from_numpy(np.expand_dims(ct_img, axis=0))
        label = torch.from_numpy(np.expand_dims(label, axis=0))
        print(f"label returned dimensions: {label.size()}")

        data_dic = {
            'pet': pet_img,
            'ct': ct_img,
            'label': label
        }


        transformed_data = self.resize(data_dic)
        del data_dic, pet_img, ct_img, label
        #save_path = "/UserData/Zach_Analysis/test_folder/saved_augmented_data/original.nii.gz"

        # Define an affine transformation matrix (identity matrix by default)
        #affine = np.eye(4)
        # Create a NIfTI image from the NumPy array
        #nii_img = nib.Nifti1Image(img, affine)

        # Save the NIfTI image as a .nii.gz file
        #nib.save(nii_img, save_path)

        # Create a NIfTI image from the NumPy array
        #nii_img = nib.Nifti1Image(transformed.detach().cpu().numpy(), affine)
        #nib.save(nii_img, "/UserData/Zach_Analysis/test_folder/saved_augmented_data/augmented.nii.gz")


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float),
            #'targets': segmentation_mask,
            'row_ids': self.row_ids[index],
            'images': transformed_data
        }