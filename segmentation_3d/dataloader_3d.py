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

        #keys = ['pet', 'ct', 'label']

        #pet_img = torch.from_numpy(pet_img)  # Convert to tensor
        #ct_img = torch.from_numpy(ct_img)  # Convert to tensor
        #label = torch.from_numpy(label)

        # Chain expand_dims with from_numpy to avoid intermediate variables
        pet_img = torch.from_numpy(np.expand_dims(pet_img, axis=0))
        ct_img = torch.from_numpy(np.expand_dims(ct_img, axis=0))
        label = torch.from_numpy(np.expand_dims(label, axis=0))

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