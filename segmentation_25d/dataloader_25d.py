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
import re

#import matplotlib as plt

from utility import rle_decode_modified, rle_decode


class TextImageDataset_v1(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, truncation=True,
                 dir_base='/home/zmh001/r-fcb-isilon/research/Bradshaw/', mode=None, transforms=None, resize=None,
                 img_size=256,
                 wordDict = None,
                 ngram_synonom = [], norm = None):  # data_path = os.path.join(dir_base,'Lymphoma_UW_Retrospective/Data/mips/')
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.report
        self.targets = self.data.label_name
        self.row_ids = self.data.index
        self.slice_num = dataframe.slice_num
        self.suv = dataframe.suv_num
        self.max_len = max_len
        self.img_size = img_size
        self.wordDict = wordDict

        self.df_data = dataframe.values
        self.transforms = transforms
        self.mode = mode
        #self.data_path = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/images_coronal_mip_uw_v2/")
        # self.label_path = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/labels_coronal_mip/")
        #self.label_path = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/labels_coronal_mip_uw_v2/")
        self.data_path_coronal = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_coronal_images/")
        self.data_path_sagittal = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_sagittal_images/")
        self.label_path_coronal = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_coronal_labels/")
        self.label_path_sagittal = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_sagittal_labels/")

        #self.data_path = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/images_coronal_mip_v13/")
        #self.label_path = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/labels_coronal_mip/")
        #self.label_path = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/labels_coronal_mip_v13/")
        self.dir_base = dir_base
        self.resize = resize
        self.norm = norm

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        #print(f"indexing variable: {index}")
        # text extraction
        #global img, image
        #print(f"index: {index}")
        text = str(self.text[index])
        #print(f"text before: {text}")
        slice_num = self.slice_num[index]
        suv = self.suv[index]
        #print(f"slice: {slice_num} suv: {suv}")
        text = text.replace(str(suv), "").replace(str(slice_num), "")
        #print(f"Text after: {text}")
        #text = re.sub(r'\d+', '', text) # remove numbers Need more sophistcated way of doing this level 3 is stripped
        #text = ""                        # remove all text
        text = " ".join(text.split())

        text = text.replace("[ALPHANUMERICID]", "")
        text = text.replace("[date]", "")
        text = text.replace("[DATE]", "")
        text = text.replace("[AGE]", "")


        text = text.replace("[ADDRESS]", "")
        text = text.replace("[PERSONALNAME]", "")
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

        label_name = self.data.label_name[index]
        #print(f"label name: {label_name}")
        img_name =  "_".join(label_name.split("_")[:3])
        #print(f"image name: {img_name}")

        img_path = os.path.join(self.data_path_sagittal, img_name) + "_suv_cropped_sag.png"
        with Image.open(img_path) as img:
            img_raw = np.array(img)
        #DCM_Img = pdcm.read_file(img_path)
        #test = plt.imshow(DCM_Img.pixel_array, cmap=plt.cm.bone)
        #plt.show()

        try:
            #DCM_Img = pdcm.read_file(img_path)
            #img_raw = DCM_Img.pixel_array
            #img_raw[img_raw > 10] = 10
            #img_norm = img_raw * (255 / np.amax(img_raw))  # puts the highest value at 255
            #img = np.uint8(img_norm)
            #img_raw = img_norm
            img = img_raw

        except:
            print("can't open image")
            print(img_path)

        #print(self.targets[index])
        #print(f"target: {self.targets[index]}")
        label_name = str(self.targets[index]) + "_sag.png"
        label_path = os.path.join(self.label_path_sagittal, label_name)
        #print(label_path)
        with Image.open(label_path) as label_load:
            label = np.array(label_load)

        # decodes the rle
        if self.targets[index] != str(-1):
            #segmentation_mask_org = rle_decode(self.targets[index], (1024, 1024))
            #segmentation_mask_org = rle_decode_modified(self.targets[index], (1024, 1024))

            segmentation_mask_org = np.uint8(label)
        else:
            segmentation_mask_org = np.zeros((1024, 1024))
            segmentation_mask_org = np.uint8(segmentation_mask_org)
        # segmentation_mask_org = Image.fromarray(segmentation_mask_org).convert("RGB")
        # makes the segmentation mask into a PIL image
        # segmentation_mask = self.resize(segmentation_mask_org)
        # print(segmentation_mask.size())
        RGB = True
        if self.transforms is not None:
            # image = self.transforms(img)

            if self.mode == "train":
                # print(type(img))
                # print(img.shape)
                if RGB:
                    img = Image.fromarray(img).convert("RGB") # should make this more rigous but stich this guy
                else:
                    img = Image.fromarray(img)

                # print(type(img))
                # img = Image.fromarray(img)
                img = np.array(img)
                # segmentation_mask_org = np.uint8(segmentation_mask_org)
                # print(type(img))
                transformed = self.transforms(image=img, mask=segmentation_mask_org)
                image = transformed['image']
                segmentation_mask_org = transformed['mask']
                image = Image.fromarray(np.uint8(image))  # makes the image into a PIL image
                image = self.resize(image)  # resizes the image to be the same as the model size
                # segmentation_mask = Image.fromarray(np.uint8(segmentation_mask))
                # segmentation_mask = self.resize(segmentation_mask)


            else:

                if RGB:
                    img = Image.fromarray(img).convert("RGB")  # makes the image into a PIL image
                    #img = np.array(img)
                    #img = self.norm(image=img)
                    #img = img["image"]
                    #img = self.norm(img)
                    #img = Image.fromarray(np.uint8(img))
                    image = self.resize(img)
                    #image = self.transforms(img)

                    #trying to do flipping
                    #transformed = self.transforms(image=img, mask=segmentation_mask_org)
                    #image = transformed['image']
                    #segmentation_mask_org = transformed['mask']
                else:
                    img = Image.fromarray(img)
                    image = self.transforms(img)


        else:
            image = img

        # print(img.shape)
        # print(segmentation_mask.shape)
        #print(np.sum(segmentation_mask_org))
        segmentation_mask = Image.fromarray(np.uint8(segmentation_mask_org))
        segmentation_mask = self.resize(segmentation_mask)
        # for showing the images with maps and such
        # plt.figure()
        # DCM_Img = pdcm.read_file(img_path)
        # img_raw = DCM_Img.pixel_array
        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(img_raw, cmap=plt.cm.bone,)
        # ax[1].imshow(image.squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        # ax[2].imshow(segmentation_mask, cmap="jet", alpha = 1)
        # ax[2].imshow(image.squeeze().cpu().detach().numpy(), cmap=plt.cm.bone, alpha = .5)
        # plt.show()
        # print("returing from dataloader")
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'targets': segmentation_mask,
            'row_ids': self.row_ids[index],
            'images': image,
            'sentence': text,
            'Label_Name': label_name
        }

class TextImageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, truncation=True,
                 dir_base='/home/zmh001/r-fcb-isilon/research/Bradshaw/', mode=None, transforms=None, resize=None,
                 img_size=256,
                 wordDict=None,
                 ngram_synonom=[],
                 norm=None):  # data_path = os.path.join(dir_base,'Lymphoma_UW_Retrospective/Data/mips/')
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.report
        self.targets = self.data.label_name
        self.row_ids = self.data.index
        self.slice_num = dataframe.slice_num
        self.suv = dataframe.suv_num
        self.max_len = max_len
        self.img_size = img_size
        self.wordDict = wordDict

        self.df_data = dataframe.values
        self.transforms = transforms
        self.mode = mode
        self.data_path_coronal = os.path.join(dir_base,
                                              "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_coronal_images/")
        self.data_path_sagittal = os.path.join(dir_base,
                                               "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_sagittal_images/")
        self.label_path_coronal = os.path.join(dir_base,
                                               "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_coronal_labels/")
        self.label_path_sagittal = os.path.join(dir_base,
                                                "Zach_Analysis/petlymph_image_data/final_2.5d_images_and_labels/output_sagittal_labels/")


        self.dir_base = dir_base
        self.resize = resize
        self.norm = norm

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        slice_num = self.slice_num[index]
        suv = self.suv[index]
        text = text.replace(str(suv), "").replace(str(slice_num), "")
        text = " ".join(text.split())
        text = text.replace("[ALPHANUMERICID]", "").replace("[date]", "").replace("[DATE]", "").replace("[AGE]", "")
        text = text.replace("[ADDRESS]", "").replace("[PERSONALNAME]", "").replace("\n", "")

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation='longest_first',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        label_name = self.data.label_name[index]
        img_name = "_".join(label_name.split("_")[:3])

        # Load sagittal image
        img_path_sag = os.path.join(self.data_path_sagittal, img_name) + "_suv_cropped_sag.png"
        with Image.open(img_path_sag) as img_sag:
            img_sag_raw = np.array(img_sag)

        # Load coronal image
        img_path_cor = os.path.join(self.data_path_coronal, img_name) + "_suv_cropped_cor.png"
        with Image.open(img_path_cor) as img_cor:
            img_cor_raw = np.array(img_cor)

        # Stack sagittal and coronal along channel dimension (H, W, 2)
        image = np.stack((img_sag_raw, img_cor_raw), axis=-1)

        # Load label (sagittal label)
        label_path = os.path.join(self.label_path_sagittal, str(self.targets[index]) + "_sag.png")
        with Image.open(label_path) as label_load:
            label_sag = np.array(label_load, dtype=np.uint8)

        # Load coronal label
        label_path_cor = os.path.join(self.label_path_coronal, str(self.targets[index]) + "_cor.png")
        with Image.open(label_path_cor) as label_cor_load:
            label_cor = np.array(label_cor_load, dtype=np.uint8)

        # Stack sagittal and coronal labels along channel dimension (H, W, 2)
        segmentation_mask = np.stack((label_sag, label_cor), axis=-1)


        if self.transforms is not None:
            # Apply the same transforms to the stacked image and mask
            transformed = self.transforms(image=image, mask=segmentation_mask)
            image = transformed['image']
            segmentation_mask_org = transformed['mask']

        # Resize image and mask
        image = Image.fromarray(image)  # Now image is PIL, preserving 2 channels if supported
        image = self.resize(image)
        image = np.array(image)  # Back to numpy (H, W, 2)

        segmentation_mask = Image.fromarray(segmentation_mask_org)
        segmentation_mask = self.resize(segmentation_mask)
        segmentation_mask = np.array(segmentation_mask, dtype=np.uint8)  # (H, W, 2)

        # Convert to torch tensors
        # image: (H, W, 2) -> (2, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        print(f"dataloader image: {image.shape()}")

        # segmentation_mask: (H, W, 2) -> (2, H, W)
        segmentation_mask = torch.from_numpy(segmentation_mask).permute(2, 0, 1).long()
        print(f"dataloader target: {segmentation_mask.shape()}")

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': segmentation_mask,
            'row_ids': self.row_ids[index],
            'images': image,
            'sentence': text,
            'Label_Name': label_name
        }

    def shuffledTextAugmentation(text):

        sentences = sent_tokenize(text)
        random.shuffle(sentences)
        shuffledText = sentences[0]
        for i in range(1, len(sentences)):
            shuffledText += " " + sentences[i]
        return shuffledText


    def synonymsReplacement(self, text):

        wordDict = self.wordDict

        newText = text
        for word in list(wordDict["synonyms"].keys()):
            if word in text:
                randValue = random.uniform(0, 1)
                if randValue <= .15:
                    randomSample = np.random.randint(low = 0, high = len(wordDict['synonyms'][word]))
                    newText = text.replace(word, wordDict["synonyms"][word][randomSample])

        return newText