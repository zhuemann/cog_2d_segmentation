import os
from sklearn import model_selection
import torchvision.transforms as transforms
from transformers import AutoTokenizer, RobertaModel, BertModel, T5Model, T5Tokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from collections import OrderedDict
from monai.losses import DeepSupervisionLoss

import numpy as np
import gc
import albumentations as albu
from .utility_3d_training import get_max_pixel_value_3d
import monai
from timm.models.swin_transformer import SwinTransformer
#from models.swin_model import SwinModel
#from transformers import SwinConfig, SwinModel
import timm
#from models.Gloria import GLoRIA
import segmentation_models_pytorch as smp
#from create_unet import load_img_segmentation_model
from models.ConTextual_seg_lang_model import Attention_ConTEXTual_Lang_Seg_Model
from utility import mask2rle
from data_prepocessing.template_removal import template_removal

import torch.nn.functional as F

import copy
import cc3d

#from PIL import Image
from monai.optimizers.lr_scheduler import WarmupCosineSchedule


#from sklearn import metrics
#from sklearn.metrics import accuracy_score, hamming_loss
from models.ConTextual_seg_vis_model import Attention_ConTEXTual_Vis_Seg_Model
#from dataloader_image_text import TextImageDataset
from segmentation_3d.dataloader_3d import TextImageDataset
#from vit_base import ViTBase16
#from utility import compute_metrics
from utility import dice_coeff
#from vgg16 import VGG16
#import matplotlib.pyplot as plt

import ssl
import nltk
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL
from .ConTEXTual_Net_3D import ConTEXTual_Net_3D
import json
from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianSmoothd,
    RandGaussianNoised,
    SpatialPadd,
    CenterSpatialCropd,
    Flipd,
    ScaleIntensityRange,
    LoadImaged,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    EnsureTyped
)

from .utility_3d_training import logits2pred
from .utility_3d_training import DiceBCELoss
from .utility_3d_training import TPFPFNHelper
from .utility_3d_training import DiceHelper

def train_3d_image_text_segmentation(config, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):
    nltk.download('punkt')
    # model specific global variables
    IMG_SIZE = config["IMG_SIZE"] #256 #1024 #512 #384
    #BATCH_SIZE = batch_size
    LR = 1e-5 #5e-5 #30e-5  #1e-4 #5e-5 #5e-5 was lr for contextualnet runs #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    #LR = 5e-4
    N_EPOCHS = epoch
    N_CLASS = n_classes
    #LR = 1e-4
    dir_base = config["dir_base"]
    seed = config["seed"]
    BATCH_SIZE = config["batch_size"]
    N_EPOCHS = config["epochs"]
    #LR = config["LR"]

    # the folder in which the test dataframe, model, results log will all be saved to
    save_location = config["save_location"]

    bert_path = os.path.join(dir_base, 'Zach_Analysis/models/rad_bert/')
    #bert_path = os.path.join(dir_base, 'Zach_Analysis/models/roberta_pretrained_v3')
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    language_model = RobertaModel.from_pretrained(bert_path, output_hidden_states=True)

    """
    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=config["train_samples"], random_state=seed, shuffle=True #stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=config["test_samples"], random_state=seed, shuffle=True #stratify=test_valid_df.label.values
    )

    train_dataframe_location = os.path.join(save_location,'pneumothorax_train_df_seed' + str(config["seed"]) + '.xlsx')
    print(train_dataframe_location)
    train_df.to_excel(train_dataframe_location, index=True)

    valid_dataframe_location = os.path.join(save_location,'pneumothorax_valid_df_seed' + str(config["seed"]) + '.xlsx')
    print(valid_dataframe_location)
    valid_df.to_excel(valid_dataframe_location, index=True)

    test_dataframe_location = os.path.join(save_location, 'pneumothorax_testset_df_seed' + str(config["seed"]) + '.xlsx')
    print(test_dataframe_location)
    test_df.to_excel(test_dataframe_location, index=True)
    """
    """
    data_base_path = os.path.join(dir_base, "Zach_Analysis/cog_data_splits/mip_dataframe/")
    train_df = pd.read_excel(data_base_path + "baseline_training_validation_cropped_coronal.xlsx", index_col="image")
    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        train_df, train_size=.85, random_state=seed, shuffle=True  # stratify=df.label.values
    )
    valid_df = test_valid_df
    test_df = test_valid_df
    """
    """
    data_base_path = os.path.join(dir_base, "Zach_Analysis/petlymph_image_data/")
    #train_df = pd.read_excel(data_base_path + "unique_labels_uw_lymphoma_anon_4_renumbered_v3.xlsx")
    #train_df = pd.read_excel(data_base_path + "remove_dups_df_5.xlsx")
    #train_df = pd.read_excel(data_base_path + "dropped_problem_segs_6_v6.xlsx")
    #train_df = pd.read_excel(data_base_path + "uw_final_df_9.xlsx")
    train_df = pd.read_excel(data_base_path + "uw_final_df_9_all.xlsx")

    # Specified labels to skip
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1",
                      "PETWB_012579_01_label_2", "PETWB_003190_01_label_3",
                      "PETWB_011401_02_label_3"]

    # Filtering .png files
    files_directory = "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/mip_with_side_errors_10/"
    files = os.listdir(files_directory)
    stripped_files = [file[:-4] for file in files if file.endswith('.png')]

    print(f"before drop length: {len(train_df)}")
    # Filtering the DataFrame
    train_df = train_df[~train_df["Label_Name"].isin(labels_to_skip)]
    train_df = train_df[~train_df["Label_Name"].isin(stripped_files)]
    print(f"after dropped length: {len(train_df)}")

    train_df = template_removal(train_df)
    print(f"after template removal {len(train_df)}")

    # Grouping by 'Petlymph' and splitting the groups
    groups = train_df.groupby('Petlymph')
    group_list = [group for _, group in groups]

    # Splitting the groups into train, validation, and test
    train_idx, test_valid_idx = model_selection.train_test_split(range(len(group_list)), train_size=0.80,
                                                                 random_state=seed)
    test_idx, valid_idx = model_selection.train_test_split(test_valid_idx, test_size=config["test_samples"],
                                                           random_state=seed)

    # Reconstruct DataFrames from grouped indices
    train_df = pd.concat([group_list[i] for i in train_idx])
    test_df = pd.concat([group_list[i] for i in test_idx])
    valid_df = pd.concat([group_list[i] for i in valid_idx])

    train_df.set_index("Petlymph", inplace=True)
    valid_df.set_index("Petlymph", inplace=True)
    test_df.set_index("Petlymph", inplace=True)
    """

    """
    print(f"before dropped length: {len(train_df)}")
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1", "PETWB_012579_01_label_2", "PETWB_003190_01_label_3",
                      "PETWB_011401_02_label_3"]

    files = os.listdir("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/mip_with_side_errors_10/")
    # Filter out only the .png files and strip the .png extension
    stripped_files = [file[:-4] for file in files if file.endswith('.png')]
    train_df = train_df[~train_df["Label_Name"].isin(labels_to_skip)]
    train_df = train_df[~train_df["Label_Name"].isin(stripped_files)]
    print(f"after dropped length: {len(train_df)}")


    train_df.set_index("Petlymph", inplace=True)

    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        train_df, train_size=.80, random_state=seed, shuffle=True  # stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=config["test_samples"], random_state=seed, shuffle=True
        # stratify=test_valid_df.label.values
    )
    """

    data_base_path = os.path.join(dir_base, "Zach_Analysis/uw_lymphoma_pet_3d/dataframes/")
    train_df = pd.read_excel(data_base_path + "training.xlsx")
    valid_df = pd.read_excel(data_base_path + "validation.xlsx")
    test_df = pd.read_excel(data_base_path + "testing.xlsx")


    #train_df = train_df.head(25)
    #valid_df = valid_df.head(25)
    #valid_df = test_valid_df
    #test_df = test_valid_df


    #valid_df =  pd.read_excel(dataframe_location, index=False)
    #test_df = pd.read_excel(dataframe_location, index=False)
    # delete this block later
    #train_frame_locaction = os.path.join(dir_base,
    #                                    "Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    #                                    "multisegmentation_model_train_v13/seed98/pneumothorax_df_trainseed98_edited.xlsx")
    #train_df = pd.read_excel(train_frame_locaction, engine='openpyxl')

    #train_df['image_id'].replace('', np.nan, inplace=True)
    #train_df.dropna(subset=['image_id'], inplace=True)
    #train_df.set_index("image_id", inplace=True)


    #valid_frame_locaction = os.path.join(dir_base,
    #                                    "Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    #                                    "multisegmentation_model_train_v13/seed98/pneumothorax_df_validseed98_copied.xlsx")
    #valid_df = pd.read_excel(valid_frame_locaction, engine='openpyxl')
    #valid_df.set_index("image_id", inplace=True)

    #test_frame_locaction = os.path.join(dir_base,
    #                                    "Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/with_augmentation/" +
    #                                    "multisegmentation_model_train_v13/seed98/pneumothorax_testset_df_seed98_copied.xlsx")
    #test_df = pd.read_excel(test_frame_locaction, engine='openpyxl')
    #test_df.set_index("image_id", inplace=True)

    #print(test_dataframe_location)

    # report invariant augmentaitons
    using_t5 = True
    if using_t5:
        albu_augs = albu.Compose([
            #albu.RandomCrop(height = 64, width=128, always_apply=True),    #take out later
            #albu.CropNonEmptyMaskIfExists(height=64, width=128, always_apply=True),
            #albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)

            albu.OneOf([
                # albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
                       ], p=.3),
            albu.OneOf([
                # albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            ], p=.3),
            albu.ShiftScaleRotate()

        ])
        """
        best augmetnations so far
        if using_t5:
            albu_augs = albu.Compose([
                albu.OneOf([
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                ], p=.3),
                albu.OneOf([
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                ], p=.3),
                albu.ShiftScaleRotate()
            ])
            """


    # emprically the good augmentations, taken from kaggle winner
    vision_only = False
    if vision_only:
        albu_augs = albu.Compose([
            #albu.HorizontalFlip(p=.5),
            #albu.CLAHE(),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=.3),
            albu.OneOf([
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=.3),
            albu.ShiftScaleRotate(),
            #albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ])
    #albu_augs = albu.Compose([])
    # used for empty augmentation tests
    #if not vision_only and not using_t5:
        #albu_augs = albu.Compose([
#
#        ])

    normalize = albu.Compose([
        #albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ])

    transforms_valid = transforms.Compose(
        [
            #transforms.RandomCrop(size=(256, 256)),
            #transforms.RandomHorizontalFlip(p=1),
            #transforms.Resize((IMG_SIZE, IMG_SIZE)),
            #albu.BBoxSafeRandomCrop(erosion_rate=.25, always_apply=True),
            #albu.CropNonEmptyMaskIfExists(height=64, width=128, always_apply=True),
            #albu.RandomCrop(height=64, width=128, always_apply=True),
            transforms.Resize(IMG_SIZE),
            #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            #albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            transforms.PILToTensor(),
            #transforms.ToTensor(), #test was pilToTesnor

            #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # transforms.Normalize((0.5,), (0.5,))
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Normalize([0.5], [0.5])
        ]
    )

    #transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    transforms_resize = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.PILToTensor()])
    #output_resize = transforms.Compose([transforms.Resize(IMG_SIZE)]) #407 x 907


    #print("train_df")
    #print(train_df)
    #print("valid df")
    #print(valid_df)
    keys = ['pet', 'ct', 'label']
    transforms_training = Compose([
        RandAffined(
            keys=keys,
            prob=0.1,
            rotate_range=[0.05, 0.05, 0.05],
            scale_range=[0.05, 0.05, 0.05],
            mode="bilinear",
            #spatial_size=self.roi_size,
            cache_grid=True,
            padding_mode="border",
        ),
        RandGaussianSmoothd(
            keys= ["pet", "ct"], prob=.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0) # prob was .2
        ),
        RandGaussianNoised(keys= ["pet", "ct"], prob=0.2, mean=0.0, std=0.1)
    ])

    """
    def debug_spatial_transform(data_dic):
        print(f"Shape of images: {data_dic['images'].shape}", flush=True)
        print(f"Shape of Label: {data_dic['label'].shape}", flush=True)
        return data_dic

    transforms_resize = Compose([
        debug_spatial_transform,
        SpatialPadd(keys = ['images', 'label'], spatial_size=(None, 192, 192, None), mode="constant", method="symmetric", constant_values=0),
        CenterSpatialCropd(keys = ['images', 'label'], roi_size=(None, 192, 192, -1)),
        debug_spatial_transform,
        # ts.append(SpatialPadd(keys = [pet_key, "label"], spatial_size = (200, 200, None), mode = "constant", method="symmetric", constant_values=0))
        # ts.append(SpatialPadd(keys = keys, spatial_size = (None, None, 680), mode = "constant", method="start"))
        # ts.append(SpatialPadd(keys = [ct_key], spatial_size = (200, 200, None), mode = "constant", method="symmetric", constant_values=-1000))

        Flipd(keys = ['images', 'label'], spatial_axis=-1), # Flip along the last dimension
        SpatialPadd(keys = ['images', 'label'], spatial_size=(None, None, None, 352), mode="constant", method="end"),
        # Pad from the end (which is the start of the original after flipping)
        Flipd(keys = ['images', 'label'], spatial_axis=-1),
        debug_spatial_transform

    ])
    """


    intensity_bounds_pt = [0, 10]
    normalize_pet_transforms = Compose([
        ScaleIntensityRange(a_min=intensity_bounds_pt[0], a_max=intensity_bounds_pt[1], b_min=0, b_max=1, clip=True)
    ])

    intensity_bounds_ct = [-150, 250]
    normalize_ct_transforms = Compose([
        ScaleIntensityRange(a_min=intensity_bounds_ct[0], a_max=intensity_bounds_ct[1], b_min=0, b_max=1, clip=True)
    ])


    def debug_spatial_transform(data_dic):
        print(f"Shape of PET: {data_dic['pet'].shape}", flush=True)
        print(f"Shape of CT: {data_dic['ct'].shape}", flush=True)
        print(f"Shape of Label: {data_dic['label'].shape}", flush=True)
        return data_dic

    length = 352
    #length = 176
    #length = 144
    #length = 112

    transforms_resize = Compose([
        #debug_spatial_transform,
        SpatialPadd(keys = ['pet', 'ct', 'label'], spatial_size=(192, 192, None), mode="constant", method="symmetric", constant_values=0),
        CenterSpatialCropd(keys = ['pet', 'ct', 'label'], roi_size=(192, 192, length)),
        #debug_spatial_transform,
        # ts.append(SpatialPadd(keys = [pet_key, "label"], spatial_size = (200, 200, None), mode = "constant", method="symmetric", constant_values=0))
        # ts.append(SpatialPadd(keys = keys, spatial_size = (None, None, 680), mode = "constant", method="start"))
        # ts.append(SpatialPadd(keys = [ct_key], spatial_size = (200, 200, None), mode = "constant", method="symmetric", constant_values=-1000))

        Flipd(keys = ['pet', 'ct', 'label'], spatial_axis=-1), # Flip along the last dimension
        SpatialPadd(keys = ['pet', 'ct', 'label'], spatial_size=(None, None, length), mode="constant", method="end"),
        # Pad from the end (which is the start of the original after flipping)
        Flipd(keys = ['pet', 'ct', 'label'], spatial_axis=-1),
        #debug_spatial_transform

    ])
    """
    transforms_resize = Compose([
        # Pad symmetrically in all dimensions to ensure minimum size
        SpatialPadd(
            keys=['pet', 'ct', 'label'],
            spatial_size=(192, 192, length),
            mode="constant",
            method="symmetric",
            constant_values=0
        ),
        # Center crop to the desired size
        CenterSpatialCropd(
            keys=['pet', 'ct', 'label'],
            roi_size=(192, 192, length)
        ),
    ])"""


    """
    transforms_resize = Compose([
        SpatialPadd(keys = ['pet', 'ct', 'label'], spatial_size=(192, 192, None), mode="constant", method="symmetric", constant_values=0),
        CenterSpatialCropd(keys = ['pet', 'ct', 'label'], roi_size=(192, 192, -1)),
        # ts.append(SpatialPadd(keys = [pet_key, "label"], spatial_size = (200, 200, None), mode = "constant", method="symmetric", constant_values=0))
        # ts.append(SpatialPadd(keys = keys, spatial_size = (None, None, 680), mode = "constant", method="start"))
        # ts.append(SpatialPadd(keys = [ct_key], spatial_size = (200, 200, None), mode = "constant", method="symmetric", constant_values=-1000))

        Flipd(keys = ['pet', 'ct', 'label'], spatial_axis=-1), # Flip along the last dimension
        SpatialPadd(keys = ['pet', 'ct', 'label'], spatial_size=(None, None, 352), mode="constant", method="end"),
        # Pad from the end (which is the start of the original after flipping)
        Flipd(keys = ['pet', 'ct', 'label'], spatial_axis=-1)
    ])
    """


    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = transforms_training, pet_norm = normalize_pet_transforms, ct_norm = normalize_ct_transforms, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None, norm = None)
    valid_set =    TextImageDataset(valid_df, tokenizer, 512,               transforms = None, pet_norm = normalize_pet_transforms, ct_norm = normalize_ct_transforms, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None, norm = normalize)
    test_set =     TextImageDataset(test_df,  tokenizer, 512,               transforms = transforms_valid, pet_norm = normalize_pet_transforms, ct_norm = normalize_ct_transforms, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None, norm = normalize)

    print(f"test set: {len(training_set)}")

    train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 8
                }

    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 8
                   }

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load_model = True
    #if load_model:
        # model is orginally from here which was saved and reloaded to get around SSL
    #    model_obj = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    #    save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp/resnet34')
    #    torch.save(model_obj.state_dict(), save_path)
    #else:
    #    model_obj = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=1) #timm-efficientnet-b8 resnet34 decoder_channels=[512, 256, 128, 64, 32]
    #    save_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp/resnet50')
    #    model_obj.load_state_dict(torch.load(save_path))

    run_from_checkpoint = False
    if run_from_checkpoint:
        checkpoint_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_pretrained_models/bert/full_gloria_checkpoint_40ep')
        #gloria_model.load_state_dict(torch.load(checkpoint_path))


    #gloria_model.to(device)

    #language_model.to(device)
    #model_obj.to(device)

    #test_obj = ConTEXTual_seg_model(lang_model=language_model, n_channels=1, n_classes=1, bilinear=False)

    #test_obj = Attention_ConTEXTual_Seg_Model_swap_v3(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)

    #test_obj = Attention_ConTEXTual_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False) #<----- this one
    #test_obj = Unet_Baseline(n_channels=3, n_classes=1, bilinear=False)

    # Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
    #configuration = SwinConfig(image_size = 1024, num_channels=1)

    # Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
    #test_obj = SwinModel(configuration)

    #test_obj.head = nn.Sequential(
    #    nn.Conv2d(in_channels=test_obj.num_channels, out_channels=1, kernel_size=1),
    #    nn.Sigmoid()
    #)
    #model = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2],
    #                        num_heads=[3, 6, 12, 24], window_size=7)

    #swin_path = "/UserData/Zach_Analysis/git_multimodal/lavt/LAVT/pretrained_weights/swin_base_patch4_window12_384_22k.pth"
    #swin_transformer = timm.create_model(
    #    'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    #    pretrained=True,
    #    features_only=True,
    #)
    #test_obj = SwinModel(backbone=model)

    #test_obj = monai.networks.nets.SwinUNETR(img_size=(1024, 1024), in_channels= 3, out_channels = 1, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), feature_size=24, norm_name='instance', drop_rate=0.1, attn_drop_rate=0.0, dropout_path_rate=0.0, normalize=True, use_checkpoint=False, spatial_dims=2, downsample='merging', use_v2=False)
    #test_obj = monai.networks.nets.DynUNet(spatial_dims=2, in_channels=3, out_channels=1, kernel_size = (3,3,3,3,3), strides=(2,2,2,2,2), upsample_kernel_size = (2, 2, 2, 2), filters=None, dropout=0.1, norm_name=('INSTANCE', {'affine': True}), act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), deep_supervision=False, deep_supr_num=1, res_block=False, trans_bias=False)
    #test_obj = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=2, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv')

    #weight = torch.load("/UserData/Zach_Analysis/models/swin/model_swinvit.pt")
    #test_obj.load_from(weights=weight)
    #print("Using pretrained self-supervied Swin UNETR backbone weights !")

    # was this one before coming back 3/20
    #test_obj = Attention_ConTEXTual_Lang_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False) # 2d version we left behind

    #test_obj = Attention_ConTEXTual_Vis_Seg_Model(n_channels=3, n_classes=1, bilinear=False)
    #test_obj = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    #model_path = os.path.join(dir_base, 'Zach_Analysis/models/smp_models/default_from_smp_three_channel/resnet50')
    #test_obj.load_state_dict(torch.load(model_path))
    #test_obj = Vision_Attention_UNet_Model(n_channels=3, n_classes=1, bilinear=False)

    #test_obj = Vision_Attention_UNet_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)
    #test_obj = Unet_Baseline(n_channels=3, n_classes=1, bilinear=True)
    #test_obj = ResAttNetUNet(lang_model=language_model, n_class=1, dir_base=dir_base)
    #test_obj = SwinUNETR(
    #    img_size=(960, 960, 96),
    #    in_channels=3,
    #    out_channels=1,
    #    feature_size=48,
    #    use_checkpoint=True,
    #).to(device)
    # This is for LAVT
    #test_obj = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    #test_obj = load_img_segmentation_model(dir_base = dir_base, pretrained_model=False)

    # Iterate over all parameters in the language model
    for param in language_model.parameters():
        param.requires_grad = False  # Disable gradient updates for these parameters

    n_class = 2
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    deep_supr_num = len(strides) - 2
    test_obj = ConTEXTual_Net_3D(spatial_dims=3,
                    in_channels=2,  # in_channels changed to 1 from in_channels
                    out_channels=n_class,
                    kernel_size=kernels,
                    filters=[64, 96, 128, 192, 256],
                    strides=strides,
                    upsample_kernel_size=strides[1:],
                    res_block=True,
                    norm_name="instance",
                    deep_supervision=False,
                    deep_supr_num=deep_supr_num,
                    language_model=language_model)

    #for param in test_obj.parameters():
    #    print(param.requires_grad)

    # Print the total number of parameters
    total_params = sum(p.numel() for p in test_obj.parameters())
    print(f"Total Parameters: {total_params}")

    #print("need to unfreeze lang params")
    #for param in language_model.parameters():
    #   param.requires_grad = False

    num_unfrozen_layers = 1  # Replace N with the number of layers you want to unfreeze
    #for param in language_model.encoder.layer[-num_unfrozen_layers:].parameters():
    #    param.requires_grad = True

    #test_obj = Attention_ConTEXTual_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)

    test_obj.to(device)

    criterion = nn.BCEWithLogitsLoss()
    #criterion = DiceBCELoss()
    loss_string = "DiceCELoss"
    criterion = DeepSupervisionLoss(loss_string)

    grad_scaler = torch.cuda.amp.GradScaler()

    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=test_obj.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=127800, eta_min=5e-5, last_epoch=-1, verbose=False)

    #optimizer = torch.optim.Adam(params=test_obj.parameters(), lr=LR) # was used for all the baselines
    #optimizer_vis = torch.optim.Adam(params = vision_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer_lang = torch.optim.Adam(params=language_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer = torch.optim.Adam(params= list(vision_model.parameters()) + list(language_model.parameters()), lr=LR, weight_decay=1e-6)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    #scheduler = MultiStepLR(optimizer, milestones=[5, 10, 25, 37, 50, 75], gamma=0.50)
    total_steps = config["epochs"]*len(train_df)
    lr_scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=5, end_lr=1e-6, cycles=2,
                                        warmup_multiplier=0.1, t_total=total_steps)

    acc_function = TPFPFNHelper()
    dice_function = DiceHelper(sigmoid=False)
    #print(test_dataframe_location)
    print("about to start training loop")
    lowest_loss = 100
    best_acc = 0
    valid_log = []
    correct_suv_log = []
    avg_loss_list = []
    for epoch in range(1, N_EPOCHS + 1):
        #vision_model.train()
        #language_model.train()
        #model_obj.train()
        test_obj.train()
        training_dice = []
        gc.collect()
        torch.cuda.empty_cache()
        #if epoch > 200:
        #    for param in language_model.encoder.layer[-num_unfrozen_layers:].parameters():
        #        param.requires_grad = True

        loss_list = []
        run_dice = []
        run_tp = 0
        run_fp = 0
        run_fn = 0
        print(lr_scheduler.get_lr())
        prediction_sum = 0

        for index, data in tqdm(enumerate(training_loader, 0)):

            optimizer.zero_grad()  # Clear gradients before each training step
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))

            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            #targets = data['targets'].to(device, dtype=torch.float)
            #targets = torch.squeeze(targets)
            #images = data['images']['pet'].to(device, dtype=torch.float)

            image_dic = data["images"]
            pet = image_dic["pet"]
            ct = image_dic["ct"]
            targets = image_dic["label"]

            # Stack images on CPU
            images = torch.stack((pet, ct), dim=1).squeeze(2)
            del pet, ct  # Free up memory

            # Move images and targets to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            del image_dic  # Free up memory

            # Proceed with model inference, potentially using AMP
            with autocast():
                outputs = test_obj(images, ids, mask, token_type_ids)
                loss = criterion(outputs, targets)

            del images
            # Backward pass with GradScaler if using AMP
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # update the learning rate
            lr_scheduler.step()

            """
            with torch.no_grad():
                pred = logits2pred(outputs, sigmoid=False)
                #acc = acc_function(pred, target)
                TP, FP, FN = acc_function(pred, targets)
                TP, FP, FN = 0, 0, 0
                run_tp += TP
                run_fp += FP
                run_fn += FN
                #print(f"true positive: {TP} false positive: {TP} false negative: {FN}")
                #dice = dice_function(pred, targets)
                #print(f"Dice: {dice}")
                dice = 0
                run_dice.append(0)
                if isinstance(dice, (list, tuple)):
                    #dice, batch_size_adjusted = dice
                    #print(f"Dice: {dice}")
                    #run_dice.append(dice.detach().cpu().numpy())
                    run_dice.append(0)
            """

            """
            image_dic = data["images"]
            ct = image_dic["ct"].to(device, dtype=torch.float)
            pet = image_dic["pet"].to(device, dtype=torch.float)
            targets = image_dic["label"].to(device, dtype=torch.float)

            images = torch.cat((pet.unsqueeze(1), ct.unsqueeze(1)), dim=1)  # Now it's [batch_size, 2, 1, H, W, D]
            del image_dic
            del ct
            del pet
            images = images.squeeze(2) # Now it's [batch_size, 2, H, W, D]
            #print(f"images: {images.shape}")
            #print(targets.sum())
            #outputs = test_obj(images, ids, mask)  # for lavt
            outputs = test_obj(images, ids, mask, token_type_ids)
            #outputs = test_obj(images)
            #outputs = torch.argmax(outputs, dim=1)
            #print(f"output size: {outputs.size()}")
            #print(f"target size: {targets.size()}")
            #outputs = output_resize(torch.squeeze(outputs, dim=1))
            #print(outputs.size())
            #outputs = torch.squeeze(outputs)
            #print(outputs.size())
            #targets = output_resize(targets)
            #optimizer.zero_grad()

            loss = criterion(outputs, targets)
            #print(f"Loss requires_grad: {loss.requires_grad}")

            #print(f"loss: {loss}")
            if _ % 400 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            """

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            #scheduler.step()


            outputs_detached = outputs.detach()
            targets_detached = targets.detach()
            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs_detached)
            outputs_detached = torch.round(sigmoid)
            prediction_sum += torch.sum(outputs_detached)

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, outputs_detached .shape[0]):
                dice = dice_coeff(outputs_detached[i], targets_detached[i])
                dice = dice.item()
                # gives a dice score of 1 if correctly predicts negative
                #if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                #    dice = 1

                training_dice.append(dice)
            del outputs, targets
            #gc.collect()
            #torch.cuda.empty_cache()
            #if index % 1000 == 0:
            #    print(f"index: {index + 1} True positive: {run_tp} False positive: {run_fp} False negative: {run_fn} Running Average Dice: {sum(run_dice) / len(run_dice)}")

        avg_training_dice = np.average(training_dice)
        print(f"Epoch {str(epoch)}, Average Training Dice Score = {avg_training_dice}")
        print(f"training prediction sum: {prediction_sum}")

        # each epoch, look at validation data
        with torch.no_grad():

            #model_obj.eval()
            test_obj.eval()
            valid_dice = []
            gc.collect()
            prediction_sum = 0
            correct_max_predictions = 0
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

                image_dic = data["images"]
                pet = image_dic["pet"]
                ct = image_dic["ct"]
                targets = image_dic["label"]

                # Stack images on CPU
                images = torch.stack((pet, ct), dim=1).squeeze(2)
                del pet, ct  # Free up memory

                # Move images and targets to device
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                del image_dic  # Free up memory


                #outputs = model_obj(images)
                #outputs = test_obj(images, ids, mask)  # for lavt
                outputs = test_obj(images, ids, mask, token_type_ids)
                #outputs = test_obj(images)

                with torch.no_grad():
                    pred = logits2pred(outputs, sigmoid=False)
                    TP, FP, FN = acc_function(pred, targets)
                    run_tp += TP
                    run_fp += FP
                    run_fn += FN
                    dice = dice_function(pred, targets)
                    if isinstance(dice, (list, tuple)):
                        dice, batch_size_adjusted = dice
                        run_dice.append(dice.detach().cpu().numpy())

                #outputs = output_resize(torch.squeeze(outputs, dim=1))
                #outputs = torch.squeeze(outputs)
                #targets = output_resize(targets)

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                outputs_detached = outputs.detach()
                targets_detached = targets.detach()
                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs_detached)
                outputs_detached = torch.round(sigmoid)
                prediction_sum += torch.sum(outputs_detached)

                max_targets, max_outputs = get_max_pixel_value_3d(images.detach().cpu().numpy(), targets_detached.cpu().numpy(), outputs_detached.cpu().numpy())
                #print(f"max target: {max_targets}")
                #print(f"max outputs: {max_outputs}")
                # calculates the dice coefficent for each image and adds it to the list
                for i in range(0, outputs.shape[0]):
                    dice = dice_coeff(outputs_detached[i], targets_detached[i])
                    dice = dice.item()
                    if torch.max(outputs_detached [i]) == 0 and torch.max(targets_detached[i]) == 0:
                        dice = 1
                    valid_dice.append(dice)
                    if max_outputs == max_targets and max_outputs != 0:
                        correct_max_predictions += 1

            #scheduler.step()
            avg_valid_dice = np.average(valid_dice)
            print(f"Epoch {str(epoch)}, Average Valid Dice Score = {avg_valid_dice}")
            print(f"validation prediction sum: {prediction_sum}")
            print(f"valid correct_max_predictions: {correct_max_predictions} or: {correct_max_predictions/len(valid_df)}%")
            valid_log.append(avg_valid_dice)
            correct_suv_log.append(correct_max_predictions)
            print(f"Validation epoch: {epoch} True positive: {run_tp} False positive: {run_fp} False negative: {run_fn} Running Average Dice: {sum(run_dice) / len(run_dice)}")

            if avg_valid_dice > best_acc:
                best_acc = avg_valid_dice

                #print(f"save location: {config['save_location']}")
                # save_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
                save_path = os.path.join(config["save_location"], "best_segmentation_model_seed_test_3D_working_test" + str(seed))
                #save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_finetuned_segmentation/forked_1/segmentation_forked_candid')
                #save_path = os.path.join(dir_base, 'Zach_Analysis/models/candid_finetuned_segmentation/forked_2/segmentation_forked_candid2')
                #save_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_3/segmentation_forked_candid')

                #save_path = os.path.join(dir_base,
                #                         'Zach_Analysis/models/candid_finetuned_segmentation/weak_supervision_models/imagenet_labeling_functions/segmentation_candid' + str(
                #                             seed))
                # torch.save(model_obj.state_dict(), '/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal')
                torch.save(test_obj.state_dict(), save_path)

    #test_obj.eval()
    row_ids = []
    # saved_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
    saved_path = os.path.join(config["save_location"], "best_segmentation_model_seed_test" + str(seed))
    #saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_1/segmentation_forked_candid')
    #saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_2/segmentation_forked_candid2')
    #saved_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_3/segmentation_forked_candid')

    #saved_path = os.path.join(dir_base,
    #                          'Zach_Analysis/models/candid_finetuned_segmentation/weak_supervision_models/imagenet_labeling_functions/segmentation_candid' + str(
    #                              seed))
    # model_obj.load_state_dict(torch.load('/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal'))
    test_obj.load_state_dict(torch.load(saved_path))
    test_obj.eval()
    pred_rle_list = []
    target_rle_list = []
    ids_list = []
    dice_list = []
    text_list = []
    label_path_list = []
    with torch.no_grad():
        test_dice = []
        gc.collect()
        prediction_sum = 0
        correct_max_predictions = 0
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)
            row_ids = data['row_ids']
            #print(data)
            #print(data.keys())
            sentences = data['sentence']
            label_names = data["Label_Name"]
            #outputs = model_obj(images)
            #outputs = test_obj(images, ids, mask) #for lavt
            outputs = test_obj(images, ids, mask, token_type_ids) #for contextual net
            #outputs = test_obj(images)

            outputs = output_resize(torch.squeeze(outputs, dim=1))
            #outputs = outputs.squeeze(outputs)
            targets = output_resize(targets)



            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            prediction_sum += torch.sum(outputs)
            row_ids.extend(data['row_ids'])

            max_targets, max_outputs = get_max_pixel_value_3d(images, targets, outputs)

            for i in range(0, outputs.shape[0]):
                output_item = outputs[i].cpu().data.numpy()
                target_item = targets[i].cpu().data.numpy()
                pred_rle = mask2rle(output_item)
                target_rle = mask2rle(target_item)
                ids_example = row_ids[i]

                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()

                if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                    dice = 1
                test_dice.append(dice)
                pred_rle_list.append(pred_rle)
                target_rle_list.append(target_rle)
                ids_list.append(ids_example)
                dice_list.append(dice)
                text_list.append(sentences[i])
                label_path_list.append(label_names[i])
                if max_outputs[i] == max_targets[i] and max_outputs[i] != 0:
                    correct_max_predictions += 1

        avg_test_dice = np.average(test_dice)
        print(f"Epoch {str(epoch)}, Average Test Dice Score = {avg_test_dice}")
        print(f"testing prediction sum: {prediction_sum}")
        print(f"test correct_max_predictions: {correct_max_predictions} or: {correct_max_predictions/len(test_df)}")

        #print(f"target: {target_rle_list}")
        #print(f"pred rle: {pred_rle_list}")
        #print(f"length pred rle: {len(pred_rle_list)}")
        #print(f"ids: {ids_list}")
        #print(f"dice: {dice_list}")

        test_df_data = pd.DataFrame(pd.Series(ids_list))
        #test_df_data["ids"] = pd.Series(ids_list)
        test_df_data["dice"] = pd.Series(dice_list)
        test_df_data["target"] = pd.Series(target_rle_list)
        test_df_data["prediction"] = pd.Series(pred_rle_list)
        test_df_data["label_names"] = pd.Series(label_path_list)
        test_df_data["sentence"] = pd.Series(text_list)

        filepath = os.path.join(config["save_location"], "prediction_dataframe" + str(seed) + '.xlsx')
        test_df_data.to_excel(filepath, index=False)

        return avg_test_dice, valid_log, correct_suv_log, correct_max_predictions





