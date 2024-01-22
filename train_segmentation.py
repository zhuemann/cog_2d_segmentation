import os
from sklearn import model_selection
import torchvision.transforms as transforms
from transformers import AutoTokenizer, RobertaModel, BertModel, T5Model, T5Tokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict


import numpy as np
import gc
import albumentations as albu

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

import torch.nn.functional as F


#from PIL import Image


#from sklearn import metrics
#from sklearn.metrics import accuracy_score, hamming_loss
from models.ConTextual_seg_vis_model import Attention_ConTEXTual_Vis_Seg_Model
from dataloader_image_text import TextImageDataset
#from vit_base import ViTBase16
#from utility import compute_metrics
from utility import dice_coeff
#from vgg16 import VGG16
#import matplotlib.pyplot as plt

import ssl
import nltk
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE




def train_image_text_segmentation(config, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):
    nltk.download('punkt')
    # model specific global variables
    IMG_SIZE = config["IMG_SIZE"] #256 #1024 #512 #384
    #BATCH_SIZE = batch_size
    LR = 30e-5 #30e-5  #1e-4 #5e-5 #5e-5 was lr for contextualnet runs #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
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

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True

    #dataframe_location = os.path.join(dir_base, "Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_text_negatives_balanced_df.xlsx")
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_positive_text_df.xlsx')
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_text_df.xlsx') #pneumothorax_df chest_tube_df rib_fracture
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx')
    # gets the candid labels and saves it off to the location
    #df = get_candid_labels(dir_base=dir_base)
    #df = get_all_text_image_pairs(dir_base=dir_base)
    #print(df)
    #df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time

    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    #print(df)
    #df.set_index("image_id", inplace=True)

    #wordReplacementPath = os.path.join(dir_base, 'Zach_Analysis/lymphoma_data/words_and_their_synonyms.xlsx')

    #dfWord = pd.read_excel(wordReplacementPath, engine='openpyxl')
    #dfWord.set_index("word", inplace=True)

    #wordDict = dfWord.to_dict()
    #for key in list(wordDict["synonyms"].keys()):
    #    string = wordDict["synonyms"][key][2:-2]
    #    wordList = string.split("', '")
    #    wordDict["synonyms"][key] = wordList

    #word_synonom = pd.read_excel(word_synonom_path, engine='openpyxl')
    #ngram_synonom = pd.read_excel(ngram_synonom_path, engine='openpyxl')


    bert_path = os.path.join(dir_base, 'Zach_Analysis/models/rad_bert/')
    #bert_path = os.path.join(dir_base, 'Zach_Analysis/models/roberta_pretrained_v3')
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    language_model = RobertaModel.from_pretrained(bert_path, output_hidden_states=True)
    #language_model = None
    # use t5 as text encoder
    #t5_path = os.path.join(dir_base, 'Zach_Analysis/models/t5_large/')
    #tokenizer = T5Tokenizer.from_pretrained(t5_path)
    #language_model = T5Model.from_pretrained(t5_path)

    #language_model = None
    #tokenizer = None

    #language_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    #tokenizer = AutoTokenizer.from_pretrained(language_path)
    #language_model = RobertaModel.from_pretrained(language_path, output_hidden_states=True)
    #load in a language model used in the contrastive learning
    """
    pretrained_model = False
    if pretrained_model:
        roberta_path_contrastive_pretraining = os.path.join(dir_base,
                                       'Zach_Analysis/models/candid_pretrained_models/roberta/full_gloria')

        gloria_model = GLoRIA(config=config, tokenizer=tokenizer, language_model=language_model)
        gloria_model.load_state_dict(torch.load(roberta_path_contrastive_pretraining))
        state_dict = gloria_model.text_encoder.state_dict()
        # create new OrderedDict that does not contain `model.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # remove `model.`
            new_state_dict[name] = v

        language_model.load_state_dict(new_state_dict)

    #language_model.load_state_dict(new_state_dict)
    """

    # takes just the last 512 tokens if there are more than 512 tokens in the text
    # df = truncate_left_text_dataset(df, tokenizer)

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
    data_base_path = os.path.join(dir_base, "Zach_Analysis/cog_data_splits/mip_dataframe/")
    train_df = pd.read_excel(data_base_path + "baseline_training_validation_cropped_coronal.xlsx", index_col="image")
    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        train_df, train_size=.85, random_state=seed, shuffle=True  # stratify=df.label.values
    )
    valid_df = test_valid_df
    test_df = test_valid_df

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
            #albu.RandomCrop(height = 256, width=256, always_apply=True),    #take out later
            #albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)

            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                #albu.RandomBrightness(),
                       ], p=.3),
            albu.OneOf([
                #albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            ], p=.3),
            #albu.ShiftScaleRotate()

        ])


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
    output_resize = transforms.Compose([transforms.Resize(IMG_SIZE)]) #407 x 907


    print("train_df")
    print(train_df)
    print("valid df")
    print(valid_df)
    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = albu_augs, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None, norm = None)
    valid_set =    TextImageDataset(valid_df, tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None, norm = normalize)
    test_set =     TextImageDataset(test_df,  tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None, norm = normalize)

    train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 8
                }

    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 4
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
    test_obj = monai.networks.nets.DynUNet(spatial_dims=2, in_channels=3, out_channels=1, kernel_size = (3,3,3,3,3), strides=(2,2,2,2,2), upsample_kernel_size = (2, 2, 2, 2), filters=None, dropout=0.1, norm_name=('INSTANCE', {'affine': True}), act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), deep_supervision=False, deep_supr_num=1, res_block=False, trans_bias=False)
    #test_obj = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=2, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}), norm=('instance', {'affine': True}), bias=True, dropout=0.0, upsample='deconv')

    #weight = torch.load("/UserData/Zach_Analysis/models/swin/model_swinvit.pt")
    #test_obj.load_from(weights=weight)
    #print("Using pretrained self-supervied Swin UNETR backbone weights !")
    # was this one before coming back 3/20
    #test_obj = Attention_ConTEXTual_Lang_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)

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

    # Print the total number of parameters
    total_params = sum(p.numel() for p in test_obj.parameters())
    print(f"Total Parameters: {total_params}")

    #print("need to unfreeze lang params")
    for param in language_model.parameters():
        param.requires_grad = False

    num_unfrozen_layers = 1  # Replace N with the number of layers you want to unfreeze
    #for param in language_model.encoder.layer[-num_unfrozen_layers:].parameters():
    #    param.requires_grad = True

    #test_obj = Attention_ConTEXTual_Seg_Model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)

    test_obj.to(device)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = DiceBCELoss()

    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=test_obj.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=127800, eta_min=5e-5, last_epoch=-1, verbose=False)

    #optimizer = torch.optim.Adam(params=test_obj.parameters(), lr=LR) # was used for all the baselines
    #optimizer_vis = torch.optim.Adam(params = vision_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer_lang = torch.optim.Adam(params=language_model.parameters(), lr=LR, weight_decay=1e-6)
    #optimizer = torch.optim.Adam(params= list(vision_model.parameters()) + list(language_model.parameters()), lr=LR, weight_decay=1e-6)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    #scheduler = MultiStepLR(optimizer, milestones=[5, 10, 25, 37, 50, 75], gamma=0.50)


    #print(test_dataframe_location)
    print("about to start training loop")
    lowest_loss = 100
    best_acc = 0
    del train_df
    valid_log = []
    avg_loss_list = []
    for epoch in range(1, N_EPOCHS + 1):
        #vision_model.train()
        #language_model.train()
        #model_obj.train()
        test_obj.train()
        training_dice = []
        gc.collect()
        torch.cuda.empty_cache()
        #if epoch > 1000:
        #    for param in language_model.encoder.layer[-num_unfrozen_layers:].parameters():
        #        param.requires_grad = True

        loss_list = []
        #print(scheduler.get_lr())
        prediction_sum = 0

        for _, data in tqdm(enumerate(training_loader, 0)):

            #print(torch.cuda.memory_summary(device=None, abbreviated=False))

            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            #print(targets.sum())
            #outputs = test_obj(images, ids, mask)  # for lavt
            outputs = test_obj(images, ids, mask, token_type_ids)
            #outputs = test_obj(images)

            #print(outputs.size())
            outputs = output_resize(torch.squeeze(outputs, dim=1))
            #print(outputs.size())
            #outputs = torch.squeeze(outputs)
            #print(outputs.size())
            #targets = output_resize(targets)
            optimizer.zero_grad()

            loss = criterion(outputs, targets)

            if _ % 400 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()


            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            prediction_sum += torch.sum(outputs)

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                # gives a dice score of 1 if correctly predicts negative
                if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                    dice = 1

                training_dice.append(dice)

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
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                targets = torch.squeeze(targets)
                images = data['images'].to(device, dtype=torch.float)

                #outputs = model_obj(images)
                #outputs = test_obj(images, ids, mask)  # for lavt
                outputs = test_obj(images, ids, mask, token_type_ids)
                #outputs = test_obj(images)

                outputs = output_resize(torch.squeeze(outputs, dim=1))
                #outputs = torch.squeeze(outputs)
                targets = output_resize(targets)

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs)
                outputs = torch.round(sigmoid)
                prediction_sum += torch.sum(outputs)

                # calculates the dice coefficent for each image and adds it to the list
                for i in range(0, outputs.shape[0]):
                    dice = dice_coeff(outputs[i], targets[i])
                    dice = dice.item()
                    if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                        dice = 1
                    valid_dice.append(dice)

            #scheduler.step()
            avg_valid_dice = np.average(valid_dice)
            print(f"Epoch {str(epoch)}, Average Valid Dice Score = {avg_valid_dice}")
            print(f"validation prediction sum: {prediction_sum}")

            valid_log.append(avg_valid_dice)

            if avg_valid_dice >= best_acc:
                best_acc = avg_valid_dice

                #print(f"save location: {config['save_location']}")
                # save_path = os.path.join(dir_base, 'Zach_Analysis/models/vit/best_multimodal_modal_forked_candid')
                save_path = os.path.join(config["save_location"], "best_segmentation_model_seed_test" + str(seed))
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
    with torch.no_grad():
        test_dice = []
        gc.collect()
        prediction_sum = 0
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)
            row_ids = data['row_ids']
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

            for i in range(0, outputs.shape[0]):
                output_item = outputs[i].cpu().data.numpy()
                target_item = targets[i].cpu().data.numpy()
                pred_rle = mask2rle(output_item)
                target_rle= mask2rle(target_item)
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

        avg_test_dice = np.average(test_dice)
        print(f"Epoch {str(epoch)}, Average Test Dice Score = {avg_test_dice}")
        print(f"testing prediction sum: {prediction_sum}")

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

        filepath = os.path.join(config["save_location"], "prediction_dataframe" + str(seed) + '.xlsx')
        test_df_data.to_excel(filepath, index=False)

        return avg_test_dice, valid_log





