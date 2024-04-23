import pandas as pd

import os
from radgraph.radgraph import GenRadGraph
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import re
import numpy as np

def remove_dups_non_anontomical_sent(df):

    # throw out the sentence if multiple label points were extracted
    df_filtered = df.drop_duplicates(subset=['Petlymph', 'Extracted Sentences'], keep=False)
    print(len(df_filtered))

    # Create an enumeration column for each label per "Petlymph" ID
    df_filtered['Label_Number'] = df_filtered.groupby('Petlymph').cumcount() + 1

    # Create the new column by combining "Petlymph" with the label enumeration
    df_filtered['Label_Name'] = df_filtered['Petlymph'] + '_label_' + df_filtered['Label_Number'].astype(str)

    # Drop the temporary 'Label_Number' column if you don't need it
    df_filtered.drop('Label_Number', axis=1, inplace=True)

    df_path = "/UserData/Zach_Analysis/suv_slice_text/uw_pet_lymphoma_next_and_previous_sentence_annotated.xlsx"
    df_anotomical_info = pd.read_excel(df_path)

    # Merge the two DataFrames on 'Petlymph' and 'Extracted sentences'
    merged_df = pd.merge(df_filtered, df_anotomical_info, on=['Extracted Sentences'])
    filtered_df = merged_df[merged_df['anatomy_available'] != 0]
    df_dropped = filtered_df.drop(
    columns=['Accession Number', 'Report', 'Impression_y', 'Indication', 'Slice_y', 'SUV_y', 'Previous Sentence',
                 'Following Sentence', 'annotation', 'anatomy', 'anatomy_available'])
    df_dropped.rename(columns={'Slice_x': 'Slice'}, inplace=True)
    df_dropped.rename(columns={'SUV_x': 'SUV'}, inplace=True)

    df_dropped = df.rename(columns={"Extracted Sentences": "sentence"})

    print(f"final df length: {len(df_dropped)}")
    return df_dropped


def find_anatomical_entities(sent, f1radgraph):
    if isinstance(sent, float):
        sent = ['None']
    if isinstance(sent, str):
        sent = [sent]

    annotation = f1radgraph(hyps=sent)
    entities = annotation[0]['entities']
    anatomy = []

    for entity in entities.values():
        token = entity['tokens']
        label = entity['label']
        relation = entity['relations']
        if 'ANAT' in label:
            anatomy.append(token)

    return annotation, anatomy


def get_anatomical_dataframe(df):
    num_patients = 442
    #data_files = './uw_pet_lymphoma_next_and_previous_sentence.xlsx'
    #df = pd.read_excel(data_files)
    num_patients = len(df)

    #radgraph.
    #f1radgraph = GenRadGraph(reward_level="partial")
    f1radgraph = GenRadGraph(reward_level="partial")

    anatomy_list = []
    annotation_list = []
    anatomy_available = []

    for ii in tqdm(range(num_patients)):
        sent = df['Extracted Sentences'][ii]
        annotation, anatomy = find_anatomical_entities(sent, f1radgraph)
        annotation_list.append(annotation)
        anatomy_list.append(anatomy)
        if len(anatomy) == 0:
            anatomy_available.append(0)
        else:
            anatomy_available.append(1)

    df['annotation'] = annotation_list
    df['anatomy'] = anatomy_list
    df['anatomy_available'] = anatomy_available
    df.to_excel('uw_pet_lymphoma_next_and_previous_sentence_annotated.xlsx', index=False)

