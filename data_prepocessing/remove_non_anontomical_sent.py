import pandas as pd

import os
from radgraph import GenRadGraph
#from radgraph import GenRadGraph
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import re
import numpy as np


def remove_terms(sentence, terms):
    # Create a pattern that matches any of the terms in the list
    # The pattern uses \b for word boundary to ensure it matches whole words only
    pattern = r'\b(' + '|'.join(map(re.escape, terms)) + r')\b'

    # Use re.sub to replace these terms with an empty string
    cleaned_sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)

    # Strip extra spaces that might be left after removal
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()

    return cleaned_sentence


# List of terms to remove
terms_to_remove = [
    "cm", "omental", "soft tissue", "right", "left", "adjacent",
    "transverse", "lateral", "component", "posterior", "subcutaneous", "nodal",
    "inferior", "axis", "posterior", "medial", "size", "FDG", "Right", "Left"
]

def contains_key_terms(sentence):
    # This regex pattern looks for words that start with 'para', or contain 'level' or 'ap window'
    pattern = r'\bpara\w*|\blevel\b|ap window'

    # Search for the pattern in the sentence, case insensitive
    if re.search(pattern, sentence, re.IGNORECASE):
        return True
    else:
        return False

def remove_non_anontomical_sent(df):

    # throw out the sentence if multiple label points were extracted
    #df_filtered = df.drop_duplicates(subset=['Petlymph', 'Extracted Sentences'], keep=False)
    #print(len(df_filtered))
    df_filtered = df

    # Create an enumeration column for each label per "Petlymph" ID
    df_filtered['Label_Number'] = df_filtered.groupby('Petlymph').cumcount() + 1

    # Create the new column by combining "Petlymph" with the label enumeration
    df_filtered['Label_Name'] = df_filtered['Petlymph'] + '_label_' + df_filtered['Label_Number'].astype(str)

    # Drop the temporary 'Label_Number' column if you don't need it
    df_filtered.drop('Label_Number', axis=1, inplace=True)

    #df_path = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v14/uw_pet_lymphoma_next_and_previous_sentence_annotated.xlsx"
    df_path = "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v1/uw_pet_annotated.xlsx"
    df_anotomical_info = pd.read_excel(df_path)

    # Merge the two DataFrames on 'Petlymph' and 'Extracted sentences'
    merged_df = pd.merge(df_filtered, df_anotomical_info, on=['Extracted Sentences'])
    filtered_df = merged_df[merged_df['anatomy_available'] != 0]    # maybe don't drop is contains AP window or level
    #print(filtered_df)
    #print(filtered_df.columns.tolist())
    #df_dropped = filtered_df.drop(
    #columns=['Accession Number', 'Report', 'Impression_y', 'Indication', 'Slice_y', 'SUV_y', 'Previous Sentence',
    #             'Following Sentence', 'annotation', 'anatomy', 'anatomy_available'])

    #df_dropped = filtered_df.drop(
    #    columns=['Unnamed: 0', 'Petlymph_y', 'Findings_y', 'Impression_y', 'Slice_y', 'SUV_y',
    #              'Previous Sentence', 'Following Sentence', 'annotation', 'anatomy', 'anatomy_available',
    #              'anatomy_available_previous', 'anatomy_available_next'])
    df_dropped = filtered_df.drop(
        columns=['Unnamed: 0', 'Petlymph_y', 'Findings_y', 'Impression_y', 'Slice_y', 'SUV_y',
                 'annotation', 'anatomy', 'anatomy_available'])

    df_dropped.rename(columns={'Slice_x': 'Slice'}, inplace=True)
    df_dropped.rename(columns={'SUV_x': 'SUV'}, inplace=True)
    df_dropped.rename(columns={'Petlymph_x': 'Petlymph'}, inplace=True)
    df_dropped.rename(columns={'Findings_x': 'Findings'}, inplace=True)
    df_dropped.rename(columns={'Impression_x': 'Impression'}, inplace=True)

    #df_dropped.rename(columns={"Extracted Sentences": "sentence"})

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


# Function to find the surrounding sentences
def find_surrounding_sentences(report, extracted_sentence):
    # clean up the text a bit same as previous code
    report_findings = report.replace("_x000D_", " ").replace("\n", " ").replace("#", " ")
    report_findings = report_findings.replace("...", " ")
    report_findings = re.sub(r"\.{2,}", ".", report_findings)
    report = re.sub(r'\s+', ' ', report_findings).strip()

    sentences = re.split(r'(?<=[.!?])\s+', report)
    previous_sentence = ''
    following_sentence = ''

    for i, sentence in enumerate(sentences):
        if extracted_sentence in sentence:
            if i > 0:
                previous_sentence = sentences[i - 1].strip()
            if i < len(sentences) - 1:
                following_sentence = sentences[i + 1].strip()
            break

    return previous_sentence, following_sentence



#def get_anatomical_dataframe(df):


if __name__ == '__main__':

    #save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v3/"
    #save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v14/"
    save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v1/"

    save_base_final = "/UserData/Zach_Analysis/petlymph_image_data/"
    df = pd.read_excel(save_base + "remove_multiple_suv_and_slice_2_test.xlsx")
    #num_patients = 442
    #data_files = './uw_pet_lymphoma_next_and_previous_sentence.xlsx'
    #df = pd.read_excel(data_files)
    num_patients = len(df)

    # Apply the function to each row in the DataFrame
    #df['Previous Sentence'], df['Following Sentence'] = zip(
    #    *df.apply(lambda row: find_surrounding_sentences(row['Findings'], row['Extracted Sentences']), axis=1))

    # Save the modified DataFrame back to an Excel file
    #df.to_excel('modified_excel_file.xlsx', index=False)


    #radgraph.
    #f1radgraph = GenRadGraph(reward_level="partial")
    f1radgraph = GenRadGraph(reward_level="partial")

    anatomy_list = []
    annotation_list = []
    anatomy_available = []

    #anatomy_available_previous = []
    #anatomy_available_next = []

    for ii in tqdm(range(num_patients)):
        sent = df['Extracted Sentences'][ii]
        annotation, anatomy = find_anatomical_entities(sent, f1radgraph)
        annotation_list.append(annotation)
        anatomy_list.append(anatomy)
        if len(anatomy) == 0:
            anatomy_available.append(0)
        else:
            anatomy_available.append(1)

        """
        previous = df['Previous Sentence'][ii]

        try:
            annotation, anatomy = find_anatomical_entities(previous, f1radgraph)
        except:
            anatomy = []
        if len(anatomy) == 0:
            anatomy_available_previous.append(0)
        else:
            anatomy_available_previous.append(1)

        next = df['Following Sentence'][ii]

        try:
            annotation, anatomy = find_anatomical_entities(next, f1radgraph)
        except:
            anatomy = []
        if len(anatomy) == 0:
            anatomy_available_next.append(0)
        else:
            anatomy_available_next.append(1)
        """


    df['annotation'] = annotation_list
    df['anatomy'] = anatomy_list
    df['anatomy_available'] = anatomy_available
    #df['anatomy_available_previous'] = anatomy_available_previous
    #df['anatomy_available_next'] = anatomy_available_next

    df.to_excel(save_base + 'uw_pet_annotated_test_dummy_sentences.xlsx', index=False)
    #return df
