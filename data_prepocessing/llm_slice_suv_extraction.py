import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas
import regex as re
import sys

import ollama


def extract_sentences_and_numbers(text, word1, word2):
    # Compile patterns to find the keywords followed by numbers
    # This regex looks for the word followed by optional spaces, possibly some words, and then a number
    word1_pattern = re.compile(re.escape(word1) + r'\s*(?:\w+\s*)*?(\d+(?:\.\d+)?)', re.IGNORECASE)
    word2_pattern = re.compile(re.escape(word2) + r'\s*(?:\w+\s*)*?(\d+(?:\.\d+)?)', re.IGNORECASE)

    # Use regex to find sentences in the text. Assuming sentences end with '.', '!', or '?'
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Initialize a list to hold the results
    results = []

    # Check each sentence for the patterns
    for i, sentence in enumerate(sentences):
        # Search for both patterns in the sentence
        word1_match = word1_pattern.search(sentence)
        word2_match = word2_pattern.search(sentence)
        # If both patterns are found, extract the numbers and add them to the results
        if word1_match and word2_match:
            word1_num = word1_match.group(1)
            word2_num = word2_match.group(1)
            results.append([sentence, word1_num, word2_num])

    return results

def extract_values(s):
    # Regular expression patterns for 'slice: <x>' and 'SUV: <y>'
    slice_pattern = r'slice:\s*(\d+)'
    suv_pattern = r'SUV:\s*([\d.]+)'
    # Search for matches in the input string with case-insensitive flag
    slice_match = re.search(slice_pattern, s, re.IGNORECASE)
    suv_match = re.search(suv_pattern, s, re.IGNORECASE)
    try:
        # Extract values if matches are found; otherwise, return None
        slice_value = int(slice_match.group(1)) if slice_match else None
        suv_value = float(suv_match.group(1)) if suv_match else None
    except ValueError as e:
        print(f"Error converting SUV value to float: {e}")
        suv_value = None
        slice_value = None

    return slice_value, suv_value

def llm_slice_suv_extraction(df):

    print("hi")
    #df_path = "/UserData/Zach_Analysis/suv_slice_text/extracted_data_lymphoma_anonymized.xlsx"
    #df_path = "/UserData/Zach_Analysis/suv_slice_text/test_samples_2000.xlsx"
    #df = pandas.read_excel(df_path)
    instruction = '''<s> [INST] 
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence: Several mildly enlarged scattered small lymph nodes in the right level 2 and 3 cervical nodal regions, largest most prominent being a 1.1 x 0.8 cm mild enlarged right level III cervical node with mild FDG uptake with SUV max 3.0 (PET/CT axial slice 90).
    [/INST]
    Slice: 90
    SUV: 3.0
    [INST]
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence: Head/Neck: There is asymmetric intense FDG uptake within the left palatine tonsillar bed (slice 57) that fuses to soft tissue fullness and shows SUV max of 7.3 compared to 4.0 on the contralateral right side. 
    [/INST]
    Slice: 57
    SUV: 7.3
    [INST]
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence: There is significant interval decrease of the metabolic activity fusing to the lateral right scapula located inferior to the right glenoid (currently SUV max is 3.3 on slice 94 compared to 6.3 on the prior PET/CT from October 28, 2015).
    [/INST]
    Slice: 94
    SUV: 3.3
    [INST]
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence For example the largest and most hypermetabolic left axillary lymph node is noted on slice 104 that measures 2.1 x 1.1 cm and shows SUV max of 5.5. 
    [/INST]
    Slice: 104
    SUV: 5.5
    [INST]
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence: Chest: There is interval reduction of hypermetabolic activity of the left axillary lymph nodes, the most FDG avid node (slice 95) measuring an SUV Max of 1.7 today (previously 2.8), which is similar to mediastinal background blood pool activity with an SUV Max of 1.9.
    [/INST]
    Slice: 95
    SUV: 1.7
    [INST]
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence: This again remains difficult to measure, but measures approximately 11.6 cm in maximum transverse dimension as compared to 23 cm previously and demonstrates heterogeneous mild FDG uptake with an SUV max of 6.0 (slice location 218).
    [/INST]
    Slice: 218
    SUV: 6.0
    [INST]
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence: For example: a right upper paratracheal node now measures SUV max of 1.5 and 0.8 cm in short axis diameter (PET/CT axial slice 95) compared to prior SUV max of 1.7 and 1.1 cm in short axis diameter.
    [/INST]
    Slice: 95
    SUV: 1.5
    </s>
    [INST]
    You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
    Sentence: '''

    instruction_v2 = """<s> [INST] 
You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
Sentence: Several mildly enlarged scattered small lymph nodes in the right level 2 and 3 cervical nodal regions, largest most prominent being a 1.1 x 0.8 cm mild enlarged right level III cervical node with mild FDG uptake with SUV max 3.0 (PET/CT axial slice 90).
[/INST]
Slice: 90
SUV: 3.0
</s>
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the slice number and its associated suv value for the current scan.
Sentence: """

    models = ['llama2-7b_smaller-instruct', 'mistral-7b-instruct', 'mixstral-8x7b-instruct']
    for model in models:
        ai_slice = []
        ai_suv = []
        accession = []
        for index, row in df.iterrows():
            print(f"index: {index}")
            #if index == 1000:
            #    break
            #if index < 1000:
            #    continue
            #accession.append(row["Accession Number"])
            accession.append(row["Petlymph"])
            sentence = row["Extracted Sentences"]
            total_prompt = instruction_v2 + sentence + "\n[/INST]"
            #generated = ollama.generate(model='mixtral-instuct', prompt=total_prompt)


            generated = ollama.generate(model=model, prompt = total_prompt)
            response = generated["response"]
            print(response)
            #print(type(response))
            #sent = extract_sentences_and_numbers(response, "slice", "suv")
            slice_val, suv_val = extract_values(response)
            #print(f"slice_val: {slice_val} and suv_val: {suv_val}")
            if slice_val is not None and suv_val is not None:
                ai_slice.append(slice_val)
                ai_suv.append(suv_val)
            else:
                ai_slice.append(0)
                ai_suv.append(0)


        # Convert lists to a DataFrame
        new_df = pd.DataFrame({model + 'AI_Slice': ai_slice, model + 'AI_SUV': ai_suv})
        if "df_suv_slice" in locals():
            # If df exists, add new columns
            df_suv_slice[new_df.columns[0]] = new_df[new_df.columns[0]]
            df_suv_slice[new_df.columns[1]] = new_df[new_df.columns[1]]
        else:
            # If df does not exist, create it from new_df
            df_suv_slice = new_df.copy()

    # Create an Excel writer object using openpyxl
    #writer = pd.ExcelWriter('ai_mistral_predictions_uw_lymphoma_anonymized.xlsx', engine='openpyxl')

    # Write the DataFrame to the Excel file
    #df.to_excel(writer, index=False, sheet_name='Predictions')
    df_to_return = pd.concat([df, df_suv_slice], axis=1)
    return df_to_return
    # Save the Excel file
    #writer.save()
    #return df_suv_slice
    #print("Excel file 'ai_predictions.xlsx' has been created.")

