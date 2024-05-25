import pandas as pd
import numpy as np
import os
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
    # Regular expression patterns for 'indication: <x>'
    indication_pattern = r'Indication:\s*(\d+)'
    # Search for matches in the input string with case-insensitive flag
    indication_match = re.search(indication_pattern, s, re.IGNORECASE)
    try:
        # Extract values if matches are found; otherwise, return None
        indication_value = int(indication_match.group(1)) if indication_match else None
    except ValueError as e:
        print(f"Error converting SUV value to float: {e}")
        indication_value = None

    return indication_value

def llm_remove_non_anatomical_sent(df):

    print("hi")
    #df_path = "/UserData/Zach_Analysis/suv_slice_text/extracted_data_lymphoma_anonymized.xlsx"
    #df_path = "/UserData/Zach_Analysis/suv_slice_text/test_samples_2000.xlsx"
    #df = pandas.read_excel(df_path)


    instruction = """<s> [INST] 
You are a helpful assistant tasked indicating if the sentence contains enough anatomical information to localize it within the human body. You should return a 1 if it contains enough information to localize the described SUV measurement and a 0 if it does not contain enough information to localize the descirbed SUV measurement.
Sentence: Several mildly enlarged scattered small lymph nodes in the right level 2 and 3 cervical nodal regions, largest most prominent being a 1.1 x 0.8 cm mild enlarged right level III cervical node with mild FDG uptake with SUV max 3.0 (PET/CT axial slice 90).
[/INST]
Indication: 1
[INST] 
You are a helpful assistant tasked indicating if the sentence contains enough anatomical information to localize it within the human body. You should return a 1 if it contains enough information to localize the described SUV measurement and a 0 if it does not contain enough information to localize the descirbed SUV measurement.
Sentence: There is intense FDG uptake at this resection bed measuring SUV max of 8.6 (PET/CT axial slice 60), which fuses to an area of ill-defined amorphus soft tissue/scarring.
[/INST]
Indication: 0
[INST] 
You are a helpful assistant tasked indicating if the sentence contains enough anatomical information to localize it within the human body. You should return a 1 if it contains enough information to localize the described SUV measurement and a 0 if it does not contain enough information to localize the descirbed SUV measurement.
Sentence: There is also increasing FDG activity associated with a small cutaneous/subcutaneous nodularity of the anterior right chest overlying the sternum, now with max SUV of 3.6 (axial slice 134) compared 1.9 previously.
[/INST]
Indication: 1
[INST] 
You are a helpful assistant tasked indicating if the sentence contains enough anatomical information to localize it within the human body. You should return a 1 if it contains enough information to localize the described SUV measurement and a 0 if it does not contain enough information to localize the descirbed SUV measurement.
Sentence: This measures SUV max of 6.6 and the soft tissue approximates 3.3 cm x 2.5 cm in maximal transverse dimensions (PET/CT axial slice 214).
[/INST]
Indication: 0
[INST] 
You are a helpful assistant tasked indicating if the sentence contains enough anatomical information to localize it within the human body. You should return a 1 if it contains enough information to localize the described SUV measurement and a 0 if it does not contain enough information to localize the descirbed SUV measurement.
Sentence: There is a small focus of moderately increased FDG uptake within the lower right thyroid lobe (slice 67 SUV max 5.4) with no CT correlate.
[/INST]
Indication: 1
</s>
[INST]
You are a helpful assistant tasked indicating if the sentence contains enough anatomical information to localize it within the human body. You should return a 1 if it contains enough information to localize the described SUV measurement and a 0 if it does not contain enough information to localize the descirbed SUV measurement.
Sentence: """


    #models = ['llama2-7b-instruct_v2', 'mistral-7b-instruct', 'mixstral-8x7b-instruct']
    #models = ['dolphin-instruct']
    models = ['dolphin-instruct', 'mistral-7b-instruct', 'mixstral-8x7b-instruct']
    for model in models:
        ai_indication = []
        accession = []
        for index, row in df.iterrows():
            print(f"index: {index}")
            #if index == 1000:
            #    break
            #if index < 1000:
            #if index > 2500:
            #if index > 10:
            #    break
            #    break
            #    continue
            #accession.append(row["Accession Number"])
            accession.append(row["Petlymph"])
            sentence = row["sentence"]
            total_prompt = instruction + sentence + "\n[/INST]"
            #generated = ollama.generate(model='mixtral-instuct', prompt=total_prompt)
            #response = generate_with_timeout(model, total_prompt)

            #if response == None:
            #    print("none response")
            #    response = ""
            #response = generate_with_timeout(model, total_prompt, timeout=60)
            #response = process_single_prompt(model, total_prompt)
            generated = ollama.generate(model=model, prompt = total_prompt)
            response = generated["response"]
            #print(response)
            #print(f"respone length: {len(response)}")
            #print(type(response))
            #print(response)
            #extraction = extract_sentences_and_numbers(response, "slice", "suv")
            #print(extraction, flush=True)
            #sent = extraction[0]
            #slice_val = extraction[1]
            #suv_val = extraction[2]
            indication_val = extract_values(response)
            #print(f"slice: {slice_val} suv: {suv_val}")
            #print(f"slice_val: {slice_val} and suv_val: {suv_val}")
            if indication_val is not None:
                ai_indication.append(indication_val)
            else:
                ai_indication.append(0)



        # Convert lists to a DataFrame
        new_df = pd.DataFrame({model + 'AI_Indication': ai_indication})
        if "df_suv_slice" in locals():
            # If df exists, add new columns
            df_suv_slice[new_df.columns[0]] = new_df[new_df.columns[0]]
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

