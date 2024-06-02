import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas
import regex as re
import sys

import ollama


import signal

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutError

# Function to setup the signal and execute your code
def generate_with_timeout_v1(model, prompt, timeout=60):
    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule the alarm
    signal.alarm(timeout)
    try:
        # Attempt to run the main function
        generated = ollama.generate(model=model, prompt=prompt)
        response = generated["response"]
        return response
    except TimeoutError:
        print("Generation timed out after 60 seconds.")
        return ""
    finally:
        # Disable the alarm
        signal.alarm(0)

import concurrent.futures
import time

def generate_with_timeout_v2(model, prompt, timeout=60):
    # Define the function that will be called asynchronously
    def generate_response():
        generated = ollama.generate(model=model, prompt=prompt)
        response = generated["response"]
        return response

    # Use ThreadPoolExecutor to manage asynchronous execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the function to the executor
        future = executor.submit(generate_response)
        try:
            # Wait for the function to complete or timeout
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("Generation timed out after 60 seconds.")
            return ""  # or handle the timeout differently, depending on your needs

def generate_response(model, prompt):
    # Simulated function to represent the call to your generative model
    #print("Generating for prompt:", prompt)
    generated = ollama.generate(model=model, prompt=prompt)
    response = generated["response"]
    return response

def process_single_prompt(model, prompt, timeout=60):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the generation task for a single prompt
        future = executor.submit(generate_response, model, prompt)
        try:
            # Try to get the result within the timeout
            result = future.result(timeout=timeout)
            #print("Result for", prompt, ":", result)
            return result
        except concurrent.futures.TimeoutError:
            #print("Timeout occurred for prompt:", prompt)
            return ""
        except Exception as e:
            #print("Error occurred for prompt:", prompt, "Error:", str(e))
            return ""

def process_prompts_one_by_one(model, prompts):
    for prompt in prompts:
        process_single_prompt(model, prompt)


def generate_with_timeout(model, prompt, timeout=60):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Define a task to run the generation
            future = executor.submit(ollama.generate, model=model, prompt=prompt)
            # Wait for the result with the specified timeout
            result = future.result(timeout=timeout)
            return result["response"]
    except concurrent.futures.TimeoutError:
        print("Timeout occurred - generation took too long")
        return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None

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
    slice_pattern = r'Measurement number:\s*(\d+)'
    # Search for matches in the input string with case-insensitive flag
    slice_match = re.search(slice_pattern, s, re.IGNORECASE)
    try:
        # Extract values if matches are found; otherwise, return None
        slice_value = int(slice_match.group(1)) if slice_match else None
    except ValueError as e:
        print(f"Error converting SUV value to float: {e}")

    return slice_value


def classify_findings(input_string):
    # Normalize the string to lower case for case insensitive comparison
    normalized_string = input_string.lower()

    # Check for presence of both or neither keywords
    has_single = "single finding" in normalized_string
    has_multiple = "multiple finding" in normalized_string

    if has_single and has_multiple:
        return None
    elif has_single:
        return 0
    elif has_multiple:
        return 1
    else:
        return None
def llm_remove_multiple_descriptions(df):

    print("hi")

    instruction_v3 = '''<s> [INST] 
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: Several mildly enlarged scattered small lymph nodes in the right level 2 and 3 cervical nodal regions, largest most prominent being a 1.1 x 0.8 cm mild enlarged right level III cervical node with mild FDG uptake with SUV max 3.0 (PET/CT axial slice 90).
[/INST]
Slice: 90
SUV: 3.0
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: Head/Neck: There is asymmetric intense FDG uptake within the left palatine tonsillar bed (slice 57) that fuses to soft tissue fullness and shows SUV max of 7.3 compared to 4.0 on the contralateral right side. 
[/INST]
Slice: 57
SUV: 7.3
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0.
Sentence: There is significant interval decrease of the metabolic activity fusing to the lateral right scapula located inferior to the right glenoid (currently SUV max is 3.3 on slice 94 compared to 6.3 on the prior PET/CT.
[/INST]
Slice: 94
SUV: 3.3
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: For example the largest and most hypermetabolic left axillary lymph node is noted on slice 104 that measures 2.1 x 1.1 cm and shows SUV max of 5.5. 
[/INST]
Slice: 104
SUV: 5.5
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: FINDINGS: The patient is imaged in the radiotherapy position, and in…the region of the cervix there is a focal region of moderately intense…tracer intense uptake (SUV max 13) on the left and is best seen in slice… 112 of the coronal IRCTAC and slice 42 in the axial WB IRCTAC.
 [/INST]
Slice: 42
SUV: 13.0
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: For example 6 mm and 8mm left retroperitoneal lymph nodes show SUV max of 2.7 and 3.5 respectively (series 2 slice 170 And 176 respectively). [/INST]
Slice: 0
SUV: 0
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: Chest: There is interval reduction of hypermetabolic activity of the left axillary lymph nodes, the most FDG avid node (slice 95) measuring an SUV Max of 1.7 today (previously 2.8), which is similar to mediastinal background blood pool activity with an SUV Max of 1.9.
[/INST]
Slice: 95
SUV: 1.7
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: This again remains difficult to measure, but measures approximately 11.6 cm in maximum transverse dimension as compared to 23 cm previously and demonstrates heterogeneous mild FDG uptake with an SUV max of 6.0 (slice location 218).
[/INST]
Slice: 218
SUV: 6.0
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0. 
Sentence: For example: a right upper paratracheal node now measures SUV max of 1.5 and 0.8 cm in short axis diameter (PET/CT axial slice 95) compared to prior SUV max of 1.7 and 1.1 cm in short axis diameter.
[/INST]
Slice: 95
SUV: 1.5
</s>
[INST]
You are a helpful assistant tasked with extracting values from a sentence. You should return the axial slice number and its associated suv value for the current scan. If there are multiple slice numbers return 0.
Sentence: '''

    instruction_v4 = '''<s> [INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. 
Sentence: In addition, a 6 mm hypermetabolic soft tissue nodule/mass is identified within the lateral portion of the left breast (slice 136), SUV max 2.5. 
[/INST]
Measurement number: 1
[INST] 
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. 
Sentence: Interval increase in metabolic activity of nodal mass adjacent to the takeoff of superior mesenteric artery (SUVmax 11.5 versus 8.7 previously as well as multiple other para-aortic/retroperitoneal lymph nodes (aortic lymph node SUVmax 10.0 versus 7.4 previously) the common iliac lymph node (SUVmax 5.6 versus 5 previously), right parasacral lymph node (CT slice location 222; SUVmax 3.8 versus 2.3 on 5/6/2014 versus 4.9 on 2/5/2014). [/INST]
Measurement number: 4
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. 
Sentence: Abdomen/Pelvis:…There is a hypermetabolic 1.6 cm soft tissue mass located at the left lower…quadrant with a SUVmax of 12.6 (Slice 144 Series 2) just inferolateral to…the left kidney. 
[/INST]
Measurement number: 1
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence: A right level IB lymph node (slice 41) has slightly increased in size from 8 x 6 mm (SUV max 4.7) on prior examination to 10 x 8 mm (SUV max 4.5) on the current examination. 
[/INST]
Measurement number: 1
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence: For example, the includes a 1.9 cm aortic bifurcation lymph node that shows SUVmax of 46.9 ( CT series 2, slice 155) and a right external iliac 1.3 cm lymph node that shows SUVmax of 41.0 (CT series 2, slice 168.) No FDG avid mesenteric lymph nodes region are noted. 
[/INST]
Measurement number: 2
</s>
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence:'''

    instruction_v5 = '''<s> [INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. 
Sentence: In addition, a 6 mm hypermetabolic soft tissue nodule/mass is identified within the lateral portion of the left breast (slice 136), SUV max 2.5. 
[/INST]
Measurement number: 1
[INST] 
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. 
Sentence: Interval increase in metabolic activity of nodal mass adjacent to the takeoff of superior mesenteric artery (SUVmax 11.5 versus 8.7 previously as well as multiple other para-aortic/retroperitoneal lymph nodes (aortic lymph node SUVmax 10.0 versus 7.4 previously) the common iliac lymph node (SUVmax 5.6 versus 5 previously), right parasacral lymph node (CT slice location 222; SUVmax 3.8 versus 2.3 on 5/6/2014 versus 4.9 on 2/5/2014). [/INST]
Measurement number: 4
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. 
Sentence: Abdomen/Pelvis:…There is a hypermetabolic 1.6 cm soft tissue mass located at the left lower…quadrant with a SUVmax of 12.6 (Slice 144 Series 2) just inferolateral to…the left kidney. 
[/INST]
Measurement number: 1
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence: A right level IB lymph node (slice 41) has slightly increased in size from 8 x 6 mm (SUV max 4.7) on prior examination to 10 x 8 mm (SUV max 4.5) on the current examination. 
[/INST]
Measurement number: 1
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence: Abdomen and pelvis: New hypermetabolic mesenteric lymph node is identified in the left lower abdomen (SUVmax 5.7) measuring 0.8 cm x 1 cm in size (CT slice location minus 595.14). 
[/INST]
Measurement number: 1
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence: Mild uptake (SUVmax 2.1, CT slice location 72) is noted in a small anterior mediastinal lymph node anterior to the left brachiocephalic vein, this is seen to better advantage on contrast-enhanced CT neck dated 4/18/2014.
[/INST]
Measurement number: 1
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence: For example, the includes a 1.9 cm aortic bifurcation lymph node that shows SUVmax of 46.9 ( CT series 2, slice 155) and a right external iliac 1.3 cm lymph node that shows SUVmax of 41.0 (CT series 2, slice 168.) No FDG avid mesenteric lymph nodes region are noted. 
[/INST]
Measurement number: 2
</s>
[INST]
You are a helpful assistant tasked with evaluating sentences from radiologists. You should return the number of distinct noted lesions. A noted lesion is any lesion which is given an anatomical location, and often a measurement like SUVmax. References to previous measurements do not count. Sentence: '''


    instruction_v6 = """instruction_v5 = '''<s> [INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: Sentence: Abdomen and pelvis: New hypermetabolic mesenteric lymph node is identified in the left lower abdomen (SUVmax 5.7) measuring 0.8 cm x 1 cm in size (CT slice location minus 595.14). 
[/INST]
Single Finding
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: Interval increase in metabolic activity of nodal mass adjacent to the takeoff of superior mesenteric artery (SUVmax 11.5 versus 8.7 previously as well as multiple other para-aortic/retroperitoneal lymph nodes (aortic lymph node SUVmax 10.0 versus 7.4 previously) the common iliac lymph node (SUVmax 5.6 versus 5 previously), right parasacral lymph node (CT slice location 222; SUVmax 3.8 versus 2.3 on 5/6/2014 versus 4.9 on 2/5/2014).
 [/INST]
Multiple Findings
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence For example, the includes a 1.9 cm aortic bifurcation lymph node that shows SUVmax of 46.9 ( CT series 2, slice 155) and a right external iliac 1.3 cm lymph node that shows SUVmax of 41.0 (CT series 2, slice 168.) No FDG avid mesenteric lymph nodes region are noted. 
[/INST]
Multiple Findings
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: In addition, a 6 mm hypermetabolic soft tissue nodule/mass is identified within the lateral portion of the left breast (slice 136), SUV max 2.5.
[/INST]
Single Finding
</s>
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: 
"""
    instruction_v7 = """instruction_v5 = '''<s> [INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: Sentence: Abdomen and pelvis: New hypermetabolic mesenteric lymph node is identified in the left lower abdomen (SUVmax 5.7) measuring 0.8 cm x 1 cm in size (CT slice location minus 595.14). 
[/INST]
Single Finding
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: Interval increase in metabolic activity of nodal mass adjacent to the takeoff of superior mesenteric artery (SUVmax 11.5 versus 8.7 previously as well as multiple other para-aortic/retroperitoneal lymph nodes (aortic lymph node SUVmax 10.0 versus 7.4 previously) the common iliac lymph node (SUVmax 5.6 versus 5 previously), right parasacral lymph node (CT slice location 222; SUVmax 3.8 versus 2.3 on versus 4.9 on.
 [/INST]
Multiple Findings
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence For example, the includes a 1.9 cm aortic bifurcation lymph node that shows SUVmax of 46.9 ( CT series 2, slice 155) and a right external iliac 1.3 cm lymph node that shows SUVmax of 41.0 (CT series 2, slice 168.) No FDG avid mesenteric lymph nodes region are noted. 
[/INST]
Multiple Findings
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: In addition, a 6 mm hypermetabolic soft tissue nodule/mass is identified within the lateral portion of the left breast (slice 136), SUV max 2.5.
[/INST]
Single Finding
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: Head and Neck: The metabolic activity noticed in the midline at the base of the tongue is immediately adjacent to the vallecula with an SUVmax of 8.3…(PET AC slice number 205).
 [/INST]
Single Finding
</s>
[INST]
Objective: Classify the sentence below based on whether they describe a single finding or multiple findings from PET/CT scans in cancer patients.
Instructions:
1. Read each sentence carefully.
2. Determine if the sentence describes exactly one finding or multiple findings.
3. Respond with "Single Finding" if the sentence only describes one finding, including its specific details and any changes compared to previous examinations.
4. Respond with "Multiple Findings" if the sentence describes more than one finding, contains multiple data points about different nodes or tissues, or lists findings without connecting details.
Sentence: 

    """
    #models = ['llama2-7b-instruct_v2', 'mistral-7b-instruct', 'mixstral-8x7b-instruct']
    #models = ['dolphin-instruct']
    models = ['dolphin-instruct', 'mistral-7b-instruct', 'mixstral-8x7b-instruct']
    for model in models:
        ai_slice = []
        ai_suv = []
        accession = []
        for index, row in df.iterrows():
            print(f"index: {index}")
            #if index == 1000:
            #    break
            #if index < 1000:
            #if index < 100:
            #    continue
            if index > 50:
                break
            #    continue
            #accession.append(row["Accession Number"])
            accession.append(row["Petlymph"])
            sentence = row["Extracted Sentences"]
            total_prompt = instruction_v6 + sentence + "\n[/INST]"
            #generated = ollama.generate(model='mixtral-instuct', prompt=total_prompt)
            #response = generate_with_timeout(model, total_prompt)

            #if response == None:
            #    print("none response")
            #    response = ""
            #response = generate_with_timeout(model, total_prompt, timeout=60)
            #response = process_single_prompt(model, total_prompt)
            generated = ollama.generate(model=model, prompt = total_prompt)
            response = generated["response"]

            #slice_val = extract_values(response)
            slice_val = classify_findings(response)
            #print(f"slice: {slice_val} suv: {suv_val}")
            #print(f"slice_val: {slice_val} and suv_val: {suv_val}")
            if slice_val is not None:
                ai_slice.append(slice_val)
            else:
                ai_slice.append(None)



        # Convert lists to a DataFrame
        new_df = pd.DataFrame({model + 'AI_Slice': ai_slice})
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

