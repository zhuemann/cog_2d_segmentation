import numpy as np
import os
import pandas
import regex as re
import sys
import pandas as pd


def sent_with_slice_suv(word1_pattern, word2_pattern, sentence):
    # Search for both patterns in the sentence
    word1_match = word1_pattern.search(sentence)
    word2_match = word2_pattern.search(sentence)
    # If both patterns are found, extract the numbers and add them to the results
    if word1_match and word2_match:
        word1_num = word1_match.group(1)
        word2_num = word2_match.group(1)
        return word1_num, word2_num
    else:
        return None, None


def detect_period_followed_by_char(text):
    # Regular expression pattern to find a period followed by a non-decimal, non-space, non-newline character
    pattern = r'\.(?![\s\d.])'

    # Search for the pattern in the text
    matches = re.finditer(pattern, text)

    # Check for matches not followed by spaces or newlines
    for match in matches:
        # Get the index of the character after the period
        index = match.end()
        # Check if the character is not a space or newline
        if index < len(text) and text[index] not in ' \n),':
            if "e.g." in text:
                return False
            return True

    return False

def add_space_after_period(text, keyword):
    # Build a dynamic pattern based on the keyword provided
    pattern = r'\.' + re.escape(keyword)

    # Use re.sub to replace the found pattern with ". {keyword}" (note the space after the period)
    updated_text = re.sub(pattern, '. ' + keyword, text)

    return updated_text


def format_periods(text):
    # Regular expression pattern to identify periods that are not followed by a number, space, new line, ')', or 'e.g.'
    # Negative lookahead is used to specify conditions where replacement should not occur

    if "i.e." in text or "e.g" in text:
        return text

    #pattern = r'\.(?!(\d|\s|\n|\)))'
    pattern = r'\.(?!(\d|\s|\n|\)|e\.g\.|i\.e\.|/))'

    # Use re.sub to replace the identified pattern with ". " (a period followed by a space)
    updated_text = re.sub(pattern, '. ', text)

    return updated_text

def extract_sentences_and_numbers(text, word1, word2, double_colon, more_colon):
    # Compile patterns to find the keywords followed by numbers
    # This regex looks for the word followed by optional spaces, possibly some words, and then a number
    word1_pattern = re.compile(re.escape(word1) + r'\s*(?:\w+\s*)*?(\d+(?:\.\d+)?)', re.IGNORECASE)
    word2_pattern = re.compile(re.escape(word2) + r'\s*(?:\w+\s*)*?(\d+(?:\.\d+)?)', re.IGNORECASE)

    # fix templated errors in text and add periods that physican didn't type
    text = add_space_after_period(text, "Physiologic")
    text = format_periods(text)

    # Use regex to find sentences in the text. Assuming sentences end with '.', '!', or '?'
    sentences = re.split(r'(?<=[.!?*\n])\s+', text)  # need to slit on new line, *
    # Initialize a list to hold the results

    # report_findings = text.replace("_x000D_", " ").replace("\n", " ").replace("#", " ")
    # report_findings = report_findings.replace("...", " ")
    # report_findings = re.sub(r"\.{2,}", ".", report_findings)
    # report_findings = re.sub(r'\s+', ' ', report_findings).strip()
    results = []

    # Check each sentence for the patterns
    for i, sentence in enumerate(sentences):

        #if sentence == "No FDG avid lung nodules are noted.Physiologic FDG uptake is present within the myocardium.":
            #print("skipped")
        #    continue
        if "Background liver metabolic activity" in sentence:
            continue
        if "Background mediastinal blood pool metabolic activity" in sentence:
            continue
        #if find_text_after_pattern(sentence, pattern=".Physiologic") != None:
        #    sentence = find_text_after_pattern(sentence, pattern=".Physiologic")
        #if detect_period_followed_by_char(sentence):
        #    print(sentence)
        #    print(format_periods(sentence))

        #if ".There" in sentence:
        #    print(sentence)
        if ";" in sentence:
            # since we found a ; in the sentence we want to check if there is a slice and suv on both sides of the ;
            splits = re.split(f';', sentence)
            # need to check each split with for loop
            if len(splits) > 2:
                #print(i)
                #print(text)
                #print(sentence)
                #print(splits)
                #print("more colons")
                #more_colon += 1
                #print(sentence)
                #print(splits)
                #print(len(splits))
                for i in range(0,len(splits)):
                    word1_num, word2_num = sent_with_slice_suv(word1_pattern, word2_pattern,splits[i])
                    if word1_num and word2_num:
                        results.append([splits[i], word1_num, word2_num])
                        #print("added: ")
                        #print(splits[i])
                break
            # there are two in the split so only split if both before and after the split have suv and
            else:
                first_half = splits[0]
                second_half = splits[1]
                first_half_word1_num, first_half_word2_num = sent_with_slice_suv(word1_pattern, word2_pattern, first_half)
                second_half_word1_num, second_half_word2_num = sent_with_slice_suv(word1_pattern, word2_pattern,
                                                                                   second_half)
                if first_half_word1_num and first_half_word2_num and second_half_word1_num and second_half_word2_num: # both halfs have suv and slice so add separately

                    double_colon += 1
                    results.append([first_half, first_half_word1_num, first_half_word2_num])
                    results.append([second_half, second_half_word1_num, second_half_word2_num])

                else: # there is ;  but both halfs don't have suv and slice so keep them together
                    word1_num, word2_num = sent_with_slice_suv(word1_pattern, word2_pattern, sentence)
                    if word1_num and word2_num:
                        results.append([sentence, word1_num, word2_num])


        else: # there is no ; so we just add them if if there is suv and slice
            word1_num, word2_num = sent_with_slice_suv(word1_pattern, word2_pattern, sentence)
            # If both patterns are found, extract the numbers and add them to the results
            if word1_num and word2_num:
                results.append([sentence, word1_num, word2_num])

    return results, double_colon, more_colon

def split_sentences():

    df_path = "Z:/Zach_Analysis/text_data/indications.xlsx"
    #df_path = "/UserData/Zach_Analysis/text_data/indications.xlsx"
    df = pandas.read_excel(df_path, sheet_name = "lymphoma_uw_only")


    # Assuming df is your original DataFrame
    sum_test = 0
    double_colon, more_colon = 0, 0
    print_examples = False
    collected_data = []  # List to collect data
    is_uw_lymphoma_data = False
    for index, row in df.iterrows():

        if is_uw_lymphoma_data:
            acession_num = row.iloc[0]
            report = row.iloc[2]
            impressions = row.iloc[3]
            indications = row.iloc[4]
            text = row.iloc[2]
        else:
            petlymph = row["research_id"]
            # acession_num = row["Accession Number"]
            report = row["findings"]
            # impressions = row["impressions"]
            impressions = row["impression"]
            # indications = row["indications"]
            report_findings = row["findings"]  # .replace("_x000D_", " ").replace("\n", " ").replace("#", " ")
            text = row["findings"]

        # if index > 10000:
        #    break
        #print(f"index: {index}")
        if isinstance(report, str):

            report_findings = text.replace("_x000D_", " ").replace("\n", " ").replace("#", " ")
            # report_findings = report_findings.replace(";", ". ") # trying getter way of handling ;
            report_findings = report_findings.replace("   ", ". ")
            report_findings = report_findings.replace("...", ". ")
            report_findings = re.sub(r"\.{2,}", ". ", report_findings)
            report_findings = re.sub(r'\s+', ' ', report_findings).strip()

            sent, double_colon, more_colon = extract_sentences_and_numbers(report_findings, "slice", "suv",
                                                                           double_colon, more_colon)

            if len(sent) > 0:
                if print_examples:
                    print(f"Extracted sentences: {sent}")
                    print(f"Whole report: {report}\n")

                for sentence in sent:
                    sentence_example = sentence[0]
                    slice_example = sentence[1]
                    suv_example = sentence[2]
                    collected_data.append({
                        # 'Accession Number': acession_num,
                        'Petlymph': petlymph,
                        'Findings': report,
                        'Impression': impressions,
                        # 'Indication': indications,
                        'Extracted Sentences': sentence_example,
                        'Slice': slice_example,
                        'SUV': suv_example
                    })
                sum_test += len(sent)

    print(f"number of examples: {sum_test}")
    print(f"double colon: {double_colon}")
    print(f"more colon: {more_colon}")

    # Convert list to DataFrame and save
    output_df = pd.DataFrame(collected_data)
    return output_df
    # output_df.to_excel('extracted_data_all_pet_anonymized.xlsx', index=False)