

import pandas as pd
import regex as re

def extract_sentences_and_numbers(text, word1, word2):
    # Compile patterns to find the keywords followed by numbers
    # This regex looks for the word followed by optional spaces, possibly some words, and then a number
    word1_pattern = re.compile(re.escape(word1) + r'\s*(?:\w+\s*)*?(\d+(?:\.\d+)?)', re.IGNORECASE)
    word2_pattern = re.compile(re.escape(word2) + r'\s*(?:\w+\s*)*?(\d+(?:\.\d+)?)', re.IGNORECASE)

    # Use regex to find sentences in the text. Assuming sentences end with '.', '!', or '?'
    sentences = re.split(r'(?<=[.!?])\s+', text)

    report_findings = text.replace("_x000D_", " ").replace("\n", " ").replace("#", " ")
    report_findings = report_findings.replace("...", " ")
    report_findings = re.sub(r"\.{2,}", ".", report_findings)
    report_findings = re.sub(r'\s+', ' ', report_findings).strip()


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

def extracting_slice_suv_values(df):

    #df_path = "Z:/Zach_Analysis/text_data/indications.xlsx"
    #df = pd.read_excel(df_path, sheet_name="lymphoma_uw_only")

    sum_test = 0
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

        print(f"Extracting slice and suv index: {index}")
        if isinstance(report, str):

            sent = extract_sentences_and_numbers(report_findings, "slice", "suv")

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

def make_better_name_later(df_location):

    slice_suv_df = extracting_slice_suv_values()
