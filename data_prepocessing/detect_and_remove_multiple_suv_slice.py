import regex as re
import pandas as pd


def llm_sentence_splitting(df):
    #add llm setence splitting
    print(df)


def count_occurrences(sentence):
    # Regex for 'slice' followed by any words and then an integer
    slice_pattern = re.compile(r'\bslice\b(?:\s+\w+)*\s+\d+\b', re.IGNORECASE)
    # Regex for 'SUV' followed by any words and then a decimal
    suv_pattern = re.compile(r'\bSUV\b(?:\s+\w+)*\s+\d+\.\d+\b', re.IGNORECASE)

    # Find all matches and count them
    slice_count = len(slice_pattern.findall(sentence))
    suv_count = len(suv_pattern.findall(sentence))

    return slice_count, suv_count

def contains_previous(sentence):
    # Compile a regex pattern to match "previous" or "previously"
    pattern = re.compile(r'\b(previous|previously)\b', re.IGNORECASE)  # maybe add "compared" "comparison"

    # Search the sentence for the pattern
    return bool(pattern.search(sentence))

def detect_and_remove_multiple_suv_slice(df):
    multiple_extractions = 0
    does_not_contain_previous = 0
    contains_previous_count = 0
    suv_equals_slice = 0
    dropped = 0
    delete_list = []
    for index, row in df.iterrows():
        # print(f"index: {index}")
        sentence = row["Extracted Sentences"]
        slice_count, suv_count = count_occurrences(sentence)

        if suv_count > 2:
            delete_list.append(row)
        # single slice multiple
        elif slice_count == 1 and suv_count > 1:
            # print(f"slice_count: {slice_count} suv_count: {suv_count}")

            # likely has two descirptions
            if suv_count > 2:
                delete_list.append(row)

            contains_previous_bool = contains_previous(sentence)
            # we will keep either way but want to track them
            if contains_previous_bool:
                contains_previous_count += 1
            else:
                does_not_contain_previous += 1
                delete_list.append(row)
        # we cut all of them if they have slice count greater than 1
        elif slice_count > 1:
            multiple_extractions += 1
            delete_list.append(row)

            # these likely can be split with llm and recovered later
            if slice_count == suv_count:
                # print(sentence)
                # print(f"slice_count: {slice_count} suv_count: {suv_count}")
                suv_equals_slice += 1

            # these might be able to be recovered but likely just want to drop them
            if slice_count != suv_count:
                dropped += 1

    print(f"starting length: {len(df)}")
    print(len(delete_list))
    # Convert rows_to_delete to a DataFrame
    df_delete = pd.DataFrame(delete_list)
    print(f"df_delete_len: {len(df_delete)}")
    print(f"df_len: {len(df)}")
    # Merge with an indicator to find which rows are in both DataFrames
    if len(df_delete) > 0:
        df_merged = df.merge(df_delete, how='outer', indicator=True)
    else:
        df_merged = df
    # Filter out the rows found in df_delete
    df_final = df_merged[df_merged['_merge'] == 'left_only'].drop('_merge', axis=1)
    print(f"final length: {len(df_final)}")
    print(f"multipled extraction: {multiple_extractions}")
    print(f"contains previously: {contains_previous_count}")
    print(f"does not contain previously: {does_not_contain_previous}")
    print(f"number of suv values is same as slice: {suv_equals_slice}")
    print(f"number will be lose: {dropped}")

    return df_final