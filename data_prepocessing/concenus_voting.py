from collections import Counter
import pandas as pd
def concenus_voting_old(df):

    num_3s = 0
    num_1s = 0
    num_0s = 0
    new_df = []
    suv_cut = 0
    templated_background = 0
    for index, row in df.iterrows():

        agree_slice = row["num_agree_slice"]
        agree_suv = row["num_agree_suv"]

        if agree_slice == 3:
            num_3s += 1
            slice_num = row["Slice"]
        if agree_suv == 3:
            suv_num = row["SUV"]
            num_3s += 1
        if agree_slice == 1:
            num_1s += 1
            if row["rules_mixtral_slice"] == 1:
                slice_num = row["Slice"]
            elif row["rules_mistral_slice"] == 1:
                slice_num = row["Slice"]
            else:
                slice_num = row["AI_Slice"]
        if agree_suv == 1:
            num_1s += 1
            if row["rules_mixtral_suv"] == 1:
                suv_num = row["SUV"]
            elif row["rules_mistral_suv"] == 1:
                suv_num = row["SUV"]
            else:
                suv_num = row["AI_SUV"]
        if agree_slice == 0:
            num_0s += 1
            continue
            print(row["Extracted Sentences"])
            slice_num = row["Slice"]
            num_0s += 1
        if agree_suv == 0:
            num_0s += 1
            continue
            print(row["Extracted Sentences"])
            suv_num = row["SUV"]
            num_0s += 1

        if suv_num < 2.5:
            suv_cut += 1
            continue

        if contains_target_phrases(row["Extracted Sentences"]):
            templated_background += 1
            # print(row["Extracted Sentences"])
            # print(f"slice: {slice_num}")
            # print(f"suv_num: {suv_num}")
            continue

        new_df.append({"Petlymph": row["Petlymph"],
                       "Findings": row["Findings"],
                       "Impression": row["Impression"],
                       "Extracted Sentences": row["Extracted Sentences"],
                       "Slice": slice_num,
                       "SUV": suv_num
                       })

    # Convert list to DataFrame and save
    output_df = pd.DataFrame(new_df)
    #output_df.to_excel('concensus_slice_suv_anonymized.xlsx', index=False)
    print(f"num 3: {num_3s}")
    print(f"num 1: {num_1s}")
    print(f"num 0: {num_0s}")
    print(f"below 2.5: {suv_cut}")
    print(f"templated lang: {templated_background}")
    return output_df

def concenus_voting(df):

    num_3s = 0
    num_1s = 0
    num_0s = 0
    new_df = []
    suv_cut = 0
    templated_background = 0
    for index, row in df.iterrows():


        slice_list = []
        suv_list = []

        slice_list.append(row["mistral-7b-instructAI_Slice"])
        slice_list.append(row["mixstral-8x7b-instructAI_Slice"])
        slice_list.append(row["llama2-instructAI_Slice"])

        suv_list.append(row["mistral-7b-instructAI_SUV"])
        suv_list.append(row["mixstral-8x7b-instructAI_SUV"])
        suv_list.append(row["llama2-instructAI_SUV"])

        slice_count = Counter(slice_list)
        suv_count = Counter(suv_list)

        slice_num = max(slice_count, key=suv_count.get)
        suv_num = max(suv_count, key=suv_count.get)
        print(slice_count)
        print(suv_count)

        new_df.append({"Petlymph": row["Petlymph"],
                       "Findings": row["Findings"],
                       "Impression": row["Impression"],
                       "Extracted Sentences": row["Extracted Sentences"],
                       "Slice": slice_num,
                       "SUV": suv_num
                       })

    # Convert list to DataFrame and save
    output_df = pd.DataFrame(new_df)
    #output_df.to_excel('concensus_slice_suv_anonymized.xlsx', index=False)
    return output_df